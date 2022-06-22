import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import pickle


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display,
    and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses

        remote mode: training is running in a remote temporary environment, and we want to visualize training progress in a local visdom;
                     with this mode turned on, we take snapshots of variables to be passed to a visdom app as pickle files, saved under
                     <checkpoints_dir>/pickle/
        """
        self.display_id = opt.display_id
        self.use_html = opt.is_train and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.remote = opt.remote # with a remote visdom
        self.remote_transfer_cmd = opt.remote_transfer_cmd

        self.pickle_dir = os.path.join(opt.checkpoints_dir, opt.name, 'pickle') # pickled variables to be passed to a remote visdom
        self.use_multi_proc = False
        self.rank = None
        
        if os.getenv('RANK') is not None:
            self.use_multi_proc = True
            self.rank = int(os.getenv('RANK'))
        elif os.getenv('LOCAL_RANK') is not None:
            self.use_multi_proc = True
            self.rank = int(os.getenv('LOCAL_RANK'))
        print('use multiprocessing:',self.use_multi_proc)
        print('process rank:',self.rank)
        
        if self.remote:
            util.mkdirs([self.pickle_dir])
            
            fn = 'opt.pickle'            
            if (self.use_multi_proc and self.rank == 0) or (not self.use_multi_proc):
                path_source = os.path.join(self.pickle_dir,fn)
                
                # opt = {'display_id': display_id, 'is_train': is_train, 'no_html': no_html, 'display_winsize': display_winsize, 'name': name,
                #        'display_port': display_port, 'display_ncols': display_ncols, 'display_server': display_server, 'display_env': display_env,
                #        'checkpoints_dir': checkpoints_dir, 'remote': remote, 'remote_transfer_cmd': remote_transfer_cmd}
                
                pickle.dump(opt,open(path_source,'wb'))
                print(f'Remote mode, snapshot created: {fn}')
                if self.remote_transfer_cmd is not None:
                    self.remote_transfer_cmd_module = self.remote_transfer_cmd.split('.')[0]  # TODO: ask wendy if this split should happen outside this func
                    self.remote_transfer_cmd_function = self.remote_transfer_cmd.split('.')[1]
                    exec(f'from {self.remote_transfer_cmd_module} import {self.remote_transfer_cmd_function}')
                    exec(f'{self.remote_transfer_cmd_function}("{path_source}")')

        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            if not opt.remote:
                self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
                if not self.vis.check_connection():
                    self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server,
        this function will start a new server at port < self.port > """

        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result, **kwargs):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
            **kwargs - - for now, to accommodate unused params
        """
        if self.remote:
            fn = 'display_current_results.pickle'
            if (self.use_multi_proc and self.rank == 0) or (not self.use_multi_proc):
                path_source = os.path.join(self.pickle_dir,fn)
                pickle.dump({'visuals':visuals,
                             'epoch':epoch,
                             'save_result':save_result},
                             open(path_source,'wb'))
                print(f'Remote mode, snapshot refreshed: {fn}')
                if self.remote_transfer_cmd is not None:
                    exec(f'from {self.remote_transfer_cmd_module} import {self.remote_transfer_cmd_function}')
                    exec(f'{self.remote_transfer_cmd_function}("{path_source}")')
        else:
            if self.display_id > 0:  # show images in the browser using visdom
                ncols = self.ncols
                if ncols > 0:        # show all the images in one visdom panel
                    ncols = min(ncols, len(visuals))
                    h, w = next(iter(visuals.values())).shape[:2]
                    table_css = """<style>
                            table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                            table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                            </style>""" % (w, h)  # create a table css
                    # create a table of images.
                    title = self.name
                    label_html = ''
                    label_html_row = ''
                    images = []
                    idx = 0
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        label_html_row += '<td>%s</td>' % label
                        images.append(image_numpy.transpose([2, 0, 1]))
                        idx += 1
                        if idx % ncols == 0:
                            label_html += '<tr>%s</tr>' % label_html_row
                            label_html_row = ''

                    white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                    while idx % ncols != 0:
                        images.append(white_image)
                        label_html_row += '<td></td>'
                        idx += 1
                    if label_html_row != '':
                        label_html += '<tr>%s</tr>' % label_html_row

                    try:
                        self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                        padding=2, opts=dict(title=title + ' images'))
                        label_html = '<table>%s</table>' % label_html
                        self.vis.text(table_css + label_html, win=self.display_id + 2,
                                      opts=dict(title=title + ' labels'))
                    except VisdomExceptionBase:
                        self.create_visdom_connections()               

                else:     # show each image in a separate visdom panel;
                    idx = 1
                    try:
                        for label, image in visuals.items():
                            image_numpy = util.tensor2im(image)
                            self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                           win=self.display_id + idx)
                            idx += 1
                    except VisdomExceptionBase:
                        self.create_visdom_connections()
                   
            if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
                self.saved = True
                # save images to the disk
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)

                # update website
                webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
                for n in range(epoch, 0, -1):
                    webpage.add_header('epoch [%d]' % n)
                    ims, txts, links = [], [], []

                    for label, image_numpy in visuals.items():
                        image_numpy = util.tensor2im(image)
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                    webpage.add_images(ims, txts, links, width=self.win_size)
                webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):                    
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
            n_proc                -- number of processes, used to properly scale counter ratio
        """
        # counter_ratio defined in train.py: float(epoch_iter) / dataset_size
        # if having 2 processes, each process obtains 50% of the data (effective dataset_size divided by half), the effective counter ratio shall multiply by 2 to compensate that
        n_proc = int(os.getenv('WORLD_SIZE',1))
        counter_ratio = counter_ratio * n_proc
        
        if self.remote:
            fn = 'plot_current_losses.pickle'
            if (self.use_multi_proc and self.rank == 0) or (not self.use_multi_proc):
                path_source = os.path.join(self.pickle_dir,fn)
                pickle.dump({'epoch':epoch,
                             'counter_ratio':counter_ratio,
                             'losses':losses},
                             open(path_source,'wb'))
                print(f'Remote mode, snapshot refreshed: {fn}, epoch: {epoch}, counter_ratio: {counter_ratio}')
                if self.remote_transfer_cmd is not None:
                    exec(f'from {self.remote_transfer_cmd_module} import {self.remote_transfer_cmd_function}')
                    exec(f'{self.remote_transfer_cmd_function}("{path_source}")')
        else:
            if not hasattr(self, 'plot_data'):
                self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
            self.plot_data['X'].append(epoch + counter_ratio)
            self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])

            try:
                self.vis.line(
                    X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                    Y=np.array(self.plot_data['Y']),
                    opts={
                        'title': self.name + ' loss over time',
                        'legend': self.plot_data['legend'],
                        'xlabel': 'epoch',
                        'ylabel': 'loss'},
                    win=self.display_id)
            except VisdomExceptionBase:
                self.create_visdom_connections()
            

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
