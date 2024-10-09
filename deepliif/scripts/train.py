"""
This script is used by cli.py trainlaunch command for DDP.
Keep this train.py up-to-date with train() in cli.py!
They are EXACTLY THE SAME.
"""

import time
from deepliif.options.train_options import TrainOptions
from deepliif.data import create_dataset
from deepliif.models import create_model, postprocess
from deepliif.options import read_model_params, Options, print_options
from deepliif.util.visualizer import Visualizer
from PIL import Image
import os

import numpy as np
import random
import json
import torch
import torch.distributed as dist
from torchvision.transforms import ToPILImage

import click

def set_seed(seed=0,rank=None):
    """
    seed: basic seed
    rank: rank of the current process, using which to mutate basic seed to have a unique seed per process
    
    output: a boolean flag indicating whether deterministic training is enabled (True) or not (False)
    """
    os.environ['DEEPLIIF_SEED'] = str(seed)

    if seed is not None:
        if rank is not None:
            seed_final = seed + int(rank)
        else:
            seed_final = seed

        os.environ['PYTHONHASHSEED'] = str(seed_final)
        random.seed(seed_final)
        np.random.seed(seed_final)
        torch.manual_seed(seed_final)
        torch.cuda.manual_seed(seed_final)
        torch.cuda.manual_seed_all(seed_final)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        print(f'deterministic training, seed set to {seed_final}')
        return True
    else:
        print(f'not using deterministic training')
        return False


@click.command()
@click.option('--dataroot', required=True, type=str,
              help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
@click.option('--name', default='experiment_name',
              help='name of the experiment. It decides where to store samples and models')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--checkpoints-dir', default='./checkpoints', help='models are saved here')
@click.option('--modalities-no', default=4, type=int, help='number of targets')
# model parameters
@click.option('--model', default='DeepLIIF', help='name of model class')
@click.option('--input-nc', default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
@click.option('--output-nc', default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
@click.option('--ngf', default=64, help='# of gen filters in the last conv layer')
@click.option('--ndf', default=64, help='# of discrim filters in the first conv layer')
@click.option('--net-d', default='n_layers',
              help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 '
                   'PatchGAN. n_layers allows you to specify the layers in the discriminator')
@click.option('--net-g', default='resnet_9blocks',
              help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_512 | unet_256 | unet_128 | unet_512_attention]; to specify different arch for generators, list arch for each generator separated by comma, e.g., --net-g=resnet_9blocks,resnet_9blocks,resnet_9blocks,unet_512_attention,unet_512_attention')
@click.option('--n-layers-d', default=4, help='only used if netD==n_layers')
@click.option('--norm', default='batch',
              help='instance normalization or batch normalization [instance | batch | none]')
@click.option('--init-type', default='normal',
              help='network initialization [normal | xavier | kaiming | orthogonal]')
@click.option('--init-gain', default=0.02, help='scaling factor for normal, xavier and orthogonal.')
@click.option('--no-dropout', is_flag=True, help='no dropout for the generator')
# dataset parameters
@click.option('--direction', default='AtoB', help='AtoB or BtoA')
@click.option('--serial-batches', is_flag=True,
              help='if true, takes images in order to make batches, otherwise takes them randomly')
@click.option('--num-threads', default=4, help='# threads for loading data')
@click.option('--batch-size', default=1, help='input batch size')
@click.option('--load-size', default=512, help='scale images to this size')
@click.option('--crop-size', default=512, help='then crop to this size')
@click.option('--max-dataset-size', type=int,
              help='Maximum number of samples allowed per dataset. If the dataset directory contains more than '
                   'max_dataset_size, only a subset is loaded.')
@click.option('--preprocess', type=str,
              help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | '
                   'scale_width_and_crop | none]')
@click.option('--no-flip', is_flag=True,
              help='if specified, do not flip the images for data augmentation')
@click.option('--display-winsize', default=512, help='display window size for both visdom and HTML')
# additional parameters
@click.option('--epoch', default='latest',
              help='which epoch to load? set to latest to use latest cached model')
@click.option('--load-iter', default=0,
              help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; '
                   'otherwise, the code will load models by [epoch]')
@click.option('--verbose', is_flag=True, help='if specified, print more debugging information')
@click.option('--lambda-L1', default=100.0, help='weight for L1 loss')
@click.option('--is-train', is_flag=True, default=True)
@click.option('--continue-train', is_flag=True, help='continue training: load the latest model')
@click.option('--epoch-count', type=int, default=0,
              help='the starting  epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>')
@click.option('--phase', default='train', help='train, val, test, etc')
# training parameters
@click.option('--n-epochs', type=int, default=100,
              help='number of epochs with the initial learning rate')
@click.option('--n-epochs-decay', type=int, default=100,
              help='number of epochs to linearly decay learning rate to zero')
@click.option('--optimizer', type=str, default='adam',
              help='optimizer from torch.optim to use, applied to both generators and discriminators [adam | sgd | adamw | ...]; the current parameters however are set up for adam, so other optimziers may encounter issue')
@click.option('--beta1', default=0.5, help='momentum term of adam')
@click.option('--lr', default=0.0002, help='initial learning rate for adam')
@click.option('--lr-policy', default='linear',
              help='learning rate policy. [linear | step | plateau | cosine]')
@click.option('--lr-decay-iters', type=int, default=50,
              help='multiply by a gamma every lr_decay_iters iterations')
@click.option('--seed', type=int, default=None, help='basic seed to be used for deterministic training, default to None (non-deterministic)')
# visdom and HTML visualization parameters
@click.option('--display-freq', default=400, help='frequency of showing training results on screen')
@click.option('--display-ncols', default=4,
              help='if positive, display all images in a single visdom web panel with certain number of images per row.')
@click.option('--display-id', default=1, help='window id of the web display')
@click.option('--display-server', default="http://localhost", help='visdom server of the web display')
@click.option('--display-env', default='main',
              help='visdom display environment name (default is "main")')
@click.option('--display-port', default=8097, help='visdom port of the web display')
@click.option('--update-html-freq', default=1000, help='frequency of saving training results to html')
@click.option('--print-freq', default=100, help='frequency of showing training results on console')
@click.option('--no-html', is_flag=True,
              help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
# network saving and loading parameters
@click.option('--save-latest-freq', default=500, help='frequency of saving the latest results')
@click.option('--save-epoch-freq', default=100,
              help='frequency of saving checkpoints at the end of epochs')
@click.option('--save-by-iter', is_flag=True, help='whether saves model by iteration')
@click.option('--remote', type=bool, default=False, help='whether isolate visdom checkpoints or not; if False, you can run a separate visdom server anywhere that consumes the checkpoints')
@click.option('--remote-transfer-cmd', type=str, default=None, help='module and function to be used to transfer remote files to target storage location, for example mymodule.myfunction')
@click.option('--dataset-mode', type=str, default='aligned',
              help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
@click.option('--padding', type=str, default='zero',
              help='chooses the type of padding used by resnet generator. [reflect | zero]')
# DeepLIIFExt params
@click.option('--seg-gen', type=bool, default=True, help='True (Translation and Segmentation), False (Only Translation).')
@click.option('--net-ds', type=str, default='n_layers',
              help='specify discriminator architecture for segmentation task [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
@click.option('--net-gs', type=str, default='unet_512',
              help='specify generator architecture for segmentation task [resnet_9blocks | resnet_6blocks | unet_512 | unet_256 | unet_128 | unet_512_attention]; to specify different arch for generators, list arch for each generator separated by comma, e.g., --net-g=resnet_9blocks,resnet_9blocks,resnet_9blocks,unet_512_attention,unet_512_attention')
@click.option('--gan-mode', type=str, default='vanilla',
              help='the type of GAN objective for translation task. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
@click.option('--gan-mode-s', type=str, default='lsgan',
              help='the type of GAN objective for segmentation task. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
# DDP related arguments
@click.option('--local-rank', type=int, default=None, help='placeholder argument for torchrun, no need for manual setup')
# Others
@click.option('--with-val', is_flag=True,
              help='use validation set to evaluate model performance at the end of each epoch')
@click.option('--debug', is_flag=True,
              help='debug mode, limits the number of data points per epoch to a small value')
@click.option('--debug-data-size', default=10, type=int, help='data size per epoch used in debug mode; due to batch size, the epoch will be passed once the completed no. data points is greater than this value (e.g., for batch size 3, debug data size 10, the effective size used in training will be 12)')
def train(dataroot, name, gpu_ids, checkpoints_dir, input_nc, output_nc, ngf, ndf, net_d, net_g,
          n_layers_d, norm, init_type, init_gain, no_dropout, direction, serial_batches, num_threads,
          batch_size, load_size, crop_size, max_dataset_size, preprocess, no_flip, display_winsize, epoch, load_iter,
          verbose, lambda_l1, is_train, display_freq, display_ncols, display_id, display_server, display_env,
          display_port, update_html_freq, print_freq, no_html, save_latest_freq, save_epoch_freq, save_by_iter,
          continue_train, epoch_count, phase, lr_policy, n_epochs, n_epochs_decay, optimizer, beta1, lr, lr_decay_iters,
          remote, remote_transfer_cmd, seed, dataset_mode, padding, model, 
          modalities_no, seg_gen, net_ds, net_gs, gan_mode, gan_mode_s, local_rank, with_val, debug, debug_data_size):
    """General-purpose training script for multi-task image-to-image translation.

    This script works for various models (with option '--model': e.g., DeepLIIF) and
    different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
    You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

    It first creates model, dataset, and visualizer given the option.
    It then does standard network training. During the training, it also visualize/save the images, print/save the loss
    plot, and save models.The script supports continue/resume training.
    Use '--continue_train' to resume your previous training.
    """
    assert model in ['DeepLIIF','DeepLIIFExt','SDG'], f'model class {model} is not implemented'
    if model == 'DeepLIIF':
        seg_no = 1
    elif model == 'DeepLIIFExt':
        if seg_gen:
            seg_no = modalities_no
        else:
            seg_no = 0
    else: # SDG
        seg_no = 0
        seg_gen = False
    
    if optimizer != 'adam':
        print(f'Optimizer torch.optim.{optimizer} is not tested. Be careful about the parameters of the optimizer.')
    
    d_params = locals()

    if gpu_ids and gpu_ids[0] == -1:
        gpu_ids = []
        
    local_rank = os.getenv('LOCAL_RANK') # DDP single node training triggered by torchrun has LOCAL_RANK
    rank = os.getenv('RANK') # if using DDP with multiple nodes, please provide global rank in env var RANK

    if len(gpu_ids) > 0:
        if local_rank is not None:
            local_rank = int(local_rank)
            torch.cuda.set_device(gpu_ids[local_rank])
            gpu_ids=[local_rank]
        else:
            torch.cuda.set_device(gpu_ids[0])

    if local_rank is not None: # LOCAL_RANK will be assigned a rank number if torchrun ddp is used
        dist.init_process_group(backend="nccl", rank=int(os.environ['RANK']), world_size=int(os.environ['WORLD_SIZE']))
        print('local rank:',local_rank)
        flag_deterministic = set_seed(seed,local_rank)
    elif rank is not None:
        flag_deterministic = set_seed(seed, rank)
    else:
        flag_deterministic = set_seed(seed)

    if flag_deterministic:
        d_params['padding'] = 'zero'
        print('padding type is forced to zero padding, because neither refection pad2d or replication pad2d has a deterministic implementation')

    # infer number of input images
    dir_data_train = dataroot + '/train'
    fns = os.listdir(dir_data_train)
    fns = [x for x in fns if x.endswith('.png')]
    print(f'{len(fns)} images found')
    img = Image.open(f"{dir_data_train}/{fns[0]}")
    print(f'image shape:',img.size)
    
    num_img = img.size[0] / img.size[1]
    assert int(num_img) == num_img, f'img size {img.size[0]} / {img.size[1]} = {num_img} is not an integer'
    num_img = int(num_img)
    
    input_no = num_img - modalities_no - seg_no
    assert input_no > 0, f'inferred number of input images is {input_no} (modalities_no {modalities_no}, seg_no {seg_no}); should be greater than 0'
    d_params['input_no'] = input_no
    d_params['scale_size'] = img.size[1]
    d_params['gpu_ids'] = gpu_ids
    
    # update generator arch
    net_g = net_g.split(',')
    assert len(net_g) in [1,modalities_no], f'net_g should contain either 1 architecture for all translation generators or the same number of architectures as the number of translation generators ({modalities_no})'
    if len(net_g) == 1:
      net_g = net_g*modalities_no
      
    
    net_gs = net_gs.split(',')
    assert len(net_gs) in [1,seg_no], f'net_gs should contain either 1 architecture for all segmentation generators or the same number of architectures as the number of segmentation generators ({seg_no})'
    if len(net_gs) == 1 and model == 'DeepLIIF':
        net_gs = net_gs*(modalities_no + seg_no)
    elif len(net_gs) == 1:
        net_gs = net_gs*seg_no
    
    d_params['net_g'] = net_g
    d_params['net_gs'] = net_gs

    # create a dataset given dataset_mode and other options
    # dataset = AlignedDataset(opt)

    opt = Options(d_params=d_params)
    print_options(opt, save=True)
    
    # set dir for train and val
    dataset = create_dataset(opt)

    # get the number of images in the dataset.
    click.echo('The number of training images = %d' % len(dataset))
    
    if with_val:
        dataset_val = create_dataset(opt,phase='val')
        data_val = [batch for batch in dataset_val]
        click.echo('The number of validation images = %d' % len(dataset_val))
        
        if model in ['DeepLIIF']: 
            metrics_val = json.load(open(os.path.join(dataset_val.dataset.dir_AB,'metrics.json')))

    # create a model given model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)

    # create a visualizer that display/save images and plots
    visualizer = Visualizer(opt)
    # the total number of training iterations
    total_iters = 0

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(epoch_count, n_epochs + n_epochs_decay + 1):
        # timer for entire epoch
        epoch_start_time = time.time()
        # timer for data loading per iteration
        iter_data_time = time.time()
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0
        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        visualizer.reset()

        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        if local_rank is not None or os.getenv('RANK') is not None: # if DDP is used, either on one node or multi nodes
            if not serial_batches: # if we want randome order in mini batches
                dataset.sampler.set_epoch(epoch)

        # inner loop within one epoch
        for i, data in enumerate(dataset):
            
            # timer for computation per iteration
            iter_start_time = time.time()
            if total_iters % print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += batch_size
            epoch_iter += batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

            # display images on visdom and save images to a HTML file
            if total_iters % display_freq == 0:
                save_result = total_iters % update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results({**model.get_current_visuals()}, epoch, save_result)

            # print training losses and save logging information to the disk
            if total_iters % print_freq == 0:
                losses = model.get_current_losses() # get training losses
                t_comp = (time.time() - iter_start_time) / batch_size
                visualizer.print_current_losses(epoch, epoch_iter, {**losses}, t_comp, t_data)
                if display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), {**losses})

            # cache our latest model every <save_latest_freq> iterations
            if total_iters % save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if save_by_iter else 'latest'
                model.save_networks(save_suffix)


            iter_data_time = time.time()
            if debug and epoch_iter >= debug_data_size:
                print(f'debug mode, epoch {epoch} stopped at epoch iter {epoch_iter} (>= {debug_data_size})')
                break

        # cache our model every <save_epoch_freq> epochs
        if epoch % save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        
        
        # validation loss and metrics calculation
        if with_val:
            losses = model.get_current_losses() # get training losses to print
            
            model.eval()
            l_losses_val = []
            l_metrics_val = []
            
            # for each val image, calculate validation loss and cell count metrics
            for j, data_val_batch in enumerate(data_val):
                # batch size is effectively 1 for validation
                model.set_input(data_val_batch)
                model.calculate_losses() # this does not optimize parameters
                visuals = model.get_current_visuals()  # get image results
                
                # val losses
                losses_val_batch = model.get_current_losses()
                l_losses_val += [(k,v) for k,v in losses_val_batch.items()]
                
                # calculate cell count metrics
                if type(model).__name__ == 'DeepLIIFModel':
                    l_seg_names = ['fake_B_5']
                    assert l_seg_names[0] in visuals.keys(), f'Cannot find {l_seg_names[0]} in generated image names ({list(visuals.keys())})'
                    seg_mod_suffix = l_seg_names[0].split('_')[-1]
                    l_seg_names += [x for x in visuals.keys() if x.startswith('fake') and x.split('_')[-1].startswith(seg_mod_suffix) and x != l_seg_names[0]]
                    # print(f'Running postprocess for {len(l_seg_names)} generated images ({l_seg_names})')
        
                    img_name_current = data_val_batch['A_paths'][0].split('/')[-1][:-4] # remove .png
                    metrics_gt = metrics_val[img_name_current]
                    
                    for seg_name in l_seg_names:
                        images = {'Seg':ToPILImage()((visuals[seg_name][0].cpu()+1)/2),
                                  #'Marker':ToPILImage()((visuals['fake_B_4'][0].cpu()+1)/2)
                                  }
                        _, scoring = postprocess(ToPILImage()((data['A'][0]+1)/2), images, opt.scale_size, opt.model)
                        
                        for k,v in scoring.items():
                            if k.startswith('num') or k.startswith('percent'):
                                # to calculate the rmse, here we calculate (x_pred - x_true) ** 2
                                l_metrics_val.append((k+'_'+seg_name,(v - metrics_gt[k])**2))
                    
                if debug and epoch_iter >= debug_data_size:
                    print(f'debug mode, epoch {epoch} stopped at epoch iter {epoch_iter} (>= {debug_data_size})')
                    break
                    
            d_losses_val = {k+'_val':0 for k in losses_val_batch.keys()}
            for k,v in l_losses_val:
                d_losses_val[k+'_val'] += v
            for k in d_losses_val:
                d_losses_val[k] = d_losses_val[k] / len(data_val)
            
            d_metrics_val = {}
            for k,v in l_metrics_val:
                try:
                    d_metrics_val[k] += v
                except:
                    d_metrics_val[k] = v
            for k in d_metrics_val:
                # to calculate the rmse, this is the second part, where d_metrics_val[k] now represents sum((x_pred - x_true) ** 2)
                d_metrics_val[k] = np.sqrt(d_metrics_val[k] / len(data_val))
            
            
            model.train()
            t_comp = (time.time() - iter_start_time) / batch_size
            visualizer.print_current_losses(epoch, epoch_iter, {**losses,**d_losses_val, **d_metrics_val}, t_comp, t_data)
            if display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), {**losses,**d_losses_val,**d_metrics_val})


        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, n_epochs + n_epochs_decay, time.time() - epoch_start_time))
        # update learning rates at the end of every epoch.
        model.update_learning_rate()


if __name__ == '__main__':
    train()
