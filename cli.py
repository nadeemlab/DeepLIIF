import os
import json
import time
import random

import click
import cv2
import torch
import numpy as np
from PIL import Image

from deepliif.data import create_dataset, transform
from deepliif.models import init_nets, infer_modalities, infer_results_for_wsi, create_model
from deepliif.util import allowed_file, Visualizer, get_information, test_diff_original_serialized, disable_batchnorm_tracking_stats
from deepliif.util.util import mkdirs, check_multi_scale
# from deepliif.util import infer_results_for_wsi
from deepliif.options import Options, print_options

import torch.distributed as dist

from packaging import version
import subprocess
import sys

import pickle


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


def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
        
@click.group()
def cli():
    """Commonly used DeepLIIF batch operations"""
    pass


@cli.command()
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
              help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_512 | unet_256 | unet_128]')
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
@click.option('--beta1', default=0.5, help='momentum term of adam')
@click.option('--lr', default=0.0002, help='initial learning rate for adam')
@click.option('--lr-policy', default='linear',
              help='learning rate policy. [linear | step | plateau | cosine]')
@click.option('--lr-decay-iters', type=int, default=50,
              help='multiply by a gamma every lr_decay_iters iterations')
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
@click.option('--local-rank', type=int, default=None, help='placeholder argument for torchrun, no need for manual setup')
@click.option('--seed', type=int, default=None, help='basic seed to be used for deterministic training, default to None (non-deterministic)')
# DeepLIIFExt params
@click.option('--seg-gen', type=bool, default=True, help='True (Translation and Segmentation), False (Only Translation).')
@click.option('--net-ds', type=str, default='n_layers',
              help='specify discriminator architecture for segmentation task [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
@click.option('--net-gs', type=str, default='unet_512',
              help='specify generator architecture for segmentation task [resnet_9blocks | resnet_6blocks | unet_512 | unet_256 | unet_128]')
@click.option('--gan-mode', type=str, default='vanilla',
              help='the type of GAN objective for translation task. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
@click.option('--gan-mode-s', type=str, default='lsgan',
              help='the type of GAN objective for segmentation task. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
def train(dataroot, name, gpu_ids, checkpoints_dir, input_nc, output_nc, ngf, ndf, net_d, net_g,
          n_layers_d, norm, init_type, init_gain, no_dropout, direction, serial_batches, num_threads,
          batch_size, load_size, crop_size, max_dataset_size, preprocess, no_flip, display_winsize, epoch, load_iter,
          verbose, lambda_l1, is_train, display_freq, display_ncols, display_id, display_server, display_env,
          display_port, update_html_freq, print_freq, no_html, save_latest_freq, save_epoch_freq, save_by_iter,
          continue_train, epoch_count, phase, lr_policy, n_epochs, n_epochs_decay, beta1, lr, lr_decay_iters,
          remote, local_rank, remote_transfer_cmd, seed, dataset_mode, padding, model, 
          modalities_no, seg_gen, net_ds, net_gs, gan_mode, gan_mode_s):
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
    
    d_params = locals()

    if gpu_ids and gpu_ids[0] == -1:
        gpu_ids = []
        
    local_rank = os.getenv('LOCAL_RANK') # DDP single node training triggered by torchrun has LOCAL_RANK
    rank = os.getenv('RANK') # if using DDP with multiple nodes, please provide global rank in env var RANK

    if len(gpu_ids) > 0:
        if local_rank is not None:
            local_rank = int(local_rank)
            torch.cuda.set_device(gpu_ids[local_rank])
            gpu_ids=[gpu_ids[local_rank]]
        else:
            torch.cuda.set_device(gpu_ids[0])

    if local_rank is not None: # LOCAL_RANK will be assigned a rank number if torchrun ddp is used
        dist.init_process_group(backend='nccl')
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
    img = Image.open(f"{dir_data_train}/{fns[0]}")
    
    num_img = img.size[0] / img.size[1]
    assert int(num_img) == num_img, f'img size {img.size[0]} / {img.size[1]} = {num_img} is not an integer'
    num_img = int(num_img)
    
    input_no = num_img - modalities_no - seg_no
    assert input_no > 0, f'inferred number of input images is {input_no}; should be greater than 0'
    d_params['input_no'] = input_no
    d_params['scale_size'] = img.size[1]

    # create a dataset given dataset_mode and other options
    # dataset = AlignedDataset(opt)

    opt = Options(d_params=d_params)
    print_options(opt, save=True)
    
    dataset = create_dataset(opt)
    # get the number of images in the dataset.
    click.echo('The number of training images = %d' % len(dataset))

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
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # print training losses and save logging information to the disk
            if total_iters % print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), losses)

            # cache our latest model every <save_latest_freq> iterations
            if total_iters % save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # cache our model every <save_epoch_freq> epochs
        if epoch % save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, n_epochs + n_epochs_decay, time.time() - epoch_start_time))
        # update learning rates at the end of every epoch.
        model.update_learning_rate()


@cli.command()
@click.option('--dataroot', required=True, type=str,
              help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
@click.option('--name', default='experiment_name',
              help='name of the experiment. It decides where to store samples and models')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--checkpoints-dir', default='./checkpoints', help='models are saved here')
@click.option('--targets-no', default=5, help='number of targets')
# model parameters
@click.option('--input-nc', default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
@click.option('--output-nc', default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
@click.option('--ngf', default=64, help='# of gen filters in the last conv layer')
@click.option('--ndf', default=64, help='# of discrim filters in the first conv layer')
@click.option('--net-d', default='n_layers',
              help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 '
                   'PatchGAN. n_layers allows you to specify the layers in the discriminator')
@click.option('--net-g', default='resnet_9blocks',
              help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_512 | unet_256 | unet_128]')
@click.option('--n-layers-d', default=4, help='only used if netD==n_layers')
@click.option('--norm', default='batch',
              help='instance normalization or batch normalization [instance | batch | none]')
@click.option('--init-type', default='normal',
              help='network initialization [normal | xavier | kaiming | orthogonal]')
@click.option('--init-gain', default=0.02, help='scaling factor for normal, xavier and orthogonal.')
@click.option('--padding-type', default='reflect', help='network padding type.')
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
@click.option('--beta1', default=0.5, help='momentum term of adam')
@click.option('--lr', default=0.0002, help='initial learning rate for adam')
@click.option('--lr-policy', default='linear',
              help='learning rate policy. [linear | step | plateau | cosine]')
@click.option('--lr-decay-iters', type=int, default=50,
              help='multiply by a gamma every lr_decay_iters iterations')
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
@click.option('--local-rank', type=int, default=None, help='placeholder argument for torchrun, no need for manual setup')
@click.option('--seed', type=int, default=None, help='basic seed to be used for deterministic training, default to None (non-deterministic)')
@click.option('--use-torchrun', type=str, default=None, help='provide torchrun options, all in one string, for example "-t3 --log_dir ~/log/ --nproc_per_node 1"; if your pytorch version is older than 1.10, torch.distributed.launch will be called instead of torchrun')
def trainlaunch(**kwargs):
    """
    A wrapper method that executes deepliif/train.py via subprocess.
    All options are the same to train() except for the additional `--use-torchrun`.
    The options received will be parsed and concatenated into a string, appended to `python deepliif/train.py ...`.

    * for developers, this at the moment can only be tested after building and installing deepliif
      because deepliif/train.py imports deepliif.xyz, and this reference is wrong until the deepliif package is installed
    """
        
    #### process options
    args = sys.argv[2:]
    
    ## args/options not needed in train,py 
    l_arg_skip = ['--use-torchrun']
    
    ## exclude the options to skip, both the option name and the value if it has
    args_final = []
    for i,arg in enumerate(args):
        if i == 0:
            if arg not in l_arg_skip:
                args_final.append(arg)
        else:
            if args[i-1] in l_arg_skip and arg.startswith('--'):
                # if the previous element is an option name to skip AND if the current element is an option name, not a value to the previous option
                args_final.append(arg)
            elif args[i-1] not in l_arg_skip and arg not in l_arg_skip:
                # if the previous element is not an option name to skip AND if the current element is not an option to remove
                args_final.append(arg)

    ## add quotes back to the input arg that had quotes, e.g., experiment name
    args_final = [f'"{arg}"' if ' ' in arg else arg for arg in args_final]
    
    ## concatenate back to a string
    options = ' '.join(args_final)

    #### locate train.py
    import deepliif
    path_train_py = deepliif.__path__[0]+'/train.py'

    #### execute train.py
    if kwargs['use_torchrun']:
        if version.parse(torch.__version__) >= version.parse('1.10.0'):
            subprocess.run(f'torchrun {kwargs["use_torchrun"]} {path_train_py} {options}',shell=True)
        else:
            subprocess.run(f'python -m torch.distributed.launch {kwargs["use_torchrun"]} {path_train_py} {options}',shell=True)
    else:
        subprocess.run(f'python {path_train_py} {options}',shell=True)




@cli.command()
@click.option('--model-dir', default='./model-server/DeepLIIF_Latest_Model', help='reads models from here')
@click.option('--output-dir', help='saves results here.')
#@click.option('--tile-size', type=int, default=None, help='tile size')
@click.option('--device', default='cpu', type=str, help='device to load model for the similarity test, either cpu or gpu')
@click.option('--verbose', default=0, type=int,help='saves results here.')
def serialize(model_dir, output_dir, device, verbose):
    """Serialize DeepLIIF models using Torchscript
    """
    #if tile_size is None:
    #    tile_size = 512
    output_dir = output_dir or model_dir
    ensure_exists(output_dir)
    
    # copy train_opt.txt to the target location
    import shutil
    if model_dir != output_dir:
        shutil.copy(f'{model_dir}/train_opt.txt',f'{output_dir}/train_opt.txt')
    
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    sample = transform(Image.new('RGB', (opt.scale_size, opt.scale_size)))
    sample = torch.cat([sample]*opt.input_no, 1)
    
    with click.progressbar(
            init_nets(model_dir, eager_mode=True, phase='test').items(),
            label='Tracing nets',
            item_show_func=lambda n: n[0] if n else n
    ) as bar:
        for name, net in bar:
            # the model should be in eval model so that there won't be randomness in tracking brought by dropout etc. layers
            # https://github.com/pytorch/pytorch/issues/23999#issuecomment-747832122
            net = net.eval()
            net = disable_batchnorm_tracking_stats(net)
            net = net.cpu()
            if name.startswith('GS'):
                traced_net = torch.jit.trace(net, torch.cat([sample, sample, sample], 1))
            else:
                traced_net = torch.jit.trace(net, sample)
                # traced_net = torch.jit.script(net)
            traced_net.save(f'{output_dir}/{name}.pt')
    
    # test: whether the original and the serialized model produces highly similar predictions
    print('testing similarity between prediction from original vs serialized models...')
    models_original = init_nets(model_dir,eager_mode=True,phase='test')
    models_serialized = init_nets(output_dir,eager_mode=False,phase='test')
    if device == 'gpu':
        sample = sample.cuda()
    else:
        sample = sample.cpu()
    for name in models_serialized.keys():
        print(name,':')
        model_original = models_original[name].cuda().eval() if device=='gpu' else models_original[name].cpu().eval()
        model_serialized = models_serialized[name].cuda() if device=='gpu' else models_serialized[name].cpu().eval()
        if name.startswith('GS'):
            test_diff_original_serialized(model_original,model_serialized,torch.cat([sample, sample, sample], 1),verbose)
        else:
            test_diff_original_serialized(model_original,model_serialized,sample,verbose)
        print('PASS')
         

@cli.command()
@click.option('--input-dir', default='./Sample_Large_Tissues/', help='reads images from here')
@click.option('--output-dir', help='saves results here.')
@click.option('--tile-size', default=None, help='tile size')
@click.option('--model-dir', default='./model-server/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--region-size', default=20000, help='Due to limits in the resources, the whole slide image cannot be processed in whole.'
                                                   'So the WSI image is read region by region. '
                                                   'This parameter specifies the size each region to be read into GPU for inferrence.')
@click.option('--eager-mode', is_flag=True, help='use eager mode (loading original models, otherwise serialized ones)')
@click.option('--color-dapi', is_flag=True, help='color dapi image to produce the same coloring as in the paper')
@click.option('--color-marker', is_flag=True, help='color marker image to produce the same coloring as in the paper')
def test(input_dir, output_dir, tile_size, model_dir, gpu_ids, region_size, eager_mode,
         color_dapi, color_marker):
    
    """Test trained models
    """
    output_dir = output_dir or input_dir
    ensure_exists(output_dir)

    image_files = [fn for fn in os.listdir(input_dir) if allowed_file(fn)]
    files = os.listdir(model_dir)
    assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_dir}'
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    opt.use_dp = False
    
    number_of_gpus_all = torch.cuda.device_count()
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        number_of_gpus = 0
        gpu_ids = [-1]
        print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')

    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(number_of_gpus_all))
    
    opt.gpu_ids = gpu_ids # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus
    
    # fix opt from old settings
    if not hasattr(opt,'modalities_no') and hasattr(opt,'targets_no'):
        opt.modalities_no = opt.targets_no - 1
        del opt.targets_no
    print_options(opt)

    with click.progressbar(
            image_files,
            label=f'Processing {len(image_files)} images',
            item_show_func=lambda fn: fn
    ) as bar:
        for filename in bar:
            if '.svs' in filename:
                start_time = time.time()
                infer_results_for_wsi(input_dir, filename, output_dir, model_dir, tile_size, region_size)
                print(time.time() - start_time)
            else:
                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
                images, scoring = infer_modalities(img, tile_size, model_dir, eager_mode, color_dapi, color_marker, opt)

                for name, i in images.items():
                    i.save(os.path.join(
                        output_dir,
                        filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
                    ))

                if scoring is not None:
                    with open(os.path.join(
                            output_dir,
                            filename.replace('.' + filename.split('.')[-1], f'.json')
                    ), 'w') as f:
                        json.dump(scoring, f, indent=2)

@cli.command()
@click.option('--input-dir', type=str, required=True, help='Path to input images')
@click.option('--output-dir', type=str, required=True, help='Path to output images')
@click.option('--validation-ratio', default=0.2,
              help='The ratio of the number of the images in the validation set to the total number of images')
def prepare_training_data(input_dir, output_dir, validation_ratio):
    """Preparing data for training

    This function, first, creates the train and validation directories inside the given dataset directory.
    Then it reads all images in the folder and saves the pairs in the train or validation directory, based on the given
    validation_ratio.
    *** for training, you need to have paired data including IHC, Hematoxylin Channel, mpIF DAPI, mpIF Lap2, mpIF
    marker, and segmentation mask in the input directory ***

    :param input_dir: Path to the input images.
    :param outputt_dir: Path to the dataset directory. The function automatically creates the train and validation
        directories inside of this directory.
    :param validation_ratio: The ratio of the number of the images in the validation set to the total number of images.
    :return:
    """
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)
    images = os.listdir(input_dir)
    for img in images:
        if 'IHC' in img:
            IHC_image = cv2.resize(cv2.imread(os.path.join(input_dir, img)), (512, 512))
            Hema_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'Hematoxylin'))), (512, 512))
            DAPI_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'DAPI'))), (512, 512))
            Lap2_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'Lap2'))), (512, 512))
            Marker_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'Marker'))), (512, 512))
            Seg_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'Seg'))), (512, 512))

            save_dir = train_dir
            if random.random() < validation_ratio:
                save_dir = val_dir
            cv2.imwrite(os.path.join(save_dir, img),
                        np.concatenate([IHC_image, Hema_image, DAPI_image, Lap2_image, Marker_image, Seg_image], 1))


@cli.command()
@click.option('--input_dir', required=True, help='path to input images')
@click.option('--output_dir', type=str, help='path to output images')
def prepare_testing_data(input_dir, dataset_dir):
    """Preparing data for testing

    This function, first, creates the test directory inside the given dataset directory.
    Then it reads all images in the folder and saves pairs in the test directory.
    *** for testing, you only need to have IHC images in the input directory ***

    :param input_dir: Path to the input images.
    :param dataset_dir: Path to the dataset directory. The function automatically creates the train and validation
        directories inside of this directory.
    :return:
    """
    test_dir = os.path.join(dataset_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    images = os.listdir(input_dir)
    for img in images:
        if 'IHC' in img:
            image = cv2.resize(cv2.imread(os.path.join(input_dir, img)), (512, 512))
            cv2.imwrite(os.path.join(test_dir, img), np.concatenate([image, image, image, image, image, image], 1))


# to load pickle file saved from gpu in a cpu environment: https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
from io import BytesIO
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


@cli.command()
@click.option('--pickle-dir', required=True, help='directory where the pickled snapshots are stored')
@click.option('--display-env', default = None, help='window name; overwrite the display-env opt from the saved pickle file')
def visualize(pickle_dir, display_env):

    path_init = os.path.join(pickle_dir,'opt.pickle')
    print(f'waiting for initialization signal from {path_init}')
    while not os.path.exists(path_init):
        time.sleep(1)

    params_opt = pickle.load(open(path_init,'rb'))
    params_opt.remote = False
    if display_env is not None:
        params_opt.display_env = display_env
    visualizer = Visualizer(params_opt)   # create a visualizer that display/save images and plots

    paths_plot = {'display_current_results':os.path.join(pickle_dir,'display_current_results.pickle'),
                'plot_current_losses':os.path.join(pickle_dir,'plot_current_losses.pickle')}

    last_modified_time = {k:0 for k in paths_plot.keys()} # initialize time

    while True:
        for method, path_plot in paths_plot.items():
            try:
                last_modified_time_plot = os.path.getmtime(path_plot)
                if last_modified_time_plot > last_modified_time[method]:
                    params_plot = CPU_Unpickler(open(path_plot,'rb')).load()
                    last_modified_time[method] = last_modified_time_plot
                    getattr(visualizer,method)(**params_plot)
                    print(f'{method} refreshed, last modified time {time.ctime(last_modified_time[method])}')
                else:
                    print(f'{method} not refreshed')
            except Exception as e:
                print(e)
        time.sleep(10)





if __name__ == '__main__':
    cli()
