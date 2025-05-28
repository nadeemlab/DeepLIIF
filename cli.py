import os
import json
import time
import random

import click
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

from deepliif.data import create_dataset, transform
from deepliif.models import init_nets, infer_modalities, infer_results_for_wsi, create_model, postprocess
from deepliif.util import allowed_file, Visualizer, get_information, test_diff_original_serialized, disable_batchnorm_tracking_stats
from deepliif.util.util import mkdirs
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
@click.option('--seg-weights', default='', type=str, help='weights used to aggregate modality images for the final segmentation image; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.25,0.15,0.25,0.1,0.25')
@click.option('--loss-weights-g', default='', type=str, help='weights used to aggregate modality-wise losses for the final loss; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.2,0.2,0.2,0.2,0.2')
@click.option('--loss-weights-d', default='', type=str, help='weights used to aggregate modality-wise losses for the final loss; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.2,0.2,0.2,0.2,0.2')
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
@click.option('--upsample', default='convtranspose', help='use upsampling instead of convtranspose [convtranspose | resize_conv | pixel_shuffle]')
@click.option('--label-smoothing', type=float,default=0.0, help='label smoothing factor to prevent the discriminator from being too confident')
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
#@click.option('--lr', default=0.0002, help='initial learning rate for adam')
@click.option('--lr-g', default=0.0002, help='initial learning rate for generator adam optimizer')
@click.option('--lr-d', default=0.0002, help='initial learning rate for discriminator adam optimizer')
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
          n_layers_d, norm, init_type, init_gain, no_dropout, upsample, label_smoothing, direction, serial_batches, num_threads,
          batch_size, load_size, crop_size, max_dataset_size, preprocess, no_flip, display_winsize, epoch, load_iter,
          verbose, lambda_l1, is_train, display_freq, display_ncols, display_id, display_server, display_env,
          display_port, update_html_freq, print_freq, no_html, save_latest_freq, save_epoch_freq, save_by_iter,
          continue_train, epoch_count, phase, lr_policy, n_epochs, n_epochs_decay, optimizer, beta1, lr_g, lr_d, lr_decay_iters,
          remote, remote_transfer_cmd, seed, dataset_mode, padding, model, seg_weights, loss_weights_g, loss_weights_d,
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
    assert model in ['DeepLIIF','DeepLIIFExt','SDG','CycleGAN'], f'model class {model} is not implemented'
    if model == 'DeepLIIF':
        seg_no = 1
    elif model == 'DeepLIIFExt':
        if seg_gen:
            seg_no = modalities_no
        else:
            seg_no = 0
    else: # SDG, CycleGAN
        seg_no = 0
        seg_gen = False
    
    if model == 'CycleGAN':
        dataset_mode = "unaligned"
    
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
    
    
    if dataset_mode == 'unaligned':
        dir_data_train = dataroot + '/trainA'
        fns = os.listdir(dir_data_train)
        fns = [x for x in fns if x.endswith('.png')]
        print(f'{len(fns)} images found in trainA')
        img = Image.open(f"{dir_data_train}/{fns[0]}")
        print(f'image shape:',img.size)

        for i in range(1, modalities_no + 1):
            dir_data_train = dataroot + f'/trainB{i}'
            fns = os.listdir(dir_data_train)
            fns = [x for x in fns if x.endswith('.png')]
            print(f'{len(fns)} images found in trainB{i}')
            img = Image.open(f"{dir_data_train}/{fns[0]}")
            print(f'image shape:',img.size)
        
        input_no = 1
        num_img = None
        
        lambda_identity = 0
        pool_size = 50 # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/train_cyclegan.sh
        
    else:
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
        
        pool_size = 0
        
    d_params['input_no'] = input_no
    d_params['scale_size'] = img.size[1]
    d_params['gpu_ids'] = gpu_ids
    d_params['lambda_identity'] = 0
    d_params['pool_size'] = pool_size
    
    
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
    
    # check seg weights and loss weights
    if len(d_params['seg_weights']) == 0:
        seg_weights = [0.25,0.15,0.25,0.1,0.25] if d_params['model'] == 'DeepLIIF' else [1 / modalities_no] * modalities_no
    else:
        seg_weights = [float(x) for x in seg_weights.split(',')]
    
    if len(d_params['loss_weights_g']) == 0:
        loss_weights_g = [0.2]*5 if d_params['model'] == 'DeepLIIF' else [1 / modalities_no] * modalities_no
    else:
        loss_weights_g = [float(x) for x in loss_weights_g.split(',')]
    
    if len(d_params['loss_weights_d']) == 0:
        loss_weights_d = [0.2]*5 if d_params['model'] == 'DeepLIIF' else [1 / modalities_no] * modalities_no
    else:
        loss_weights_d = [float(x) for x in loss_weights_d.split(',')]
    
    assert sum(seg_weights) == 1, 'seg weights should add up to 1'
    assert sum(loss_weights_g) == 1, 'loss weights g should add up to 1'
    assert sum(loss_weights_d) == 1, 'loss weights d should add up to 1'
    
    if model == 'DeepLIIF':
        # +1 because input becomes an additional modality used in generating the final segmentation
        assert len(seg_weights) == modalities_no+1, 'seg weights should have the same number of elements as number of modalities to be generated'
        assert len(loss_weights_g) == modalities_no+1, 'loss weights g should have the same number of elements as number of modalities to be generated'
        assert len(loss_weights_d) == modalities_no+1, 'loss weights d should have the same number of elements as number of modalities to be generated'

    else:
        assert len(seg_weights) == modalities_no, 'seg weights should have the same number of elements as number of modalities to be generated'
        assert len(loss_weights_g) == modalities_no, 'loss weights g should have the same number of elements as number of modalities to be generated'
        assert len(loss_weights_d) == modalities_no, 'loss weights d should have the same number of elements as number of modalities to be generated'

    d_params['seg_weights'] = seg_weights
    d_params['loss_G_weights'] = loss_weights_g
    d_params['loss_D_weights'] = loss_weights_d
    
    del d_params['loss_weights_g']
    del d_params['loss_weights_d']
    
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
@click.option('--seg-weights', default='', type=str, help='weights used to aggregate modality images for the final segmentation image; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.25,0.15,0.25,0.1,0.25')
@click.option('--loss-weights-g', default='', type=str, help='weights used to aggregate modality-wise losses for the final loss; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.2,0.2,0.2,0.2,0.2')
@click.option('--loss-weights-d', default='', type=str, help='weights used to aggregate modality-wise losses for the final loss; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.2,0.2,0.2,0.2,0.2')
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
@click.option('--upsample', default='convtranspose', help='use upsampling instead of convtranspose [convtranspose | resize_conv | pixel_shuffle]')
@click.option('--label-smoothing', type=float,default=0.0, help='label smoothing factor to prevent the discriminator from being too confident')
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
#@click.option('--lr', default=0.0002, help='initial learning rate for adam')
@click.option('--lr-g', default=0.0002, help='initial learning rate for generator adam optimizer')
@click.option('--lr-d', default=0.0002, help='initial learning rate for discriminator adam optimizer')
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
# trainlaunch DDP related arguments
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
    path_train_py = deepliif.__path__[0]+'/scripts/train.py'
    
    #### find out GPUs to use
    gpu_ids = [args_final[i+1] for i,v in enumerate(args_final) if v=='--gpu-ids']
    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    
    if len(gpu_ids) > 0:
        opt_env = f"CUDA_VISIBLE_DEVICES=\"{','.join(gpu_ids)}\""
    else:
        opt_env = ''

    #### execute train.py
    if kwargs['use_torchrun']:
        if version.parse(torch.__version__) >= version.parse('1.10.0'):
            cmd = f'{opt_env} torchrun {kwargs["use_torchrun"]} {path_train_py} {options}'
        else:
            cmd = f'{opt_env} python -m torch.distributed.launch {kwargs["use_torchrun"]} {path_train_py} {options}'
    else:
        cmd = f'{opt_env} python {path_train_py} {options}'
    
    print('Executing command:',cmd)
    subprocess.run(cmd,shell=True)




@cli.command()
@click.option('--model-dir', default='./model-server/DeepLIIF_Latest_Model', help='reads models from here')
@click.option('--output-dir', help='saves results here.')
#@click.option('--tile-size', type=int, default=None, help='tile size')
@click.option('--device', default='cpu', type=str, help='device to run serialization as well as load model for the similarity test, either cpu or gpu')
@click.option('--epoch', default='latest', type=str, help='epoch to load and serialize')
@click.option('--verbose', default=0, type=int,help='saves results here.')
def serialize(model_dir, output_dir, device, epoch, verbose):
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
    
    # load and update opt for serialization
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    opt.epoch = epoch
    if device == 'gpu':
        opt.gpu_ids = [0] # use gpu 0, in case training was done on larger machines
    else:
        opt.gpu_ids = [] # use cpu
    
    print_options(opt)
    sample = transform(Image.new('RGB', (opt.scale_size, opt.scale_size)))
    sample = torch.cat([sample]*opt.input_no, 1)
    
    with click.progressbar(
            init_nets(model_dir, eager_mode=True, opt=opt, phase='test').items(),
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
    models_original = init_nets(model_dir,eager_mode=True,opt=opt,phase='test')
    models_serialized = init_nets(output_dir,eager_mode=False,opt=opt,phase='test')
    
    if device == 'gpu':
        sample = sample.cuda()
    else:
        sample = sample.cpu()
    for name in models_serialized.keys():
        print(name,':')
        model_original = models_original[name].cuda().eval() if device=='gpu' else models_original[name].cpu().eval()
        model_serialized = models_serialized[name].cuda().eval() if device=='gpu' else models_serialized[name].cpu().eval()
        if name.startswith('GS'):
            test_diff_original_serialized(model_original,model_serialized,torch.cat([sample, sample, sample], 1),verbose)
        else:
            test_diff_original_serialized(model_original,model_serialized,sample,verbose)
        print('PASS')
         

@cli.command()
@click.option('--input-dir', default='./Sample_Large_Tissues/', help='reads images from here')
@click.option('--output-dir', help='saves results here.')
@click.option('--tile-size', type=click.IntRange(min=1, max=None), required=True, help='tile size')
@click.option('--model-dir', default='./model-server/DeepLIIF_Latest_Model/', help='load models from here.')
@click.option('--filename-pattern', default='*', help='run inference on files of which the name matches the pattern.')
@click.option('--gpu-ids', type=int, multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--region-size', default=20000, help='Due to limits in the resources, the whole slide image cannot be processed in whole.'
                                                   'So the WSI image is read region by region. '
                                                   'This parameter specifies the size each region to be read into GPU for inferrence.')
@click.option('--eager-mode', is_flag=True, help='use eager mode (loading original models, otherwise serialized ones)')
@click.option('--epoch', default='latest',
              help='for eager mode, which epoch to load? set to latest to use latest cached model')
@click.option('--seg-intermediate', is_flag=True, help='also save intermediate segmentation images (currently only applies to DeepLIIF model)')
@click.option('--seg-only', is_flag=True, help='save only the final segmentation image (currently only applies to DeepLIIF model); overwrites --seg-intermediate')
@click.option('--color-dapi', is_flag=True, help='color dapi image to produce the same coloring as in the paper')
@click.option('--color-marker', is_flag=True, help='color marker image to produce the same coloring as in the paper')
@click.option('--BtoA', is_flag=True, help='for models trained with unaligned dataset, this flag instructs to load generatorB instead of generatorA')
def test(input_dir, output_dir, tile_size, model_dir, filename_pattern, gpu_ids, region_size, eager_mode, epoch,
         seg_intermediate, seg_only, color_dapi, color_marker, btoa):
    
    """Test trained models
    """
    output_dir = output_dir or input_dir
    ensure_exists(output_dir)
    
    if seg_intermediate and seg_only:
        seg_intermediate = False

    if filename_pattern == '*':
        print('use all alowed files')
        image_files = [fn for fn in os.listdir(input_dir) if allowed_file(fn)]
    else:
        import glob
        print('match files using filename pattern',filename_pattern)
        image_files = [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, filename_pattern))]
    print(len(image_files),'image files')
    
    files = os.listdir(model_dir)
    assert 'train_opt.txt' in files, f'file train_opt.txt is missing from model directory {model_dir}'
    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    opt.use_dp = False
    opt.BtoA = btoa
    opt.epoch = epoch
    
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
                infer_results_for_wsi(input_dir, filename, output_dir, model_dir, tile_size, region_size, seg_only=seg_only)
                print(time.time() - start_time)
            else:
                img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
                images, scoring = infer_modalities(img, tile_size, model_dir, eager_mode, color_dapi, color_marker, opt, return_seg_intermediate=seg_intermediate, seg_only=seg_only)

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
    # tensor float 32 is available on nvidia ampere cards (e.g, a100, a40) and provides better performance at the cost of a bit lower precision
    # in 1.7-1.11, pytorch by default enables tf32 when possible 
    # currently convolutions still uses tf32 by default while matmul does not and needs to be enabled manually
    # see this issue for a discussion: https://github.com/pytorch/pytorch/issues/67384
    torch.backends.cuda.matmul.allow_tf32 = True
    cli()
