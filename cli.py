import os
import json
import time
import random

import click
import cv2
import torch
import numpy as np
from PIL import Image

from deepliif.data import create_dataset, AlignedDataset, transform
from deepliif.models import inference, compute_overlap, init_nets, DeepLIIFModel
from deepliif.util import allowed_file, Visualizer


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
@click.option('--gpu-ids', default=-1, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
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
def train(dataroot, name, gpu_ids, checkpoints_dir, targets_no, input_nc, output_nc, ngf, ndf, net_d, net_g,
          n_layers_d, norm, init_type, init_gain, no_dropout, direction, serial_batches, num_threads,
          batch_size, load_size, crop_size, max_dataset_size, preprocess, no_flip, display_winsize, epoch, load_iter,
          verbose, lambda_l1, is_train, display_freq, display_ncols, display_id, display_server, display_env,
          display_port, update_html_freq, print_freq, no_html, save_latest_freq, save_epoch_freq, save_by_iter,
          continue_train, epoch_count, phase, lr_policy, n_epochs, n_epochs_decay, beta1, lr, lr_decay_iters):
    """General-purpose training script for multi-task image-to-image translation.

    This script works for various models (with option '--model': e.g., DeepLIIF) and
    different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
    You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

    It first creates model, dataset, and visualizer given the option.
    It then does standard network training. During the training, it also visualize/save the images, print/save the loss
    plot, and save models.The script supports continue/resume training.
    Use '--continue_train' to resume your previous training.
    """

    # create a dataset given dataset_mode and other options
    dataset = AlignedDataset(dataroot, load_size, crop_size, input_nc, output_nc, direction, targets_no, preprocess,
                             no_flip, phase, max_dataset_size)

    dataset = create_dataset(dataset, batch_size, serial_batches, num_threads, max_dataset_size)
    # get the number of images in the dataset.
    click.echo('The number of training images = %d' % len(dataset))

    # create a model given model and other options
    model = DeepLIIFModel(gpu_ids, is_train, checkpoints_dir, name, preprocess, targets_no, input_nc, output_nc, ngf,
                          net_g, norm, no_dropout, init_type, init_gain, ndf, net_d, n_layers_d, lr, beta1, lambda_l1,
                          lr_policy)
    # regular setup: load and print networks; create schedulers
    model.setup(lr_policy, epoch_count, n_epochs, n_epochs_decay, lr_decay_iters, continue_train, load_iter, epoch,
                verbose)

    # create a visualizer that display/save images and plots
    visualizer = Visualizer(display_id, is_train, no_html, display_winsize, name, display_port, display_ncols,
                            display_server, display_env, checkpoints_dir)
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
@click.option('--input-dir', default='./Sample_Large_Tissues/', help='reads images from here')
@click.option('--output-dir', help='saves results here.')
@click.option('--tile-size', default=512, help='tile size')
def test(input_dir, output_dir, tile_size):
    """Test trained models
    """
    output_dir = output_dir or input_dir
    ensure_exists(output_dir)

    image_files = [fn for fn in os.listdir(input_dir) if allowed_file(fn)]

    with click.progressbar(
            image_files,
            label=f'Processing {len(image_files)} images',
            item_show_func=lambda fn: fn
    ) as bar:
        for filename in bar:
            img = Image.open(os.path.join(input_dir, filename))

            images, scoring = inference(
                img,
                tile_size=tile_size,
                overlap_size=compute_overlap(img.size, tile_size)
            )

            for name, i in images.items():
                i.save(os.path.join(
                    output_dir,
                    filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
                ))

            with open(os.path.join(
                    output_dir,
                    filename.replace('.' + filename.split('.')[-1], f'.json')
            ), 'w') as f:
                json.dump(scoring, f, indent=2)


@cli.command()
@click.option('--models-dir', default='./model-server/DeepLIIF_Latest_Model', help='reads models from here')
@click.option('--output-dir', help='saves results here.')
def serialize(models_dir, output_dir):
    """Serialize DeepLIIF models using Torchscript
    """
    output_dir = output_dir or models_dir

    with Image.open('./Sample_Large_Tissues/ROI_7.png') as img:
        sample = transform(img.resize((512, 512)))

    with click.progressbar(
            init_nets(models_dir, eager_mode=True).items(),
            label='Tracing nets',
            item_show_func=lambda n: n[0] if n else n
    ) as bar:
        for name, net in bar:
            traced_net = torch.jit.trace(net, sample)
            traced_net.save(f'{output_dir}/{name}.pt')


@cli.command()
@click.option('--input_dir', required=True, help='Path to input images')
@click.option('--output_dir', type=str, help='Path to output images')
@click.option('--validation_ratio', default=0.2,
              help='The ratio of the number of the images in the validation set to the total number of images')
def prepare_training_data(input_dir, dataset_dir, validation_ratio):
    """Preparing data for training

    This function, first, creates the train and validation directories inside the given dataset directory.
    Then it reads all images in the folder and saves the pairs in the train or validation directory, based on the given
    validation_ratio.
    *** for training, you need to have paired data including IHC, Hematoxylin Channel, mpIF DAPI, mpIF Lap2, mpIF
    marker, and segmentation mask in the input directory ***

    :param input_dir: Path to the input images.
    :param dataset_dir: Path to the dataset directory. The function automatically creates the train and validation
        directories inside of this directory.
    :param validation_ratio: The ratio of the number of the images in the validation set to the total number of images.
    :return:
    """
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
        os.mkdir(val_dir)
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


if __name__ == '__main__':
    cli()
