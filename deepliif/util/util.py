"""This module contains simple helper functions """
from __future__ import print_function

from functools import wraps
from time import time

import cv2
import torch
import numpy as np
from PIL import Image
import os
import math

def timeit(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        print(f'{f.__name__} {time() - ts}')

        return result

    return wrap


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor_to_pil(t):
    return Image.fromarray(tensor2im(t))


def image_to_array(t):
    return np.array(t)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def imadjust(x, gamma=0.7, c=0, d=1):
    """
    Adjusting the image contrast and brightness

    :param x: Input array
    :param gamma: Gamma value
    :param c: Minimum value
    :param d: Maximum value
    :return: Adjusted image
    """
    a = x.min()
    b = x.max()
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def adjust_dapi(inferred_tile, orig_tile):
    """Adjusts the intensity of mpIF DAPI image

    Parameters:
        inferred_tile (Image) -- inferred tile image
        orig_tile (Image) -- original tile image

    """
    inferred_tile_array = image_to_array(inferred_tile)
    orig_tile_array = image_to_array(orig_tile)

    multiplier = 8 / math.log(np.max(orig_tile_array))

    if np.mean(orig_tile_array) < 200:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier * math.log(np.mean(inferred_tile_array)) / math.log(np.mean(orig_tile_array)),
                                  c=5, d=255).astype(np.uint8)

    else:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier,
                                  c=5, d=255).astype(np.uint8)
    return Image.fromarray(processed_tile)


def adjust_marker(inferred_tile, orig_tile):
    """Adjusts the intensity of mpIF marker image

    Parameters:
        inferred_tile (Image) -- inferred tile image
        orig_tile (Image) -- original tile image

    """
    inferred_tile_array = image_to_array(inferred_tile)
    orig_tile_array = image_to_array(orig_tile)

    multiplier = 10 / math.log(np.max(orig_tile_array))

    if np.mean(orig_tile_array) < 200:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier * math.log(np.std(inferred_tile_array)) / math.log(
                                      np.std(orig_tile_array)),
                                  c=5, d=255).astype(np.uint8)

    else:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier,
                                  c=5, d=255).astype(np.uint8)
    return Image.fromarray(processed_tile)


