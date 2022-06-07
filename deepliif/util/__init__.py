"""This package includes a miscellaneous collection of useful helper functions."""
import os
import collections

import torch
import numpy as np
from PIL import Image, ImageOps

from skimage.filters import threshold_multiotsu

from .visualizer import Visualizer

# Postfixes not to consider for segmentation
from ..postprocessing import imadjust
import cv2

excluding_names = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg', 'Marked', 'SegRefined', 'SegOverlaid', 'Marker', 'Lap2']
# Image extensions to consider
image_extensions = ['.png', '.jpg', '.tif', '.jpeg']


def allowed_file(filename):
    name, extension = os.path.splitext(filename)
    image_type = name.split('_')[-1]  # Read image type

    return extension in image_extensions and image_type not in excluding_names


def chunker(iterable, size):
    for i in range(size):
        yield iterable[i::size]


Tile = collections.namedtuple('Tile', 'i, j, img')


def output_size(img, tile_size):
    return (max(round(img.width / tile_size) * tile_size, tile_size),
            max(round(img.height / tile_size) * tile_size, tile_size))


def generate_tiles(img, tile_size, overlap_size, mean_background_val):
    img = img.resize(output_size(img, tile_size))
    # Adding borders with size of given overlap around the whole slide image
    img = ImageOps.expand(img, border=overlap_size, fill=tuple(mean_background_val))
    rows = int(img.height / tile_size)  # Number of tiles in the row
    cols = int(img.width / tile_size)  # Number of tiles in the column

    # Generating the tiles
    for i in range(cols):
        for j in range(rows):
            yield Tile(j, i, img.crop((
                i * tile_size, j * tile_size,
                i * tile_size + tile_size + 2 * overlap_size,
                j * tile_size + tile_size + 2 * overlap_size
            )))


def stitch(tiles, tile_size, overlap_size):
    rows = max(t.i for t in tiles) + 1
    cols = max(t.j for t in tiles) + 1

    width = tile_size * cols
    height = tile_size * rows

    new_im = Image.new('RGB', (width, height))

    for t in tiles:
        img = t.img.resize((tile_size + 2 * overlap_size, tile_size + 2 * overlap_size))
        img = img.crop((overlap_size, overlap_size, overlap_size + tile_size, overlap_size + tile_size))

        new_im.paste(img, (t.j * tile_size, t.i * tile_size))

    return new_im


def calculate_background_mean_value(img):
    img = cv2.fastNlMeansDenoisingColored(np.array(img), None, 10, 10, 7, 21)
    img = np.array(img, dtype=float)
    thresh_val = 15
    sub_0_1 = np.abs(np.subtract(img[:, :, 0], img[:, :, 1]))
    sub_0_2 = np.abs(np.subtract(img[:, :, 0], img[:, :, 2]))
    sub_1_2 = np.abs(np.subtract(img[:, :, 1], img[:, :, 2]))
    can_be_back = np.logical_and(np.logical_and(sub_0_1 < thresh_val, sub_0_2 < thresh_val), sub_1_2 < thresh_val)
    return np.mean(img[can_be_back], axis=0).astype(np.uint8)


def calculate_background_area(img):
    total_pixel_no = img.width * img.height
    img = img.convert('RGB')
    img = cv2.fastNlMeansDenoisingColored(np.array(img), None, 10, 10, 7, 21)
    img = np.array(img, dtype=float)
    thresh_val = 15
    sub_0_1 = np.abs(np.subtract(img[:, :, 0], img[:, :, 1]))
    sub_0_2 = np.abs(np.subtract(img[:, :, 0], img[:, :, 2]))
    sub_1_2 = np.abs(np.subtract(img[:, :, 1], img[:, :, 2]))
    can_be_back = np.logical_and(np.logical_and(sub_0_1 < thresh_val, sub_0_2 < thresh_val), sub_1_2 < thresh_val)
    can_be_fore = np.logical_and(np.subtract(img[:, :, 2], img[:, :, 0]) > 5,
                                 np.subtract(img[:, :, 2], img[:, :, 1]) > 5)
    back_pixel_no = np.count_nonzero(np.logical_and(can_be_back, 1 - can_be_fore))
    return int(back_pixel_no / total_pixel_no * 100) if total_pixel_no > 0 else 0


def adjust_background_tile(img):
    image = img.copy()
    image = np.array(image.convert('L'))
    unique_vals = np.unique(image)
    if len(unique_vals) > 3:
        thresholds = threshold_multiotsu(image)
        regions = np.digitize(image, bins=thresholds)
        image = imadjust(image, np.mean(image[regions == 0]) / 20, 0, 255)
    image = Image.fromarray(image).convert('RGB')
    # print(np.mean(image[regions == 0]))
    # print(np.mean(image[regions == 1]))
    return image
