"""This package includes a miscellaneous collection of useful helper functions."""
import os
import collections

import torch
import numpy as np
from PIL import Image, ImageOps
import javabridge
import bioformats

from .visualizer import Visualizer

# Postfixes not to consider for segmentation
excluding_names = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg', 'Marked', 'SegRefined', 'SegOverlaid', 'Marker', 'Lap2']
# Image extensions to consider
image_extensions = ['.png', '.jpg', '.tif', '.czi']


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


def generate_tiles(img, tile_size, overlap_size):
    img = img.resize(output_size(img, tile_size))
    # Adding borders with size of given overlap around the whole slide image
    img = ImageOps.expand(img, border=overlap_size, fill='white')
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

def read_input_image(input_addr):
    if input_addr.endswith('.czi'):
        img = read_czi_file(input_addr)
    else:
        img = Image.open(input_addr)
    return img

def read_czi_file(input_addr):
    javabridge.start_vm(class_path=bioformats.JARS)

    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    filename = input_addr
    metadata = bioformats.get_omexml_metadata(filename)
    omexml = bioformats.OMEXML(metadata)

    print('SizeX:', omexml.image().Pixels.SizeX)
    print('SizeY:', omexml.image().Pixels.SizeY)
    print('SizeZ:', omexml.image().Pixels.SizeZ)
    print('SizeC:', omexml.image().Pixels.SizeC)
    print('SizeT:', omexml.image().Pixels.SizeT)
    print('PixelType:', omexml.image().Pixels.PixelType)

    if omexml.image().Pixels.PixelType == 'uint8':
        pixels = bioformats.load_image(filename, rescale=False)
    else:
        pixels = bioformats.load_image(filename, rescale=True)
        pixels *= 255
        pixels = np.rint(pixels).astype(np.uint8)

    javabridge.kill_vm()

    image = Image.fromarray(pixels)
    return image
