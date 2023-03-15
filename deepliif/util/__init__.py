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

import bioformats
import javabridge
import bioformats.omexml as ome
import tifffile as tf

import pickle
import sys

excluding_names = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg', 'Marked', 'SegRefined', 'SegOverlaid', 'Marker', 'Lap2']
# Image extensions to consider
image_extensions = ['.png', '.jpg', '.tif', '.jpeg', '.svs']


def allowed_file(filename):
    """
    This function checks if the format of the file is acceptable.
    :param filename: The name of the file.
    :return: True if the format is acceptable, otherwise False.
    """
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
    mean_background_val = calculate_background_mean_value(img)
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


def format_image_for_tiling(img, tile_size, overlap_size):
    mean_background_val = calculate_background_mean_value(img)
    img = img.resize(output_size(img, tile_size))
    # Adding borders with size of given overlap around the whole slide image
    img = ImageOps.expand(img, border=overlap_size, fill=tuple(mean_background_val))
    rows = int(img.height / tile_size)
    cols = int(img.width / tile_size)
    return img, rows, cols


def extract_tile(img, tile_size, overlap_size, i, j):
    return img.crop((
        i * tile_size, j * tile_size,
        i * tile_size + tile_size + 2 * overlap_size,
        j * tile_size + tile_size + 2 * overlap_size
    ))


def create_image_for_stitching(tile_size, rows, cols):
    width = tile_size * cols
    height = tile_size * rows
    return Image.new('RGB', (width, height))


def stitch_tile(img, tile, tile_size, overlap_size, i, j):
    tile = tile.resize((tile_size + 2 * overlap_size, tile_size + 2 * overlap_size))
    tile = tile.crop((overlap_size, overlap_size, overlap_size + tile_size, overlap_size + tile_size))
    img.paste(tile, (i * tile_size, j * tile_size))


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
    """
    Adjust the contrast of a background tile.
    :param img: The image Pillow object.
    :return: The adjusted image.
    """
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


def read_bioformats_image_with_reader(path, channel=0, region=(0, 0, 0, 0)):
    """
    Using this function, you can read a specific region of a large image by giving the region bounding box (XYWH format)
    and the channel number.

    :param path: The address to the file.
    :param channel: The channel number.
    :param region: The bounding box around the region of interest (XYWH format).
    :return: The specified region of interest image (numpy array).
    """
    javabridge.start_vm(class_path=bioformats.JARS)

    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    with bioformats.ImageReader(path) as reader:
        return reader.read(t=channel, XYWH=region)


def get_information(filename):
    """
    This function reads all information in the xml of the given ome image.

    :param filename: The address to the ome image.
    :return: size_x, size_y, size_z, size_c, size_t, pixel_type
    """
    javabridge.start_vm(class_path=bioformats.JARS)

    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    metadata = bioformats.get_omexml_metadata(filename)
    omexml = bioformats.OMEXML(metadata)
    size_x, size_y, size_z, size_c, size_t, pixel_type = omexml.image().Pixels.SizeX, \
                                                         omexml.image().Pixels.SizeY, \
                                                         omexml.image().Pixels.SizeZ, \
                                                         omexml.image().Pixels.SizeC, \
                                                         omexml.image().Pixels.SizeT, \
                                                         omexml.image().Pixels.PixelType
    print('SizeX:', size_x, ' SizeY:', size_y, ' SizeZ:', size_z, ' SizeC:', size_c, ' SizeT:', size_t, ' PixelType:', pixel_type)
    return size_x, size_y, size_z, size_c, size_t, pixel_type


def write_big_tiff_file(output_addr, img, tile_size):
    """
    This function write the image into a big tiff file using the tiling and compression.
    The user can specify the tile size used for saving tiled image.
    It is saving the image in pyramid format in 3 different sub-resolutions using resampling.

    :param output_addr: The address to save the image.
    :param img: The image array.
    :param tile_size: The tile size used for writing the tiled image.
    :return:
    """
    with tf.TiffWriter(output_addr, bigtiff=True) as tif:
        options = dict(tile=(tile_size, tile_size), compression='deflate')
        tif.write(img, subifds=3, **options)
        # save pyramid levels to the two subifds
        # in production use resampling to generate sub-resolutions
        tif.write(img[::2, ::2], subfiletype=1, **options)
        tif.write(img[::4, ::4], subfiletype=1, **options)
        tif.write(img[::8, ::8], subfiletype=1, **options)

    # bioformats.write_image(output_addr, img, bioformats.PT_UINT8, c=0, z=0, t=0, size_c=1, size_z=1, size_t=1, channel_names=None)


def write_ome_tiff_file(img, output_file, SizeT=1, SizeZ=1, SizeC=1, SizeX=2048, SizeY=2048, channel_names=None, Series = 0, scalex = 0.10833, scaley = 0.10833, scalez = 0.3, pixeltype ='uint8', dimorder ='TZCYX'):
    """
    This function writes an ome tiff image along with the corresponding xml file.

    :param img: The image array.
    :param output_file: The address for saving the output image.
    :param SizeT: Size t
    :param SizeZ: Size z
    :param SizeC: Size c
    :param SizeX: Size x
    :param SizeY: Size y
    :param channel_names: The name of the channels to be saved.
    :param Series: The series number.
    :param scalex: Physical Size x
    :param scaley: Physical Size y
    :param scalez: Physical Size z
    :param pixeltype: The pixeltype.
    :param dimorder: The dimension order to save the image (default: TZCYX).
    :return:
    """

    if channel_names is None:
        channel_names = ['C1']

    def writeplanes(pixel, SizeT=1, SizeZ=1, SizeC=1, order='TZCYX', verbose=False):
        if order == 'TZCYX':
            p.DimensionOrder = ome.DO_XYCZT
            counter = 0
            for t in range(SizeT):
                for z in range(SizeZ):
                    for c in range(SizeC):

                        if verbose:
                            print('Write PlaneTable: ', t, z, c),
                            sys.stdout.flush()

                        pixel.Plane(counter).TheT = t
                        pixel.Plane(counter).TheZ = z
                        pixel.Plane(counter).TheC = c
                        counter = counter + 1

        return pixel



    # Getting metadata info
    omexml = ome.OMEXML()
    omexml.image(Series).Name = output_file
    p = omexml.image(Series).Pixels
    p.SizeX, p.SizeY, p.SizeC, p.SizeT, p.SizeZ = SizeX, SizeY, SizeC, SizeT, SizeZ
    p.PhysicalSizeX, p.PhysicalSizeY, p.PhysicalSizeZ = np.float(scalex), np.float(scaley), np.float(scalez)
    p.PixelType = pixeltype
    p.channel_count = SizeC

    for i in range(len(channel_names)):
        p.Channel(i).set_Name(channel_names[i])
        p.Channel(i).set_ID(channel_names[i])
    p.plane_count = SizeZ * SizeT * SizeC
    p = writeplanes(p, SizeT=SizeT, SizeZ=SizeZ, SizeC=SizeC, order=dimorder)

    for c in range(SizeC):
        if pixeltype == 'uint8':
            p.Channel(c).SamplesPerPixel = 1
        if pixeltype == 'uint16':
            p.Channel(c).SamplesPerPixel = 2

    omexml.structured_annotations.add_original_metadata(
        ome.OM_SAMPLES_PER_PIXEL, str(SizeC))

    # Converting to omexml
    xml = omexml.to_xml()

    with tf.TiffWriter(output_file, bigtiff=True) as tif:
        tif.write(img
                 , tile=1000
                 , description=xml
                 , photometric='minisblack'
                 , metadata={'axes': 'TZCYX'
                 , 'DimensionOrder': 'TZCYX'
                 , 'Resolution': 0.10833
                 , 'Channels': channel_names}
                 )


def write_ome_tiff_file_array(results_array, output_addr, size_t, size_z, size_c, size_x, size_y):
    """
    This function writes an ome tiff file where each channel is a modality.
    :param results_array: The dictionary containing all modalities
    where the key is the modality name and the value is the modality array.
    :param output_addr: The address to the ome tiff file for saving the multi-channel image.
    :param size_t: Size T
    :param size_z: Size z
    :param size_c: Number of channels
    :param size_x: Size x
    :param size_y: Size y
    :return:
    """
    all_images = []
    channel_names = []
    dapi = results_array['DAPI']
    lap2 = results_array['Lap2']
    marker = results_array['Marker']
    seg = results_array['Seg']
    all_images = np.zeros((dapi.shape[0], dapi.shape[1], 6), dtype=np.uint8)
    all_images[:, :, 0] = dapi[:, :, 0]
    all_images[:, :, 1] = lap2[:, :, 1]
    all_images[:, :, 2] = marker[:, :, 0]
    all_images[:, :, 3] = seg[:, :, 1]
    all_images[:, :, 4] = seg[:, :, 0]
    all_images[:, :, 5] = seg[:, :, 2]
    channel_names = ['DAPI', 'Lap2', 'Marker', 'Segmentation', 'Positive', 'Negative']
    all_images = np.array(all_images)
    all_images = all_images.reshape((1, 1, all_images.shape[0], all_images.shape[1], all_images.shape[2]))

    write_ome_tiff_file(all_images,
                        output_addr,
                        SizeT=size_t, SizeZ=size_z, SizeC=len(channel_names), SizeX=size_x, SizeY=size_y,
                        channel_names=channel_names)


def write_results_to_pickle_file(output_addr, results):
    """
    This function writes data into the pickle file.
    :param output_addr: The address of the pickle file to write data into.
    :param results: The data to be written into the pickle file.
    :return:
    """
    pickle_obj = open(output_addr, "wb")
    pickle.dump(results, pickle_obj)
    pickle_obj.close()


def read_results_from_pickle_file(input_addr):
    """
    This function reads data from a pickle file and returns it.
    :param input_addr: The address to the pickle file.
    :return: The data inside pickle file.
    """
    pickle_obj = open(input_addr, "rb")
    results = pickle.load(pickle_obj)
    pickle_obj.close()
    return results

def test_diff_original_serialized(model_original,model_serialized,example,verbose=0):
    threshold = 10

    orig_res = model_original(example)
    if verbose > 0:
        print('Original:')
        print(orig_res.shape)
        print(orig_res[0, 0:10])
        print('min abs value:{}'.format(torch.min(torch.abs(orig_res))))

    ts_res = model_serialized(example)
    if verbose > 0:
        print('Torchscript:')
        print(ts_res.shape)
        print(ts_res[0, 0:10])
        print('min abs value:{}'.format(torch.min(torch.abs(ts_res))))

    abs_diff = torch.abs(orig_res-ts_res)
    if verbose > 0:
        print('Dif sum:')
        print(torch.sum(abs_diff))
        print('max dif:{}'.format(torch.max(abs_diff)))

    assert torch.sum(abs_diff) <= threshold, f"Sum of difference in predicted values {torch.sum(abs_diff)} is larger than threshold {threshold}"

def disable_batchnorm_tracking_stats(model):
    # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/16
    # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/67
    # https://github.com/pytorch/pytorch/blob/ca39c5b04e30a67512589cafbd9d063cc17168a5/torch/nn/modules/batchnorm.py#L158
    for m in model.modules():
        for child in m.children():
            if type(child) == torch.nn.BatchNorm2d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None
    return model
