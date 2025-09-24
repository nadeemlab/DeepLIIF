"""This package includes a miscellaneous collection of useful helper functions."""
import os
import collections

import atexit
import functools
import threading

import torch
import numpy as np
from PIL import Image, ImageOps

from skimage.filters import threshold_multiotsu

from .visualizer import Visualizer

# Postfixes not to consider for segmentation
from ..postprocessing import imadjust
import cv2

import pickle
import sys

import bioformats
import javabridge
import bioformats.omexml as ome
import tifffile as tf

from tifffile import TiffFile
import zarr


excluding_names = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg', 'Marked', 'SegRefined', 'SegOverlaid', 'Marker', 'Lap2']
# Image extensions to consider
image_extensions = ['.png', '.jpg', '.tif', '.jpeg']


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


class InferenceTiler:
    """
    Iterable class to tile image(s) and stitch result tiles together.

    To perform inference on a large image, that image will need to be
    tiled into smaller tiles that can be run individually and then
    stitched back together. This class wraps the functionality as an
    iterable object that can accept a single image or list of images
    if multiple images are taken as input for inference.

    An overlap size can be specified so that neighboring tiles will
    overlap at the edges, helping to reduce seams or other artifacts
    near the edge of a tile. Padding of a solid color around the
    perimeter of the tile is also possible, if needed. The specified
    tile size includes this overlap and pad sizes, so a tile size of
    512 with an overlap size of 32 and pad size of 16 would have a
    central area of 416 pixels that are stitched into the result image.

    Example Usage
    -------------
    tiler = InferenceTiler(img, 512, 32)
    for tile in tiler:
        result_tiles = infer(tile)
        tiler.stitch(result_tiles)
    images = tiler.results()
    """

    def __init__(self, orig, tile_size, overlap_size=0, pad_size=0, pad_color=(255, 255, 255)):
        """
        Initialize for tiling an image or list of images.

        Parameters
        ----------
        orig : Image | list(Image)
            Original image or list of images to be tiled.
        tile_size: int
            Size (width and height) of the tiles to be generated.
        overlap_size: int [default: 0]
            Amount of overlap on each side of the tile.
        pad_size: int [default: 0]
            Amount of solid color padding around perimeter of tile.
        pad_color: tuple(int, int, int) [default: (255,255,255)]
            RGB color to use for padding.
        """

        if tile_size <= 0:
            raise ValueError('InfereneTiler input tile_size must be positive and non-zero')
        if overlap_size < 0:
            raise ValueError('InfereneTiler input overlap_size must be positive or zero')
        if pad_size < 0:
            raise ValueError('InfereneTiler input pad_size must be positive or zero')

        self.single_orig = not type(orig) is list
        if self.single_orig:
            orig = [orig]

        for i in range(1, len(orig)):
            if orig[i].size != orig[0].size:
                raise ValueError('InferenceTiler input images do not have the same size.')
        self.orig_width = orig[0].width
        self.orig_height = orig[0].height

        # patch size to extract from input image, which is then padded to tile size
        patch_size = tile_size - (2 * pad_size)

        # make sure width and height are both at least patch_size
        if orig[0].width < patch_size:
            for i in range(len(orig)):
                while orig[i].width < patch_size:
                    mirrored = ImageOps.mirror(orig[i])
                    orig[i] = ImageOps.expand(orig[i], (0, 0, orig[i].width, 0))
                    orig[i].paste(mirrored, (mirrored.width, 0))
                orig[i] = orig[i].crop((0, 0, patch_size, orig[i].height))
        if orig[0].height < patch_size:
            for i in range(len(orig)):
                while orig[i].height < patch_size:
                    flipped = ImageOps.flip(orig[i])
                    orig[i] = ImageOps.expand(orig[i], (0, 0, 0, orig[i].height))
                    orig[i].paste(flipped, (0, flipped.height))
                orig[i] = orig[i].crop((0, 0, orig[i].width, patch_size))
        self.image_width = orig[0].width
        self.image_height = orig[0].height

        overlap_width = 0 if patch_size >= self.image_width else overlap_size
        overlap_height = 0 if patch_size >= self.image_height else overlap_size
        center_width = patch_size - (2 * overlap_width)
        center_height = patch_size - (2 * overlap_height)
        if center_width <= 0 or center_height <= 0:
            raise ValueError('InferenceTiler combined overlap_size and pad_size are too large')

        self.c0x = pad_size                               # crop offset for left of non-pad content in result tile
        self.c0y = pad_size                               # crop offset for top of non-pad content in result tile
        self.c1x = overlap_width + pad_size               # crop offset for left of center region in result tile
        self.c1y = overlap_height + pad_size              # crop offset for top of center region in result tile
        self.c2x = patch_size - overlap_width + pad_size  # crop offset for right of center region in result tile
        self.c2y = patch_size - overlap_height + pad_size # crop offset for bottom of center region in result tile
        self.c3x = patch_size + pad_size                  # crop offset for right of non-pad content in result tile
        self.c3y = patch_size + pad_size                  # crop offset for bottom of non-pad content in result tile
        self.p1x = overlap_width               # paste offset for left of center region w.r.t (x,y) coord
        self.p1y = overlap_height              # paste offset for top of center region w.r.t (x,y) coord
        self.p2x = patch_size - overlap_width  # paste offset for right of center region w.r.t (x,y) coord
        self.p2y = patch_size - overlap_height # paste offset for bottom of center region w.r.t (x,y) coord

        self.overlap_width = overlap_width
        self.overlap_height = overlap_height
        self.patch_size = patch_size
        self.center_width = center_width
        self.center_height = center_height

        self.orig = orig
        self.tile_size = tile_size
        self.pad_size = pad_size
        self.pad_color = pad_color
        self.res = {}

    def __iter__(self):
        """
        Generate the tiles as an iterable.

        Tiles are created and iterated over from top left to bottom
        right, going across the rows. The yielded tile(s) match the
        type of the original input when initialized (either a single
        image or a list of images in the same order as initialized).
        The (x, y) coordinate of the current tile is maintained
        internally for use in the stitch function.
        """

        for y in range(0, self.image_height, self.center_height):
            for x in range(0, self.image_width, self.center_width):
                if x + self.patch_size > self.image_width:
                    x = self.image_width - self.patch_size
                if y + self.patch_size > self.image_height:
                    y = self.image_height - self.patch_size
                self.x = x
                self.y = y
                tiles = [im.crop((x, y, x + self.patch_size, y + self.patch_size)) for im in self.orig]
                if self.pad_size != 0:
                    tiles = [ImageOps.expand(t, self.pad_size, self.pad_color) for t in tiles]
                yield tiles[0] if self.single_orig else tiles

    def stitch(self, result_tiles):
        """
        Stitch result tiles into the result images.

        The key names for the dictionary of result tiles are used to
        stitch each tile into its corresponding final image in the
        results attribute. If a result image does not exist for a
        result tile key name, then it will be created. The result tiles
        are stitched at the location from which the list iterated tile
        was extracted.

        Parameters
        ----------
        result_tiles : dict(str: Image)
            Dictionary of result tiles from the inference.
        """

        for k, tile in result_tiles.items():
            if k not in self.res:
                self.res[k] = Image.new('RGB', (self.image_width, self.image_height))
            if tile.size != (self.tile_size, self.tile_size):
                tile = tile.resize((self.tile_size, self.tile_size))
            self.res[k].paste(tile.crop((self.c1x, self.c1y, self.c2x, self.c2y)), (self.x + self.p1x, self.y + self.p1y))

            # top left corner
            if self.x == 0 and self.y == 0:
                self.res[k].paste(tile.crop((self.c0x, self.c0y, self.c1x, self.c1y)), (self.x, self.y))
            # top row
            if self.y == 0:
                self.res[k].paste(tile.crop((self.c1x, self.c0y, self.c2x, self.c1y)), (self.x + self.p1x, self.y))
            # top right corner
            if self.x == self.image_width - self.patch_size and self.y == 0:
                self.res[k].paste(tile.crop((self.c2x, self.c0y, self.c3x, self.c1y)), (self.x + self.p2x, self.y))
            # left column
            if self.x == 0:
                self.res[k].paste(tile.crop((self.c0x, self.c1y, self.c1x, self.c2y)), (self.x, self.y + self.p1y))
            # right column
            if self.x == self.image_width - self.patch_size:
                self.res[k].paste(tile.crop((self.c2x, self.c1y, self.c3x, self.c2y)), (self.x + self.p2x, self.y + self.p1y))
            # bottom left corner
            if self.x == 0 and self.y == self.image_height - self.patch_size:
                self.res[k].paste(tile.crop((self.c0x, self.c2y, self.c1x, self.c3y)), (self.x, self.y + self.p2y))
            # bottom row
            if self.y == self.image_height - self.patch_size:
                self.res[k].paste(tile.crop((self.c1x, self.c2y, self.c2x, self.c3y)), (self.x + self.p1x, self.y + self.p2y))
            # bottom right corner
            if self.x == self.image_width - self.patch_size and self.y == self.image_height - self.patch_size:
                self.res[k].paste(tile.crop((self.c2x, self.c2y, self.c3x, self.c3y)), (self.x + self.p2x, self.y + self.p2y))

    def results(self):
        """
        Return a dictionary of result images.

        The keys for the result images are the same as those used for
        the result tiles in the stitch function. This function should
        only be called once, since the stitched images will be cropped
        if the original image size was less than the patch size.
        """

        if self.orig_width != self.image_width or self.orig_height != self.image_height:
            return {k: im.crop((0, 0, self.orig_width, self.orig_height)) for k, im in self.res.items()}
        else:
            return {k: im for k, im in self.res.items()}


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


def infer_background_colors(dir_data, sample_size=5, input_no=1, modalities_no=4,
                                    seg_no=1, tile_size=32, return_list=False):
    fns = [x for x in os.listdir(dir_data) if x.endswith('.png')]
    sample_size = min(sample_size, len(fns))
    w, h, num_img = None, None, None

    background_colors = {}

    for fn in fns[:sample_size]:
        img = Image.open(f"{dir_data}/{fn}")
        
        if w is None:
            num_img = img.size[0] / img.size[1]
            num_img = int(num_img)
            w, h = img.size
        
        background_colors_img = infer_background_colors_for_img(img, input_no=input_no, modalities_no=modalities_no, seg_no=seg_no, tile_size=tile_size, w=w, h=h, num_img=num_img)
        
        for mod_id, rgb_avg in background_colors_img.items():
            try:
                background_colors[mod_id].append(rgb_avg)
            except:
                background_colors[mod_id] = [rgb_avg]
    
    background_colors = {k:np.mean(v,axis=0).astype(np.uint8) for k,v in background_colors.items()}
    
    if return_list:
        return [tuple(e) for e in background_colors.values()]
    else:
        return background_colors


def infer_background_colors_for_img(img, input_no=1, modalities_no=4, seg_no=1, tile_size=32,
                                    w=None, h=None, num_img=None):
    """
    Estimate background colors for a given RGB image.
    The empty tiles are determined by applying is_empty() function on segmentation modalities.
    If multiple segmentation modalities present, only common empty tiles are used for background
    color calculation.
    """
    from ..models import is_empty
    
    if w is None:
        num_img = img.size[0] / img.size[1]
        num_img = int(num_img)
        w, h = img.size
            
    empty_tiles = {}
    l_box = []
    background_colors = {}
    
    for i in range(num_img-seg_no, num_img):
        img_mod = img.crop((h*i,0,h*(i+1),h))
        l_box_mod = []
        for x in range(0, h, tile_size):
            for y in range(0, h, tile_size):
                box = (x, y, x+tile_size, y+tile_size)
                tile = img_mod.crop(box)
                if is_empty(tile):
                    l_box_mod.append(box)
        l_box.append(l_box_mod)

    l_box_final = set()
    if len(l_box) > 1:
        # only keep overlapped boxes
        for l in l_box:
            l_box_final = l_box_final & set(l)
        l_box_final = list(l_box_final)
    else:
        l_box_final = l_box[0]
    #print(f'{len(l_box_final)} tiles are considered empty using segmentation modalities')

    for i in range(input_no, modalities_no+input_no):
        empty_tiles[i] = []
        img_mod = img.crop((h*i,0,h*(i+1),h))
        for box in l_box_final:
            tile = img_mod.crop(box)
            empty_tiles[i].append(tile)

        img_avg = np.mean(np.stack(empty_tiles[i], axis=0), axis=0) # take an average across all empty images
        rgb_avg = np.mean(img_avg,axis=(0,1)).astype(np.uint8)
        background_colors[i] = rgb_avg

    return background_colors


def image_variance_gray(img):
    px = np.asarray(img) if img.mode == 'L' else np.asarray(img.convert('L'))
    idx = np.logical_and(px != 255, px != 0)
    val = px[idx]
    if val.shape[0] == 0:
        return 0
    var = np.var(val)
    return var


def image_variance_rgb(img):
    px = np.asarray(img) if img.mode == 'RGB' else np.asarray(img.convert('RGB'))
    nonwhite = np.any(px != [255, 255, 255], axis=-1)
    nonblack = np.any(px != [0, 0, 0], axis=-1)
    idx = np.logical_and(nonwhite, nonblack)
    val = px[idx]
    if val.shape[0] == 0:
        return [0, 0, 0]
    var = np.var(val, axis=0)
    return var



def init_javabridge_bioformats():
    """
    Initialize javabridge for use with bioformats.
    Run as daemon so no need to explicitly call kill_vm.
    This function will only run once; repeat calls do nothing.
    """

    if not hasattr(init_javabridge_bioformats, 'called'):
        # https://github.com/LeeKamentsky/python-javabridge/issues/155
        old_init = threading.Thread.__init__
        threading.Thread.__init__ = functools.partialmethod(old_init, daemon=True)
        javabridge.start_vm(class_path=bioformats.JARS)
        threading.Thread.__init__ = old_init
        atexit.register(javabridge.kill_vm)

        rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
        rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                            "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
        logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "WARN", "Lch/qos/logback/classic/Level;")
        javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

        init_javabridge_bioformats.called = True


def read_bioformats_image_with_reader(path, channel=0, region=(0, 0, 0, 0)):
    """
    Using this function, you can read a specific region of a large image by giving the region bounding box (XYWH format)
    and the channel number.

    :param path: The address to the file.
    :param channel: The channel number.
    :param region: The bounding box around the region of interest (XYWH format).
    :return: The specified region of interest image (numpy array).
    """
    init_javabridge_bioformats()
    with bioformats.ImageReader(path) as reader:
        return reader.read(t=channel, XYWH=region)


def get_information(filename):
    """
    This function reads all information in the xml of the given ome image.

    :param filename: The address to the ome image.
    :return: size_x, size_y, size_z, size_c, size_t, pixel_type
    """
    init_javabridge_bioformats()
    metadata = bioformats.get_omexml_metadata(filename)
    omexml = bioformats.OMEXML(metadata)
    size_x, size_y, size_z, size_c, size_t, pixel_type = omexml.image().Pixels.SizeX, \
                                                         omexml.image().Pixels.SizeY, \
                                                         omexml.image().Pixels.SizeZ, \
                                                         omexml.image().Pixels.SizeC, \
                                                         omexml.image().Pixels.SizeT, \
                                                         omexml.image().Pixels.PixelType
    #print('SizeX:', size_x, ' SizeY:', size_y, ' SizeZ:', size_z, ' SizeC:', size_c, ' SizeT:', size_t, ' PixelType:', pixel_type)
    return size_x, size_y, size_z, size_c, size_t, pixel_type


class WSIReader:
    """
    Assumes the file is a single image (e.g., not a stacked
    OME TIFF) and will always return uint8 pixel type data.
    """

    def __init__(self, path):
        init_javabridge_bioformats()
        metadata = bioformats.get_omexml_metadata(path)
        omexml = bioformats.OMEXML(metadata)

        self._path = path
        self._width = omexml.image().Pixels.SizeX
        self._height = omexml.image().Pixels.SizeY
        self._pixel_type = omexml.image().Pixels.PixelType

        self._tif = None
        if self._pixel_type == 'uint8':
            try:
                self._file = None
                self._file = open(path, 'rb')
                self._tif = TiffFile(self._file)
                self._zarr = zarr.open(self._tif.pages[0].aszarr(), mode='r')
            except Exception as e:
                if self._tif is not None:
                    self._tif.close()
                    self._tif = None
                if self._file is not None:
                    self._file.close()

        self._bfreader = None
        if self._tif is None:
            self._rescale = (self._pixel_type != 'uint8')
            self._bfreader = bioformats.ImageReader(path)

        if self._tif is None and self._bfreader is None:
            raise Exception('Cannot read WSI file.')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._tif is not None:
            self._tif.close()
            self._file.close()
        if self._bfreader is not None:
            self._bfreader.close()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def read(self, xywh):
        if self._tif is not None:
            x, y, w, h = xywh
            return self._zarr[y:y+h, x:x+w]

        px = self._bfreader.read(XYWH=xywh, rescale=self._rescale)
        if self._rescale:
            px = (px * 255).astype(np.uint8)
        return px


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
                child.running_mean_backup = child.running_mean
                child.running_mean = None
                child.running_var_backup = child.running_var
                child.running_var = None
    return model

def enable_batchnorm_tracking_stats(model):
    """
    This is needed during training when val set loss/metrics calculation is enabled.
    In this case, we need to switch to eval mode for inference, which triggers
    disable_batchnorm_tracking_stats(). After the evaluation, the model should be
    set back to train mode, where running stats are restored for batchnorm layers.
    """
    for m in model.modules():
        for child in m.children():
            if type(child) == torch.nn.BatchNorm2d:
                child.track_running_stats = True
                assert hasattr(child, 'running_mean_backup') and hasattr(child, 'running_var_backup'), 'enable_batchnorm_tracking_stats() is supposed to be executed after disable_batchnorm_tracking_stats() is applied'
                child.running_mean = child.running_mean_backup
                child.running_var = child.running_var_backup
    return model


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
