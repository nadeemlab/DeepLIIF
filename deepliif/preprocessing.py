import os
import collections

from PIL import Image, ImageOps
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms

from deepliif.data.base_dataset import __make_power_2

# Postfixes not to consider for segmentation
excluding_names = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg', 'Marked', 'SegRefined', 'SegOverlaid', 'Marker', 'Lap2']
# Image extensions to consider
image_extensions = ['.png', '.jpg', '.tif']

Tile = collections.namedtuple('Tile', 'i, j, img')


def allowed_file(filename):
    name, extension = os.path.splitext(filename)
    image_type = name.split('_')[-1]  # Read image type

    return extension in image_extensions and image_type not in excluding_names


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


def transform(img):
    return default_collate([transforms.Compose([
        transforms.Lambda(lambda i: __make_power_2(i, base=4, method=Image.BICUBIC)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(img)])
