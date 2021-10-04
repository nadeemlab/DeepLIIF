import os
import shutil

from PIL import Image

from deepliif.options.processing_options import ProcessingOptions
from deepliif.preprocessing import allowed_file, output_size, generate_tiles

if __name__ == '__main__':
    """
    Preparing images for training/testing.
    1. Resizes the whole slide image to the given size or the closest rectangle/square.
    2. Breaks the resized whole slide image into tiles with overlaps.
    3. Saves the tiles to output directory.

    :param input_dir: path to input images
    :param output_dir: path to output images
    :param resize_size: resizing size of the whole slide image
    :param tile_size: size of the tiles
    :param overlap_size: size of the overlaps between tiles
    :param resize_self: if True, resize to the closest proper rectangle, else resize to the given resize_size
    :return:
    """
    # get training options
    opt = ProcessingOptions().parse()

    # Creating an empty directory for output images
    if os.path.exists(opt.output_dir):
        shutil.rmtree(opt.output_dir)
    os.makedirs(opt.output_dir)

    for img_name in os.listdir(opt.input_dir):
        if allowed_file(img_name):
            img = Image.open(os.path.join(opt.input_dir, img_name))

            img = img.resize(output_size(img, opt.tile_size))

            for tile in generate_tiles(img, opt.tile_size, opt.overlap_size):
                img = Image.new('RGB', (tile.img.width * 6, tile.img.height))

                for i in range(6):
                    img.paste(tile.img, (i * opt.tile_size, 0))

                img.save(os.path.join(
                    opt.output_dir,
                    img_name.replace(f".{img_name.split('.')[-1]}", f'_{tile.i}_{tile.j}.png')
                ))
