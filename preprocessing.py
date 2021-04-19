import os
import shutil
import cv2
import numpy as np

from options.processing_options import ProcessingOptions

# Postfixes not to consider for segmentation
excluding_names = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg', 'Marked']
# Image extensions to consider
image_extensions = ['png', 'jpg', 'tif']


def prepare_images(input_dir, output_dir, resize_size=None, tile_size=256, overlap_size=6, resize_self=False):
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

    if resize_size is None:   # Setting default resize_size
        resize_size = [0, 0]

    # Creating an empty directory for output images
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)


    images = os.listdir(input_dir)
    print(images)
    for img_name in images:
        image_extension = img_name.split('.')[-1]  # Read image extension
        image_type = img_name.split('.')[0].split('_')[-1]  # Read image type
        if image_extension in image_extensions and image_type not in excluding_names:
            print(img_name)
            img = cv2.imread(os.path.join(input_dir, img_name))

            if resize_self:  # If the resize_size not given by the user
                if abs(img.shape[1] - img.shape[0]) < 200:  # Calculating the closest square (dividable by tile_size) for resizing the whole slide image
                    tile_no = 1 * max(int(round(img.shape[0] / tile_size)), int(round(img.shape[1] / tile_size)))
                    resize_size = [tile_no * tile_size, tile_no * tile_size]
                else:  # Calculating the closest rectangle (dividable by tile_size) for resizing the whole slide image
                    resize_size = [1*int(round(img.shape[1] / tile_size) * tile_size), 1*int(round(img.shape[0] / tile_size) * tile_size)]

            iter_val_x = int(resize_size[1] / tile_size)  # Number of tiles in the row
            iter_val_y = int(resize_size[0] / tile_size)  # Number of tiles in the column
            img = cv2.resize(img, (resize_size[0], resize_size[1]), interpolation=cv2.INTER_CUBIC)

            # Adding borders with size of given overlap around the whole slide image
            new_img = np.ones((img.shape[0] + 2 * overlap_size, img.shape[1] + 2 * overlap_size, 3), dtype=np.uint8) * 255
            new_img[overlap_size:img.shape[0]+overlap_size, overlap_size:img.shape[1]+overlap_size] = img.copy()

            # Generating the tiles and saving them in output directory
            for i in range(iter_val_x):
                for j in range(iter_val_y):
                    crop = new_img[i * (tile_size): i * (tile_size) + tile_size + 2 * overlap_size, j * (tile_size): j * (tile_size) + tile_size + 2 * overlap_size]
                    crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
                    if np.mean(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)) < 240:
                        cv2.imwrite(os.path.join(output_dir,
                                        img_name.replace('.' + image_extension,
                                                         '_' + str(i) + '_' + str(j) + '.png')),
                           np.concatenate([crop, crop, crop, crop, crop, crop], 1))


if __name__ == '__main__':
    opt = ProcessingOptions().parse()   # get training options
    prepare_images(input_dir=opt.input_dir, output_dir=opt.output_dir, resize_size=opt.resize_size, tile_size=opt.tile_size, overlap_size=opt.overlap_size, resize_self=opt.resize_self)   # Preparing images for training and testing

