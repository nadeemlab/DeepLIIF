import cv2
import os
import numpy as np

from PostProcessSegmentationMask import post_process_segmentation_mask
from deepliif.options.processing_options import ProcessingOptions


# Image extensions to consider
image_extensions = ['png', 'jpg', 'tif']

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


def post_process(input_dir, output_dir, resize_size=None, image_size=None, tile_size=256, overlap_size=6, input_orig_dir='',
                 save_types=None, post_process_seg_mask=False):

    """
    Post processing the results of the model:
    1. Stitching the tiles with overlaps and creating the final whole slide image.
    2. Saving the final images with proper postfix in the output_directory.

    :param input_dir: Path to the input images.
    :param output_dir: Path to the output images.
    :param resize_size: Resizing size of the images.
    :param image_size: Size of the original whole slide image.
    :param tile_size: Size of the tiles.
    :param overlap_size: Size of the overlap between tiles.
    :param input_orig_dir: Path to the original whole slide images (not preprocessed).
    :param save_types: Type of the images to be saved.
    :return:
    """
    if image_size is None:
        image_size = [1000, 1000]
    if resize_size is None:
        resize_size = [2000, 2000]
    if save_types is None:
        if post_process_seg_mask:
            save_types = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg', 'SegOverlaid', 'SegRefined']
        else:
            save_types = ['Hema', 'DAPI', 'DAPILap2', 'Ki67', 'Seg']


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if post_process_seg_mask:
        print('Post Processing Segmentation Mask Started!')
        post_process_segmentation_mask(input_dir)
        img_types = {'fake_B_1': 'Hema', 'fake_B_2': 'DAPI', 'fake_B_3': 'DAPILap2', 'fake_B_4': 'Ki67',
                     'fake_B_5': 'Seg', 'Seg_Overlaid_': 'SegOverlaid', 'Seg_Refined_': 'SegRefined'}
        print('Post Processing Segmentation Mask Finished!')
    else:
        img_types = {'fake_B_1': 'Hema', 'fake_B_2': 'DAPI', 'fake_B_3': 'DAPILap2', 'fake_B_4': 'Ki67',
                     'fake_B_5': 'Seg'}

    images = os.listdir(input_dir)
    images.sort()
    images_dict = {}
    images_size = {}
    original_image_sizes = {}
    print('Creating Whole Slide Image Started!')
    for img_name in images:
        for img_type in img_types.keys():
            if img_type in img_name:
                orig_name_splits = img_name.split('_')
                orig_name = ''
                for k in range(len(orig_name_splits) - 6):
                    orig_name += orig_name_splits[k] + '_'
                orig_name += orig_name_splits[len(orig_name_splits) - 6]
                img_key = orig_name + '_' + img_types[img_type] + '.png'

                if img_key not in images_dict:
                    if input_orig_dir != '':
                        image_extension = 'png'
                        for img_ext in image_extensions:
                            if os.path.exists(os.path.join(input_orig_dir, orig_name + '.' + img_ext)):
                                image_extension = img_ext
                        img = cv2.imread(os.path.join(input_orig_dir, orig_name + '.' + image_extension))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Calculating the resize_size
                        if abs(img.shape[1] - img.shape[0]) < 200: # if the image dimensions are close, resize it to form a square
                            tile_no = 1 * max(int(round(img.shape[0] / tile_size)),
                                              int(round(img.shape[1] / tile_size)))
                            resize_size = [tile_no * tile_size, tile_no * tile_size]
                        else: # if the image dimensions are not close, resize it to the closest rectangle
                            resize_size = [1 * int(round(img.shape[1] / tile_size) * tile_size),
                                           1 * int(round(img.shape[0] / tile_size) * tile_size)]

                        images_size[img_key] = img.shape

                    # Creating the image with the resize size with borders equal to overlap size
                    images_dict[img_key] = np.zeros((resize_size[1] + overlap_size * 2, resize_size[0] + overlap_size * 2, 3),dtype=np.uint8)

                    # Create an image with default values for Hematoxylin channel
                    # in case there is no corresponding image in the dataset
                    if img_types[img_type] == 'Hema':
                        images_dict[img_key][:,:,0] = 210
                        images_dict[img_key][:,:,1] = 210
                        images_dict[img_key][:,:,2] = 200

                # Index of the cropped image in the final whole slide image
                index_x = int(img_name.split('_')[-5])
                index_y = int(img_name.split('_')[-4])

                # Reading the image and adjusting if needed
                image = cv2.imread(os.path.join(input_dir, img_name))
                if img_types[img_type] == 'DAPI' or img_types[img_type] == 'Ki67':
                    image = imadjust(image, gamma=1, c=0, d=255)

                # Resizing the image to the tile_size with overlap_size
                image = cv2.resize(image, (tile_size + 2 * overlap_size, tile_size + 2 * overlap_size), interpolation=cv2.INTER_CUBIC)

                # Insert the image in the proper location in the final image.
                images_dict[img_key][index_x * (tile_size) + overlap_size:(index_x + 1) * (tile_size) + overlap_size,
                index_y * (tile_size) + overlap_size:(index_y + 1) * (tile_size) + overlap_size] = image[
                                                                                                   overlap_size:overlap_size + tile_size,
                                                                                                   overlap_size:overlap_size + tile_size]

    print('Saving Inferred Modalities and Segmentation Mask of Whole Slide Image!')
    # Saving the whole slide images
    for img_key in images_dict.keys():
        if img_key in images_size.keys():
            image_size = images_size[img_key]
        if img_key.split('_')[-1].split('.')[0] in save_types:
            curr_img = images_dict[img_key]
            new_image = curr_img[overlap_size:curr_img.shape[0]-overlap_size, overlap_size:curr_img.shape[1]-overlap_size]
            img = cv2.resize(new_image, (image_size[1],image_size[0]), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_dir, img_key), img)


if __name__ == '__main__':
    opt = ProcessingOptions().parse()   # get training options
    post_process(input_dir=opt.input_dir, output_dir=opt.output_dir, resize_size=opt.resize_size, tile_size=opt.tile_size, overlap_size=opt.overlap_size, input_orig_dir=opt.input_orig_dir, image_size=opt.image_size, post_process_seg_mask=opt.post_process_seg_mask)

