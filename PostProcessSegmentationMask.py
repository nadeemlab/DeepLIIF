import os.path
import sys

import cv2
import numpy as np
import scipy.ndimage as ndi

from deepliif.postprocessing import overlay, refine, remove_cell_noise, remove_background_noise, \
    remove_small_objects_from_image


def align_seg_on_image(input_image, input_mask, output_image, thresh=100, noise_objects_size=100):
    seg_image = cv2.cvtColor(cv2.imread(input_mask), cv2.COLOR_BGR2RGB)
    orig_image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB)

    final_mask = orig_image.copy()
    processed_mask = np.zeros_like(orig_image)

    red = seg_image[:, :, 0]
    blue = seg_image[:, :, 2]
    boundary = seg_image[:, :, 1]

    boundary[boundary < thresh] = 0

    positive_cells = np.zeros((seg_image.shape[0], seg_image.shape[1]), dtype=np.uint8)
    negative_cells = np.zeros((seg_image.shape[0], seg_image.shape[1]), dtype=np.uint8)

    positive_cells[red > thresh] = 255
    positive_cells[boundary > thresh] = 0
    negative_cells[blue > thresh] = 255
    negative_cells[boundary > thresh] = 0

    negative_cells[red >= blue] = 0
    positive_cells[blue > red] = 0

    positive_cells = cv2.morphologyEx(positive_cells, cv2.MORPH_DILATE, kernel=np.ones((2, 2)))
    negative_cells = cv2.morphologyEx(negative_cells, cv2.MORPH_DILATE, kernel=np.ones((2, 2)))

    negative_cells = remove_background_noise(negative_cells, boundary)
    positive_cells = remove_background_noise(positive_cells, boundary)

    negative_cells, positive_cells = remove_cell_noise(negative_cells, positive_cells)
    positive_cells, negative_cells = remove_cell_noise(positive_cells, negative_cells)

    negative_cells = remove_small_objects_from_image(negative_cells, noise_objects_size)
    negative_cells = ndi.binary_fill_holes(negative_cells).astype(np.uint8)

    positive_cells = remove_small_objects_from_image(positive_cells, noise_objects_size)
    positive_cells = ndi.binary_fill_holes(positive_cells).astype(np.uint8)

    contours, hierarchy = cv2.findContours(positive_cells,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(final_mask, contours, -1, (255, 0, 0), 2)

    contours, hierarchy = cv2.findContours(negative_cells,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(final_mask, contours, -1, (0, 0, 255), 2)

    processed_mask[positive_cells > 0] = (0, 0, 255)
    processed_mask[negative_cells > 0] = (255, 0, 0)

    contours, hierarchy = cv2.findContours(positive_cells,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(processed_mask, contours, -1, (0, 255, 0), 2)

    contours, hierarchy = cv2.findContours(negative_cells,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(processed_mask, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(output_image, cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB))
    cv2.imwrite(output_image.replace('Overlaid', 'Refined'), processed_mask)


def align_seg_on_image2(input_image, input_mask, output_image, thresh=100, noise_objects_size=20):
    seg_image = cv2.cvtColor(cv2.imread(input_mask), cv2.COLOR_BGR2RGB)
    orig_image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB)

    overlaid_mask = overlay(orig_image, seg_image, thresh, noise_objects_size)
    cv2.imwrite(output_image, cv2.cvtColor(overlaid_mask, cv2.COLOR_BGR2RGB))

    refined_mask = refine(orig_image, seg_image, thresh, noise_objects_size)
    cv2.imwrite(output_image.replace('Overlaid', 'Refined'), refined_mask)


def post_process_segmentation_mask(input_dir, seg_thresh=100, noise_object_size=100):
    images = os.listdir(input_dir)
    for img in images:
        if '_fake_B_5.png' in img:
            align_seg_on_image2(os.path.join(input_dir, img.replace('_fake_B_5', '_real_A')),
                                os.path.join(input_dir, img),
                                os.path.join(input_dir, img.replace('_fake_B_5', '_Seg_Overlaid_')),
                                thresh=seg_thresh, noise_objects_size=noise_object_size)
        elif '_Seg.png' in img:
            align_seg_on_image2(os.path.join(input_dir, img.replace('_Seg', '')),
                                os.path.join(input_dir, img),
                                os.path.join(input_dir, img.replace('_Seg', '_SegOverlaid')),
                                thresh=seg_thresh, noise_objects_size=noise_object_size)


if __name__ == '__main__':
    base_dir = sys.argv[1]
    segmentation_thresh = 100
    noise_obj_size = 20
    if len(sys.argv) > 2:
        segmentation_thresh = int(sys.argv[2])
    if len(sys.argv) > 3:
        noise_obj_size = int(sys.argv[3])

    post_process_segmentation_mask(base_dir, segmentation_thresh, noise_obj_size)
