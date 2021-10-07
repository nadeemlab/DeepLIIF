import cv2
from PIL import Image
import skimage.measure
from skimage.morphology import remove_small_objects
import numpy as np
import scipy.ndimage as ndi


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


def remove_small_objects_from_image(img, min_size=100):
    image_copy = img.copy()
    image_copy[img > 0] = 1
    image_copy = image_copy.astype(bool)
    removed_red_channel = remove_small_objects(image_copy, min_size=min_size).astype(np.uint8)
    img[removed_red_channel == 0] = 0

    return img


def remove_background_noise(mask, mask_boundary):
    labeled = skimage.measure.label(mask, background=0)
    padding = 5
    for i in range(1, len(np.unique(labeled))):
        component = np.zeros_like(mask)
        component[labeled == i] = mask[labeled == i]
        component_bound = np.zeros_like(mask_boundary)
        component_bound[max(0, min(np.nonzero(component)[0]) - padding): min(mask_boundary.shape[1],
                                                                             max(np.nonzero(component)[0]) + padding),
        max(0, min(np.nonzero(component)[1]) - padding): min(mask_boundary.shape[1],
                                                             max(np.nonzero(component)[1]) + padding)] \
            = mask_boundary[max(0, min(np.nonzero(component)[0]) - padding): min(mask_boundary.shape[1], max(
            np.nonzero(component)[0]) + padding),
              max(0, min(np.nonzero(component)[1]) - padding): min(mask_boundary.shape[1],
                                                                   max(np.nonzero(component)[1]) + padding)]
        if len(np.nonzero(component_bound)[0]) < len(np.nonzero(component)[0]) / 3:
            mask[labeled == i] = 0
    return mask


def remove_cell_noise(mask1, mask2):
    labeled = skimage.measure.label(mask1, background=0)
    padding = 2
    for i in range(1, len(np.unique(labeled))):
        component = np.zeros_like(mask1)
        component[labeled == i] = mask1[labeled == i]
        component_bound = np.zeros_like(mask2)
        component_bound[
        max(0, min(np.nonzero(component)[0]) - padding): min(mask2.shape[1], max(np.nonzero(component)[0]) + padding),
        max(0, min(np.nonzero(component)[1]) - padding): min(mask2.shape[1], max(np.nonzero(component)[1]) + padding)] \
            = mask2[max(0, min(np.nonzero(component)[0]) - padding): min(mask2.shape[1],
                                                                         max(np.nonzero(component)[0]) + padding),
              max(0, min(np.nonzero(component)[1]) - padding): min(mask2.shape[1],
                                                                   max(np.nonzero(component)[1]) + padding)]
        if len(np.nonzero(component_bound)[0]) > len(np.nonzero(component)[0]) / 3:
            mask1[labeled == i] = 0
            mask2[labeled == i] = 255
    return mask1, mask2


def positive_negative_masks(mask, thresh=100, noise_objects_size=20):
    positive_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    negative_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    red = mask[:, :, 0]
    blue = mask[:, :, 2]
    boundary = mask[:, :, 1]

    positive_mask[red > thresh] = 255
    positive_mask[boundary > thresh] = 0
    positive_mask[blue > red] = 0

    negative_mask[blue > thresh] = 255
    negative_mask[boundary > thresh] = 0
    negative_mask[red >= blue] = 0

    def inner(img):
        img = remove_small_objects_from_image(img, noise_objects_size)
        img = ndi.binary_fill_holes(img).astype(np.uint8)

        return cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel=np.ones((2, 2)))

    return inner(positive_mask), inner(negative_mask)


def refine(img, seg_img, thresh=100, noise_objects_size=20):
    positive_mask, negative_mask = positive_negative_masks(seg_img, thresh, noise_objects_size)

    refined_mask = np.zeros_like(img)

    refined_mask[positive_mask > 0] = (0, 0, 255)
    refined_mask[negative_mask > 0] = (255, 0, 0)

    contours, _ = cv2.findContours(positive_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(refined_mask, contours, -1, (0, 255, 0), 2)

    contours, _ = cv2.findContours(negative_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(refined_mask, contours, -1, (0, 255, 0), 2)

    return refined_mask


def overlay(img, seg_img, thresh=100, noise_objects_size=20):
    positive_mask, negative_mask = positive_negative_masks(seg_img, thresh, noise_objects_size)

    overlaid_mask = img.copy()

    contours, _ = cv2.findContours(positive_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlaid_mask, contours, -1, (0, 0, 255), 2)

    contours, _ = cv2.findContours(negative_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlaid_mask, contours, -1, (255, 0, 0), 2)

    return overlaid_mask
