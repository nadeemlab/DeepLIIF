import cv2
from PIL import Image
import skimage.measure
from skimage import feature
from skimage.morphology import remove_small_objects
import numpy as np
import scipy.ndimage as ndi
from numba import jit
import math


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


@jit(nopython=True)
def compute_cell_mapping(new_mapping, image_size, small_object_size=20):
    marked = [[False for _ in range(image_size[1])] for _ in range(image_size[0])]
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if marked[i][j] is False and (new_mapping[i, j, 0] > 0 or new_mapping[i, j, 2] > 0):
                cluster_red_prob, cluster_blue_prob, cluster_red_no, cluster_blue_no = 0, 0, 0, 0
                pixels = [(i, j)]
                cluster = [(i, j)]
                marked[i][j] = True
                while len(pixels) > 0:
                    pixel = pixels.pop()
                    cluster_red_prob += new_mapping[pixel[0], pixel[1], 0]
                    cluster_blue_prob += new_mapping[pixel[0], pixel[1], 2]
                    if new_mapping[pixel[0], pixel[1], 0] > 0:
                        cluster_red_no += 1
                    if new_mapping[pixel[0], pixel[1], 2] > 0:
                        cluster_blue_no += 1
                    for neigh_i in range(-1, 2):
                        for neigh_j in range(-1, 2):
                            neigh_pixel = (pixel[0] + neigh_i, pixel[1] + neigh_j)
                            if 0 <= neigh_pixel[0] < image_size[0] and 0 <= neigh_pixel[1] < image_size[1] and marked[neigh_pixel[0]][neigh_pixel[1]] is False and (new_mapping[neigh_pixel[0], neigh_pixel[1], 0] > 0 or new_mapping[neigh_pixel[0], neigh_pixel[1], 2] > 0):
                                cluster.append(neigh_pixel)
                                pixels.append(neigh_pixel)
                                marked[neigh_pixel[0]][neigh_pixel[1]] = True
                cluster_value = None
                if cluster_red_prob < cluster_blue_prob or cluster_red_no * 2 < cluster_blue_no:
                    cluster_value = (0, 0, 255)
                elif cluster_blue_prob < cluster_red_prob or cluster_blue_no * 2 < cluster_red_no:
                    cluster_value = (255, 0, 0)
                if len(cluster) < small_object_size:
                    cluster_value = (0, 0, 0)
                if cluster_value is not None:
                    for node in cluster:
                        new_mapping[node[0], node[1]] = cluster_value
    return new_mapping


@jit(nopython=True)
def remove_noises(channel, image_size, small_object_size=20):
    marked = [[False for _ in range(image_size[1])] for _ in range(image_size[0])]
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if marked[i][j] is False and channel[i, j] > 0:
                pixels = [(i, j)]
                cluster = [(i, j)]
                marked[i][j] = True
                while len(pixels) > 0:
                    pixel = pixels.pop()
                    for neigh_i in range(-1, 2):
                        for neigh_j in range(-1, 2):
                            neigh_pixel = (pixel[0] + neigh_i, pixel[1] + neigh_j)
                            if 0 <= neigh_pixel[0] < image_size[0] and 0 <= neigh_pixel[1] < image_size[1] and marked[neigh_pixel[0]][neigh_pixel[1]] is False and channel[neigh_pixel[0], neigh_pixel[1]] > 0:
                                cluster.append(neigh_pixel)
                                pixels.append(neigh_pixel)
                                marked[neigh_pixel[0]][neigh_pixel[1]] = True

                cluster_value = None
                if len(cluster) < small_object_size:
                    cluster_value = 0
                if cluster_value is not None:
                    for node in cluster:
                        channel[node[0], node[1]] = cluster_value
    return channel


def remove_noises_fill_empty_holes(label_img, size=200):
    inverse_img = 255 - label_img
    inverse_img_removed = remove_noises(inverse_img, inverse_img.shape, small_object_size=size)
    label_img[inverse_img_removed == 0] = 255
    return label_img


def positive_negative_masks(mask, thresh=100, noise_objects_size=20):
    positive_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    negative_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    red = mask[:, :, 0]
    blue = mask[:, :, 2]
    boundary = mask[:, :, 1]

    boundary[boundary < 80] = 0

    positive_mask[red > thresh] = 255
    positive_mask[boundary > thresh] = 0
    positive_mask[blue > red] = 0

    negative_mask[blue > thresh] = 255
    negative_mask[boundary > 0] = 0
    negative_mask[red >= blue] = 0

    cell_mapping = np.zeros_like(mask)
    cell_mapping[:, :, 0] = positive_mask
    cell_mapping[:, :, 2] = negative_mask

    compute_cell_mapping(cell_mapping, mask.shape, small_object_size=50)
    cell_mapping[cell_mapping > 0] = 255

    positive_mask = cell_mapping[:, :, 0]
    negative_mask = cell_mapping[:, :, 2]

    def inner(img):
        img = remove_small_objects_from_image(img, noise_objects_size)
        img = ndi.binary_fill_holes(img).astype(np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel=np.ones((2, 2)))

    # return inner(positive_mask), inner(negative_mask)
    return remove_noises_fill_empty_holes(positive_mask, noise_objects_size), remove_noises_fill_empty_holes(negative_mask, noise_objects_size)


def refine(img, seg_img, thresh=100, noise_objects_size=20):
    positive_mask, negative_mask = positive_negative_masks(seg_img, thresh, noise_objects_size)

    refined_mask = np.zeros_like(img)

    refined_mask[positive_mask > 0] = (0, 0, 255)
    refined_mask[negative_mask > 0] = (255, 0, 0)

    edges = feature.canny(positive_mask, sigma=3).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(refined_mask, contours, -1, (0, 255, 0), 2)

    edges = feature.canny(negative_mask, sigma=3).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(refined_mask, contours, -1, (0, 255, 0), 2)

    return refined_mask


def overlay(img, seg_img, thresh=100, noise_objects_size=20):
    positive_mask, negative_mask = positive_negative_masks(seg_img, thresh, noise_objects_size)

    overlaid_mask = img.copy()

    edges = feature.canny(positive_mask, sigma=3).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlaid_mask, contours, -1, (0, 0, 255), 2)

    edges = feature.canny(negative_mask, sigma=3).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlaid_mask, contours, -1, (255, 0, 0), 2)

    return overlaid_mask


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


def adjust_dapi(inferred_tile, orig_tile):
    """Adjusts the intensity of mpIF DAPI image

    Parameters:
        inferred_tile (Image) -- inferred tile image
        orig_tile (Image) -- original tile image

    """
    inferred_tile_array = np.array(inferred_tile)
    orig_tile_array = np.array(orig_tile)

    multiplier = 8 / math.log(np.max(orig_tile_array))

    if np.mean(orig_tile_array) < 200:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier * math.log(np.mean(inferred_tile_array)) / math.log(np.mean(orig_tile_array)),
                                  c=5, d=255).astype(np.uint8)

    else:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier,
                                  c=5, d=255).astype(np.uint8)
    return Image.fromarray(processed_tile)


def adjust_marker(inferred_tile, orig_tile):
    """Adjusts the intensity of mpIF marker image

    Parameters:
        inferred_tile (Image) -- inferred tile image
        orig_tile (Image) -- original tile image

    """
    inferred_tile_array = np.array(inferred_tile)
    orig_tile_array = np.array(orig_tile)

    multiplier = 10 / math.log(np.max(orig_tile_array))

    if np.mean(orig_tile_array) < 200:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier * math.log(np.std(inferred_tile_array)) / math.log(
                                      np.std(orig_tile_array)),
                                  c=5, d=255).astype(np.uint8)

    else:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier,
                                  c=5, d=255).astype(np.uint8)
    return Image.fromarray(processed_tile)

