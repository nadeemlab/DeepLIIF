import os.path
import cv2
import numpy as np
from scipy import ndimage
from numba import jit
from skimage import measure, feature


def get_average_cell_size(image):
    label_image = measure.label(image, background=0)
    labels = np.unique(label_image)
    average_cell_size = 0
    for _label in range(1, len(labels)):
        indices = np.where(label_image == _label)
        pixel_count = np.count_nonzero(image[indices])
        average_cell_size += pixel_count
    average_cell_size /= len(labels)
    return average_cell_size


@jit(nopython=True)
def get_average_cell_size_gpu(label_image, image_size, labels_no):
    average_cell_size = 0
    for _label in labels_no:
        if _label == 0:
            continue
        pixel_count = 0
        for index_x in range(image_size[0]):
            for index_y in range(image_size[1]):
                if label_image[index_x, index_y] == _label:
                    pixel_count += 1
        average_cell_size += pixel_count
    average_cell_size /= len(labels_no)
    return average_cell_size


@jit(nopython=True)
def compute_cell_mapping(new_mapping, image_size, small_object_size=20):
    marked = [[False for _ in range(image_size[1])] for _ in range(image_size[0])]
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if marked[i][j] is False and (new_mapping[i, j, 0] > 0 or new_mapping[i, j, 2] > 0):
                cluster_red_no, cluster_blue_no = 0, 0
                pixels = [(i, j)]
                cluster = [(i, j)]
                marked[i][j] = True
                while len(pixels) > 0:
                    pixel = pixels.pop()
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
                if cluster_red_no < cluster_blue_no:
                    cluster_value = (0, 0, 255)
                else:
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


def positive_negative_masks(mask, thresh=100, boundary_thresh=100, noise_objects_size=50):
    positive_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    negative_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    red = mask[:, :, 0]
    blue = mask[:, :, 2]
    boundary = mask[:, :, 1]

    # Filtering boundary pixels
    boundary[boundary < boundary_thresh] = 0

    positive_mask[red > thresh] = 255
    positive_mask[boundary > 0] = 0
    positive_mask[blue > red] = 0

    negative_mask[blue > thresh] = 255
    negative_mask[boundary > 0] = 0
    negative_mask[red >= blue] = 0

    cell_mapping = np.zeros_like(mask)
    cell_mapping[:, :, 0] = positive_mask
    cell_mapping[:, :, 2] = negative_mask

    compute_cell_mapping(cell_mapping, mask.shape, small_object_size=noise_objects_size)
    cell_mapping[cell_mapping > 0] = 255

    positive_mask = cell_mapping[:, :, 0]
    negative_mask = cell_mapping[:, :, 2]

    # return remove_noises_fill_empty_holes(positive_mask, noise_objects_size), remove_noises_fill_empty_holes(negative_mask, noise_objects_size)
    return positive_mask, negative_mask


def create_final_segmentation_mask_with_boundaries(positive_mask, negative_mask):
    refined_mask = np.zeros((positive_mask.shape[0], positive_mask.shape[1], 3), dtype=np.uint8)

    refined_mask[positive_mask > 0] = (255, 0, 0)
    refined_mask[negative_mask > 0] = (0, 0, 255)

    edges = feature.canny(refined_mask[:,:,0], sigma=3).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(refined_mask, contours, -1, (0, 255, 0), 2)

    edges = feature.canny(refined_mask[:,:,2], sigma=3).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(refined_mask, contours, -1, (0, 255, 0), 2)

    return refined_mask


def count_number_of_cells(input_dir):
    images = os.listdir(input_dir)
    total_red = 0
    total_blue = 0
    for img in images:
        image = cv2.cvtColor(cv2.imread(os.path.join(input_dir, img)), cv2.COLOR_BGR2RGB)
        image = image[:,5*512:]
        red = image[:,:,0]
        blue = image[:,:,2]
        labeled_red, nr_objects_red = ndimage.label(red > 0)
        labeled_blue, nr_objects_blue = ndimage.label(blue > 0)
        total_red += nr_objects_red
        total_blue += nr_objects_blue
    return total_red, total_blue
