import math
import cv2
from PIL import Image
import skimage.measure
from skimage import feature
from skimage.morphology import remove_small_objects
import numpy as np
import scipy.ndimage as ndi
from numba import jit


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


def create_basic_segmentation_mask(img, seg_img, thresh=80, noise_objects_size=20, small_object_size=50):
    positive_mask, negative_mask = positive_negative_masks_basic(img, seg_img, thresh, noise_objects_size, small_object_size)

    mask = np.zeros_like(img)

    mask[positive_mask > 0] = (255, 0, 0)
    mask[negative_mask > 0] = (0, 0, 255)

    return mask


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

    Returns:
        processed_tile (Image) -- the adjusted mpIF DAPI image
    """
    inferred_tile_array = np.array(inferred_tile)
    orig_tile_array = np.array(orig_tile)

    multiplier = 8 / math.log(np.max(orig_tile_array))

    if np.mean(orig_tile_array) < 200:
        processed_tile = imadjust(inferred_tile_array,
                                  gamma=multiplier * math.log(np.mean(inferred_tile_array)) / math.log(
                                      np.mean(orig_tile_array)),
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

    Returns:
        processed_tile (Image) -- the adjusted marker image
    """
    inferred_tile_array = np.array(inferred_tile)
    orig_tile_array = np.array(orig_tile)

    multiplier = 8 / math.log(np.max(orig_tile_array))

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


# Values for uint8 masks
MASK_UNKNOWN = 50
MASK_POSITIVE = 200
MASK_NEGATIVE = 150
MASK_BACKGROUND = 0
MASK_CELL = 255
MASK_CELL_POSITIVE = 201
MASK_CELL_NEGATIVE = 151
MASK_BOUNDARY_POSITIVE = 202
MASK_BOUNDARY_NEGATIVE = 152


@jit(nopython=True)
def in_bounds(array, index):
    return index[0] >= 0 and index[0] < array.shape[0] and index[1] >= 0 and index[1] < array.shape[1]


def create_posneg_mask(seg, thresh):
    """Create a mask of positive and negative pixels."""

    cell = np.logical_and(np.add(seg[:,:,0], seg[:,:,2], dtype=np.uint16) > thresh, seg[:,:,1] <= 80)
    pos = np.logical_and(cell, seg[:,:,0] >= seg[:,:,2])
    neg = np.logical_xor(cell, pos)

    mask = np.full(seg.shape[0:2], MASK_UNKNOWN, dtype=np.uint8)
    mask[pos] = MASK_POSITIVE
    mask[neg] = MASK_NEGATIVE

    return mask


@jit(nopython=True)
def mark_background(mask):
    """Mask all background pixels by 4-connected region growing unknown boundary pixels."""

    seeds = []
    for i in range(mask.shape[0]):
        if mask[i, 0] == MASK_UNKNOWN:
            seeds.append((i, 0))
        if mask[i, mask.shape[1]-1] == MASK_UNKNOWN:
            seeds.append((i, mask.shape[1]-1))
    for j in range(mask.shape[1]):
        if mask[0, j] == MASK_UNKNOWN:
            seeds.append((0, j))
        if mask[mask.shape[0]-1, j] == MASK_UNKNOWN:
            seeds.append((mask.shape[0]-1, j))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(seeds) > 0:
        seed = seeds.pop()
        if mask[seed] == MASK_UNKNOWN:
            mask[seed] = MASK_BACKGROUND
            for n in neighbors:
                idx = (seed[0] + n[0], seed[1] + n[1])
                if in_bounds(mask, idx) and mask[idx] == MASK_UNKNOWN:
                    seeds.append(idx)


@jit(nopython=True)
def compute_cell_classification(mask, marker, size_thresh, marker_thresh, size_thresh_upper = None):
    """
    Compute the mapping of the mask to positive and negative cell classification.

    Parameters
    ==========
    mask: 2D uint8 numpy array with pixels labeled as positive, negative, background, or unknown.
          After the function executes, the pixels will be labeled as background or cell/boundary pos/neg.
    marker: 2D uint8 numpy array with the restained marker values
    size_thresh: Lower size threshold in pixels. Only include cells larger than this count.
    size_thresh_upper: Upper size threshold in pixels, or None. Only include cells smaller than this count.
    marker_thresh: Classify cell as positive if any marker value within the cell is above this threshold.

    Returns
    =======
    Dictionary with the following values:
        num_total (integer) -- total number of cells in the image
        num_pos (integer) -- number of positive cells in the image
        num_neg (integer) -- number of negative calles in the image
        percent_pos (floating point) -- percentage of positive cells to all cells (IHC score)
    """

    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    border_neighbors = [(0, -1), (-1, 0), (1, 0), (0, 1)]
    positive_cell_count, negative_cell_count = 0, 0

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == MASK_POSITIVE or mask[y, x] == MASK_NEGATIVE:
                seeds = [(y, x)]
                cell_coords = []
                count = 1
                count_posneg = 1 if mask[y, x] != MASK_UNKNOWN else 0
                count_positive = 1 if mask[y, x] == MASK_POSITIVE else 0
                max_marker = marker[y, x] if marker is not None else 0
                mask[y, x] = MASK_CELL
                cell_coords.append((y, x))

                while len(seeds) > 0:
                    seed = seeds.pop()
                    for n in neighbors:
                        idx = (seed[0] + n[0], seed[1] + n[1])
                        if in_bounds(mask, idx) and (mask[idx] == MASK_POSITIVE or mask[idx] == MASK_NEGATIVE or mask[idx] == MASK_UNKNOWN):
                            seeds.append(idx)
                            if mask[idx] == MASK_POSITIVE:
                                count_positive += 1
                            if mask[idx] != MASK_UNKNOWN:
                                count_posneg += 1
                            if marker is not None and marker[idx] > max_marker:
                                max_marker = marker[idx]
                            mask[idx] = MASK_CELL
                            cell_coords.append(idx)
                            count += 1

                if count > size_thresh and (size_thresh_upper is None or count < size_thresh_upper):
                    if (count_positive/count_posneg) >= 0.5 or max_marker > marker_thresh:
                        fill_value = MASK_CELL_POSITIVE
                        border_value = MASK_BOUNDARY_POSITIVE
                        positive_cell_count += 1
                    else:
                        fill_value = MASK_CELL_NEGATIVE
                        border_value = MASK_BOUNDARY_NEGATIVE
                        negative_cell_count += 1
                else:
                    fill_value = MASK_BACKGROUND
                    border_value = MASK_BACKGROUND

                for coord in cell_coords:
                    is_boundary = False
                    for n in border_neighbors:
                        idx = (coord[0] + n[0], coord[1] + n[1])
                        if in_bounds(mask, idx) and mask[idx] == MASK_BACKGROUND:
                            is_boundary = True
                            break
                    if is_boundary:
                        mask[coord] = border_value
                    else:
                        mask[coord] = fill_value

    counts = {
        'num_total': positive_cell_count + negative_cell_count,
        'num_pos': positive_cell_count,
        'num_neg': negative_cell_count,
    }
    return counts


@jit(nopython=True)
def enlarge_cell_boundaries(mask):
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == MASK_BOUNDARY_POSITIVE or mask[y, x] == MASK_BOUNDARY_NEGATIVE:
                value = MASK_POSITIVE if mask[y, x] == MASK_BOUNDARY_POSITIVE else MASK_NEGATIVE
                for n in neighbors:
                    idx = (y + n[0], x + n[1])
                    if in_bounds(mask, idx) and mask[idx] != MASK_BOUNDARY_POSITIVE and mask[idx] != MASK_BOUNDARY_NEGATIVE:
                        mask[idx] = value
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == MASK_POSITIVE:
                mask[y, x] = MASK_BOUNDARY_POSITIVE
            elif mask[y, x] == MASK_NEGATIVE:
                mask[y, x] = MASK_BOUNDARY_NEGATIVE


@jit(nopython=True)
def compute_cell_sizes(mask):
    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    sizes = []

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == MASK_POSITIVE or mask[y, x] == MASK_NEGATIVE:
                seeds = [(y, x)]
                count = 1
                mask[y, x] = MASK_CELL_POSITIVE if mask[y, x] == MASK_POSITIVE else MASK_CELL_NEGATIVE

                while len(seeds) > 0:
                    seed = seeds.pop()
                    for n in neighbors:
                        idx = (seed[0] + n[0], seed[1] + n[1])
                        if in_bounds(mask, idx) and (mask[idx] == MASK_POSITIVE or mask[idx] == MASK_NEGATIVE or mask[idx] == MASK_UNKNOWN):
                            seeds.append(idx)
                            if mask[idx] == MASK_POSITIVE:
                                mask[idx] = MASK_CELL_POSITIVE
                            elif mask[idx] == MASK_NEGATIVE:
                                mask[idx] = MASK_CELL_NEGATIVE
                            else:
                                mask[idx] = MASK_CELL
                            count += 1

                sizes.append(count)

    return sizes


@jit(nopython=True)
def create_kde(values, count, bandwidth = 1.0):
    gaussian_denom_inv = 1 / math.sqrt(2 * math.pi);
    max_value = max(values) + 1;
    step = max_value / count;
    n = values.shape[0];
    h = bandwidth;
    h_inv = 1 / h;
    kde = np.zeros(count, dtype=np.float32)

    for i in range(count):
        x = i * step
        total = 0
        for j in range(n):
            val = (x - values[j]) * h_inv;
            total += math.exp(-(val*val/2)) * gaussian_denom_inv; # Gaussian
        kde[i] = total / (n*h);

    return kde, step


def calc_default_size_thresh(mask, resolution):
    sizes = compute_cell_sizes(mask)
    mask[mask == MASK_CELL_POSITIVE] = MASK_POSITIVE
    mask[mask == MASK_CELL_NEGATIVE] = MASK_NEGATIVE
    mask[mask == MASK_CELL] = MASK_UNKNOWN

    if len(sizes) > 0:
        kde, step = create_kde(np.sqrt(sizes), 500)
        idx = 1
        for i in range(1, kde.shape[0]-1):
            if kde[i] < kde[i-1] and kde[i] < kde[i+1]:
                idx = i
                break
        thresh_sqrt = (idx - 1) * step

        allowed_range_sqrt = (4, 7, 10) # [min, default, max] for default sqrt size thresh at 40x
        if resolution == '20x':
            allowed_range_sqrt = (3, 4, 6)
        elif resolution == '10x':
            allowed_range_sqrt = (2, 2, 3)

        if thresh_sqrt < allowed_range_sqrt[0]:
            thresh_sqrt = allowed_range_sqrt[0]
        elif thresh_sqrt > allowed_range_sqrt[2]:
            thresh_sqrt = allowed_range_sqrt[1]

        return round(thresh_sqrt * thresh_sqrt)

    else:
        return 0


def calc_default_marker_thresh(marker):
    if marker is not None:
        nonzero = marker[marker != 0]
        marker_range = (round(np.percentile(nonzero, 0.1)), round(np.percentile(nonzero, 99.9))) if nonzero.shape[0] > 0 else (0, 0)
        return round((marker_range[1] - marker_range[0]) * 0.9) + marker_range[0]
    else:
        return 0


def compute_results(orig, seg, marker, resolution=None, seg_thresh=150, size_thresh='auto', marker_thresh='auto', size_thresh_upper=None):
    mask = create_posneg_mask(seg, seg_thresh)
    mark_background(mask)

    if size_thresh == 'auto':
        size_thresh = calc_default_size_thresh(mask, resolution)
    if marker_thresh is None:
        marker_thresh = 0
        marker = None
    elif marker_thresh == 'auto':
        marker_thresh = calc_default_marker_thresh(marker)

    counts = compute_cell_classification(mask, marker, size_thresh, marker_thresh, size_thresh_upper)
    enlarge_cell_boundaries(mask)

    scoring = {
        'num_total': counts['num_total'],
        'num_pos': counts['num_pos'],
        'num_neg': counts['num_neg'],
        'percent_pos': round(counts['num_pos'] / counts['num_total'] * 100, 1) if counts['num_pos'] > 0 else 0,
        'prob_thresh': seg_thresh,
        'size_thresh': size_thresh,
        'size_thresh_upper': size_thresh_upper,
        'marker_thresh': marker_thresh if marker is not None else None,
    }

    overlay = np.copy(orig)
    overlay[mask == MASK_BOUNDARY_POSITIVE] = (255, 0, 0)
    overlay[mask == MASK_BOUNDARY_NEGATIVE] = (0, 0, 255)

    refined = np.zeros_like(seg)
    refined[mask == MASK_CELL_POSITIVE, 0] = 255
    refined[mask == MASK_CELL_NEGATIVE, 2] = 255
    refined[mask == MASK_BOUNDARY_POSITIVE, 1] = 255
    refined[mask == MASK_BOUNDARY_NEGATIVE, 1] = 255

    return overlay, refined, scoring
