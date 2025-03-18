import math
import warnings

import numpy as np
from numba import jit, typed
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


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


# Default postprocessing values
DEFAULT_SEG_THRESH = 150
DEFAULT_NOISE_THRESH = 4

# Values for uint8 label masks
LABEL_UNKNOWN = 50
LABEL_POSITIVE = 200
LABEL_NEGATIVE = 150
LABEL_BACKGROUND = 0
LABEL_CELL = 100
LABEL_BORDER_POS = 220
LABEL_BORDER_NEG = 170
LABEL_BORDER_POS2 = 221
LABEL_BORDER_NEG2 = 171


def to_array(img, grayscale=False):
    """
    Convert a color image to an array of pixels.

    Parameters
    ----------
    img : Image | ndarray
        Image to convert. If an array is provided instead, it is used directly.
    grayscale : bool
        Whether the input image should be converted to grayscale
        by taking the maximum channel value for each pixel.

    Returns
    -------
    ndarray
        A 2D (grayscale) or 3D array with the pixels of the converted image.
    """

    if isinstance(img, Image.Image):
        img = np.asarray(img) if img.mode == 'RGB' else np.asarray(img.convert('RGB'))
    if grayscale and len(img.shape) == 3:
        img = img.max(axis=-1)
    return img


@jit(nopython=True)
def in_bounds(array, index):
    """
    Check if an index is valid for an array.

    Parameters
    ----------
    array : ndarray
        2D array.
    index : tuple
        2-element tuple with index values matching the array shape (e.g., for a pixel array where
        array.shape[0] is height and array.shape[1] is width, then index will be in (y, x) order).

    Returns
    -------
    bool
        Whether or not the index is within the bounds of the array.
    """

    return index[0] >= 0 and index[0] < array.shape[0] and index[1] >= 0 and index[1] < array.shape[1]


@jit(nopython=True)
def create_posneg_mask(seg, thresh):
    """
    Create a mask of positive and negative pixels from the segmentation image.

    Parameters
    ----------
    seg : ndarray
        3D uint8 array (2D image w/ 3 channels) with segmentation probabilities.
    thresh : int
        Threshold to use in determining if a pixel should be labeled as positive/negative.

    Returns
    -------
    ndarray
        2D uint8 array of mask values with every pixel labeled as unknown, positive, or negative.
    """

    mask = np.full(seg.shape[0:2], LABEL_UNKNOWN, dtype=np.uint8)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if seg[y, x, 0] + seg[y, x, 2] > thresh and seg[y, x, 1] <= 80:
                if seg[y, x, 0] >= seg[y, x, 2]:
                    mask[y, x] = LABEL_POSITIVE
                else:
                    mask[y, x] = LABEL_NEGATIVE

    return mask


@jit(nopython=True)
def mark_background(mask):
    """
    Mask all background pixels in-place by 4-connected region growing unknown boundary pixels.

    Parameters
    ----------
    mask: ndarray
        2D uint8 array with pixels labeled as positive, negative, or unknown.
        After the function executes, the pixels will be labeled as background, positive, negative, or unknown.
    """

    seeds = []
    for i in range(mask.shape[0]):
        if mask[i, 0] == LABEL_UNKNOWN:
            seeds.append((i, 0))
        if mask[i, mask.shape[1]-1] == LABEL_UNKNOWN:
            seeds.append((i, mask.shape[1]-1))
    for j in range(mask.shape[1]):
        if mask[0, j] == LABEL_UNKNOWN:
            seeds.append((0, j))
        if mask[mask.shape[0]-1, j] == LABEL_UNKNOWN:
            seeds.append((mask.shape[0]-1, j))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(seeds) > 0:
        seed = seeds.pop()
        if mask[seed] == LABEL_UNKNOWN:
            mask[seed] = LABEL_BACKGROUND
            for n in neighbors:
                idx = (seed[0] + n[0], seed[1] + n[1])
                if in_bounds(mask, idx) and mask[idx] == LABEL_UNKNOWN:
                    seeds.append(idx)


@jit(nopython=True)
def compute_cell_mapping(mask, marker, noise_thresh):
    """
    Compute the mapping from mask to positive and negative cells.

    Parameters
    ----------
    mask : ndarray
        2D uint8 array with pixels labeled as positive, negative, background, or unknown.
        After the function executes, the pixels will be labeled as background or cell.
    marker : ndarray
        2D uint8 array with the inferred marker values.

    Returns
    -------
    typed.List[tuple] :
        Cell data as a list of 7-element tuples with the following:
        [0] - number of pixels in the cell
        [1] - whether the cell is positive (True) or negative (False)
        [2] - marker value for the cell
        [3-4] - first pixel coordinates (x, y) of the cell cluster
        [5-6] - centroid of the cell (x, y)
    """

    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    cells = typed.List()

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] != LABEL_BACKGROUND and mask[y, x] != LABEL_CELL:
                seeds = [(y, x)]
                count = 1
                count_positive = 1 if mask[y, x] == LABEL_POSITIVE else 0
                count_negative = 1 if mask[y, x] == LABEL_NEGATIVE else 0
                max_marker = marker[y, x] if marker is not None else 0
                mask[y, x] = LABEL_CELL
                center_y = y
                center_x = x

                while len(seeds) > 0:
                    seed = seeds.pop()
                    for n in neighbors:
                        idx = (seed[0] + n[0], seed[1] + n[1])
                        if in_bounds(mask, idx) and mask[idx] != LABEL_BACKGROUND and mask[idx] != LABEL_CELL:
                            seeds.append(idx)
                            if mask[idx] == LABEL_POSITIVE:
                                count_positive += 1
                            elif mask[idx] == LABEL_NEGATIVE:
                                count_negative += 1
                            if marker is not None and marker[idx] > max_marker:
                                max_marker = marker[idx]
                            mask[idx] = LABEL_CELL
                            center_y += idx[0]
                            center_x += idx[1]
                            count += 1

                if count > noise_thresh:
                    center_y = int(round(center_y / count))
                    center_x = int(round(center_x / count))
                    positive = True if count_positive >= count_negative else False
                    cells.append((count, positive, max_marker, x, y, center_x, center_y))

    return cells


def get_cells_info(seg, marker, resolution, noise_thresh, seg_thresh):
    """
    Find all cells in the segmentation image that are larger than the noise threshold.

    Parameters
    ----------
    seg : Image | ndarray
        Inferred segmentation map image.
    marker : Image | ndarray
        Inferred marker image.
    resolution: string
        The resolution/magnification of the original image.  Valid values are '10x', '20x', or '40x'.
    noise_thresh : int
        Threshold for tiny noise to ignore (include only cells larger than this value).
    seg_thresh : int
        Threshold to use in determining if a pixel should be labeled as positive/negative.

    Returns
    -------
    ndarray :
        Label mask.
    typed.List[tuple] :
        Cell data as a list of 7-element tuples.
    dict :
        Calculated default values.
    """

    seg = to_array(seg)
    if marker is not None:
        marker = to_array(marker, True)
    mask = create_posneg_mask(seg, seg_thresh)
    mark_background(mask)
    cellsinfo = compute_cell_mapping(mask, marker, noise_thresh)

    defaults = {}
    sizes = np.zeros(len(cellsinfo), dtype=np.int64)
    for i in range(len(cellsinfo)):
        sizes[i] = cellsinfo[i][0]
    defaults['size_thresh'] = calculate_default_size_threshold(sizes, resolution)
    if marker is not None:
        defaults['marker_thresh'] = calculate_default_marker_threshold(marker)

    return mask, cellsinfo, defaults


@jit(nopython=True)
def create_kde(values, count, bandwidth = 1.0):
    """
    Create Gaussian kernel density estimate (KDE) for values with count number of bins.

    Parameters
    ----------
    values : list
        Input values.
    count : int
        Number of bins for KDE.
    bandwidth: float
        Bandwidth (smoothing parameter) for KDE.

    Returns
    -------
    ndarray :
        Kernel density estimate.
    float :
        Step size.
    """

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


@jit(nopython=True)
def calculate_default_size_threshold(cell_sizes, resolution='40x'):
    """
    Calculate a default size threshold to exclude small cells.

    Parameters
    ----------
    cell_sizes : ndarray
        1D array of cell sizes.
    resolution : string
        The resolution/magnification of the original image.  Valid values are '10x', '20x', or '40x'.

    Returns
    -------
    int :
        Default size threshold.
    """

    if cell_sizes.shape[0] > 1:
        kde, step = create_kde(np.sqrt(cell_sizes), 500)
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


def calculate_stain_range(stain):
    """
    Calculate the range of the 99.9 percentile of non-zero pixels in the stain image.

    Parameters
    ----------
    stain : ndarray
        2D uint8 array (image).

    Returns
    -------
    tuple[int] :
        2-element tuple with 0.1 and 99.9 percentile values.
    """

    nonzero = stain[stain != 0]
    if nonzero.shape[0] > 0:
        return (round(np.percentile(nonzero, 0.1)), round(np.percentile(nonzero, 99.9)))
    else:
        return (0, 0)


def calculate_default_marker_threshold(marker):
    """
    Calculate a default threshold for a marker image as 90% of the 99.9 percentile range.

    Parameters
    ----------
    marker : ndarray
        2D uint8 array (image).

    Results
    -------
    int :
        Default marker threshold.
    """

    marker_range = calculate_stain_range(marker)
    return round((marker_range[1] - marker_range[0]) * 0.9) + marker_range[0]


@jit(nopython=True)
def get_cell_boundary(mask, x, y):
    """
    Get the boundary contour pixels for a cell, and also the bounding box.
    The provided starting (x, y) pixel must be the first pixel encountered
    from the top left, whether found by searching via rows or columns.

    Parameters
    ----------
    mask : ndarray
        2D uint8 ndarray of background and cell labels
    x : int
        x-coordinate of the first pixel of the cell
    y : int
        y-coordinate of the first pixel of the cell

    Returns
    -------
    list :
        Bounding box of the cell as a list of two 2-element tuples.
    list :
        All boundary pixels (x, y) going clockwise from first point.
    """

    w = mask.shape[1]
    h = mask.shape[0]

    if not in_bounds(mask, (y, x)) or mask[y, x] == LABEL_BACKGROUND:
        return None, None

    '''
    In normal xy coordinates, check neighbors clockwise in the following order:
     0 1 2
     7 - 3
     6 5 4
    List neighbors in xy coordinates, but xy are switched to yx for numpy array access.
    '''
    neighbors = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)] # (dx, dy)
    neighbors *= 2

    boundary = [(x, y)]
    min_x = x
    min_y = y
    max_x = x
    max_y = y

    # Go counter-clockwise to find previous pixel
    idx = 6
    while idx >= 0:
        nx = x + neighbors[idx][0]
        ny = y + neighbors[idx][1]
        if in_bounds(mask, (ny, nx)) and mask[ny, nx] != LABEL_BACKGROUND:
            break
        idx -= 1
    if idx < 0:
        return [(x, y), (x, y)], [(x, y)]

    px = x + neighbors[idx][0]
    py = y + neighbors[idx][1]
    boundary = [(px, py), (x, y)]

    # Go clockwise to get border pixels in order
    while True:
        dx = px - x
        dy = py - y
        idx = neighbors.index((dx, dy)) + 1
        while True:
            nx = x + neighbors[idx][0]
            ny = y + neighbors[idx][1]
            if in_bounds(mask, (ny, nx)) and mask[ny, nx] != LABEL_BACKGROUND:
                break
            idx += 1
        px = x
        py = y
        x = nx
        y = ny
        boundary.append((x, y))

        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y

        if px == boundary[0][0] and py == boundary[0][1] and x == boundary[1][0] and y == boundary[1][1]:
            break

    return [(min_x, min_y), (max_x, max_y)], boundary[1:-1]


def make_simple_contour(points):
    """
    Make a simplified version of a contour by removing redundant points within
    straight lines (i.e., each straight line segment will be reduced to contain
    only the first and last point for that segment).  It is assumed that the
    vectors between points are all one of the eight pixel neighbor directions.
    This means that either one of the x- or y-direction must be zero, or if
    both values are non-zero then they must be the same (i.e., [1,0], [0,2],
    [2,2], and [3,3] are vall valid direction vectors, but [1,2] is not).
    The input parameter of contour points must contain at least one point.

    Parameters
    ----------
    points : list
        Contour of boundary points (x, y).

    Returns
    -------
    list :
        Simplified contour of boundary points (x, y).
    """

    # always keep first point
    simple = [(points[0][0], points[0][1])]

    # if only one point in contour, then done
    if len(points) == 1:
        return simple

    # for all middle points (exclude first and last)
    for i in range(1, len(points) - 1):
        dx0 = points[i][0] - points[i-1][0]
        dy0 = points[i][1] - points[i-1][1]
        dx1 = points[i+1][0] - points[i][0]
        dy1 = points[i+1][1] - points[i][1]
        same_dx = (dx0 == dx1) or (dx0 > 0 and dx1 > 0) or (dx0 < 0 and dx1 < 0)
        same_dy = (dy0 == dy1) or (dy0 > 0 and dy1 > 0) or (dy0 < 0 and dy1 < 0)
        if not same_dx or not same_dy:
            simple.append((points[i][0], points[i][1]))

    # for last point (calculate p[n]-p[n-1] and p[0]-p[n])
    dx0 = points[-1][0] - points[-2][0]
    dy0 = points[-1][1] - points[-2][1]
    dx1 = points[0][0] - points[-1][0]
    dy1 = points[0][1] - points[-1][1]
    same_dx = (dx0 == dx1) or (dx0 > 0 and dx1 > 0) or (dx0 < 0 and dx1 < 0)
    same_dy = (dy0 == dy1) or (dy0 > 0 and dy1 > 0) or (dy0 < 0 and dy1 < 0)
    if not same_dx or not same_dy:
        simple.append((points[-1][0], points[-1][1]))

    return simple


def make_full_contour(points):
    """
    Convert a simplified contour to a complete pixel-by-pixel contour
    (i.e., every point is an 8-neighbor of the previous point).  It is
    assumed that vectors between points are all one of the eight pixel
    neighbor directions.  The input parameter of contour points must
    contain at least one point.

    Parameters
    ----------
    points : list
        Contour of boundary points (x, y).

    Returns
    -------
    list :
        Full contour of boundary points (x, y).
    """

    # start with first point
    full = [(points[0][0], points[0][1])]

    # for all remaining points:
    for i in range(1, len(points)):

        # calculate direction from last full point to current input point
        dx = points[i][0] - full[-1][0]
        dy = points[i][1] - full[-1][1]
        dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        dy = 1 if dy > 0 else (-1 if dy < 0 else 0)

        # add direction to last full point until reach current input point
        while full[-1][0] != points[i][0] or full[-1][1] != points[i][1]:
            full.append((full[-1][0] + dx, full[-1][1] + dy))

    # calculate direction from last full point until first point
    dx = full[0][0] - full[-1][0]
    dy = full[0][1] - full[-1][1]
    dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    dy = 1 if dy > 0 else (-1 if dy < 0 else 0)

    # add direction to last full point until reach first point (avoid duplicate)
    while full[-1][0] + dx != full[0][0] or full[-1][1] + dy != full[0][1]:
        full.append((full[-1][0] + dx, full[-1][1] + dy))

    return full


def to_base92(values, min_len=1):
    """
    Convert integer values to base92, offset by 35 for printable ASCII values
    (i.e., output characters for [0-91] are in the range [35-126]).
    All encodings will have the same number of characters, of at least min_len
    (i.e., smaller in length encodings will be padded with 35).

    Parameters
    ----------
    values : int | list[int] | tuple[int]
        The integer value(s) to a base92 ASCII encoding.
    min_len : int
        The minimum number of characters for the base92 ASCII encoding of each value.

    Returns
    -------
    string | list[string] :
        The converted value(s).
    """

    multi = type(values) is list or type(values) is tuple
    if not multi:
        values = [values]

    results = []
    for val in values:
        res = ''
        while val > 0:
            res += chr((val % 92) + 35)
            val //= 92
        results.append(res)

    max_len = max(len(r) for r in results)
    fixed_len = max_len if max_len > min_len else min_len

    for i in range(len(results)):
        while len(results[i]) < fixed_len:
            results[i] += chr(35)
        results[i] = results[i][::-1]

    if not multi:
        results = results[0]
    return results


def from_base92(val):
    """
    Convert from base92 ASCII encoding to integer value.

    Parameters
    ----------
    val : string
        The base92 ASCII encoded value.

    Returns
    -------
    int :
        The converted value.
    """

    res = 0
    for v in val:
        res *= 92
        res += (ord(v) - 35)
    return res


def encode_cell_data_v4(data):
    """
    Encode as v4 the provided cell data to string.

    Parameters
    ----------
    data : dict
        Dictionary of cell data.

    Returns
    -------
    string :
        Encoded cell data as a single ASCII string.
    """

    cell = '' # encoded cell data as string

    # encode cell size (in pixels)
    size = to_base92(data['size'])
    size_len = len(size)
    cell += size

    # encode cell classification (pos/neg) and marker value
    positive = int(data['positive'])
    marker = data['marker']
    classification = (marker * 2) + positive
    cell += to_base92(classification, 2)

    # encode anchor point (bbox top left) and extent (bbox bottom right)
    topleft = to_base92(data['bbox'][0])
    topleft_len = len(topleft[0])
    cell += topleft[0]
    cell += topleft[1]

    # encode extent (bbox bottom right), centroid, and first boundary contour point
    # as offsets from the previously encoded anchor point (bbox top left)
    x = data['bbox'][0][0]
    y = data['bbox'][0][1]
    offsets = [*data['bbox'][1], *data['centroid'], *data['boundary'][0]]
    for j in range(0, len(offsets), 2):
        offsets[j] -= x
        offsets[j+1] -= y
    offsets = to_base92(offsets)
    offsets_len = len(offsets[0])
    cell += offsets[0]
    cell += offsets[1]
    cell += offsets[2]
    cell += offsets[3]
    cell += offsets[4]
    cell += offsets[5]

    # encode number of chars for variable length encodations and prepend to cell string
    encoded_lens = ((size_len - 1) * 16) + ((topleft_len - 1) * 4) + (offsets_len - 1)
    encoded_lens = chr(encoded_lens + 35)
    cell = encoded_lens + cell

    # encode remaining boundary contour points using Freeman chain code
    # Freeman chain code:
    #   3  2  1
    #    \ | /
    #   4-- --0
    #    / | \
    #   5  6  7
    boundary = ''
    for j in range(1, len(data['boundary'])):
        dx = data['boundary'][j][0] - data['boundary'][j-1][0]
        dy = data['boundary'][j][1] - data['boundary'][j-1][1]
        if dx >= 1 and dy == 0:
            direction = 0
        elif dx >= 1 and dy <= -1:
            direction = 1
        elif dx == 0 and dy <= -1:
            direction = 2
        elif dx <= -1 and dy <= -1:
            direction = 3
        elif dx <= -1 and dy == 0:
            direction = 4
        elif dx <= -1 and dy >= 1:
            direction = 5
        elif dx == 0 and dy >= 1:
            direction = 6
        elif dx >= 1 and dy >= 1:
            direction = 7
        else: # this should not (cannot) happen, so if it does, then exit
            exit()
        distance = max(abs(dx), abs(dy))
        if distance == 0: # this should not (cannot) happen, but if duplicate point, then skip
            continue
        while distance > 10:
            encoded = (10 * 8) + direction
            boundary += chr(encoded + 35)
            distance -= 10
        encoded = (distance * 8) + direction
        boundary += chr(encoded + 35)
    cell += boundary

    return cell


def decode_cell_data_v4(cell):
    """
    Decode v4 encoded cell string and return dictionary of cell data.

    Parameters
    ----------
    cell : string
        Encoded cell data as a single ASCII string.

    Returns
    -------
    dict :
        Dictionary with the decoded cell data.
    """

    data = {} # decoded cell data

    # decode number of chars for variable length encodations
    n = ord(cell[0]) - 35
    ns = (n // 16) + 1      # num chars for cell size
    na = ((n // 4) % 4) + 1 # num chars for anchor coordinates
    no = (n % 4) + 1        # num chars for offset coordinates

    # decode cell size (in pixels)
    data['size'] = from_base92(cell[1:1+ns])

    # decode cell classification (pos/neg) and marker value
    classification = from_base92(cell[1+ns:3+ns])
    data['positive'] = bool(classification % 2)
    data['marker'] = classification // 2

    # decode anchor point (bbox top left) and extent (bbox bottom right)
    x = from_base92(cell[3+ns:3+ns+na])
    y = from_base92(cell[3+ns+na:3+ns+2*na])
    ex = x + from_base92(cell[3+ns+2*na:3+ns+2*na+no])
    ey = y + from_base92(cell[3+ns+2*na+no:3+ns+2*na+2*no])
    data['bbox'] = [(x, y), (ex, ey)]

    # decode centroid point
    cx = x + from_base92(cell[3+ns+2*na+2*no:3+ns+2*na+3*no])
    cy = y + from_base92(cell[3+ns+2*na+3*no:3+ns+2*na+4*no])
    data['centroid'] = (cx, cy)

    # decode first boundary contour points
    bx = x + from_base92(cell[3+ns+2*na+4*no:3+ns+2*na+5*no])
    by = y + from_base92(cell[3+ns+2*na+5*no:3+ns+2*na+6*no])
    data['boundary'] = [(bx, by)]

    # directions using Freeman chain code
    freeman = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]

    # decode remaining boundary contour points
    prev_direction = None
    for c in cell[3+ns+2*na+6*no:]:
        point = ord(c) - 35
        distance = point // 8
        direction = freeman[point % 8]
        movement = (direction[0]*distance, direction[1]*distance)
        px = data['boundary'][-1][0] + movement[0]
        py = data['boundary'][-1][1] + movement[1]
        if direction == prev_direction:
            data['boundary'].pop()
        data['boundary'].append((px, py))
        prev_direction = direction

    return data


@jit(nopython=True)
def create_cell_classification(mask, cellsinfo,
                               size_thresh=0,
                               marker_thresh=None,
                               size_thresh_upper=None):
    """
    Create final cell classification in-place for the mask and
    calculate counts for positive and negative cell counts.

    Parameters
    ----------
    mask : ndarray
        2D uint8 array label map.
    cellsinfo : list
        Information about each cell found from the segmentation.
    size_thresh : int
        Include only cells larger than this size.
    marker_thresh : int
        Make cell positive if marker value is above this threshold (override original classification).
    size_thresh_upper : int
        Include only cells smaller than this size.

    Results
    -------
    dict :
        Dictionary with the counts of positive, negative, and total cells.
    """

    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    border_neighbors = [(0, -1), (-1, 0), (1, 0), (0, 1)]
    num_pos, num_neg = 0, 0
    if marker_thresh is None:
        marker_thresh = 255

    for cell in cellsinfo:
        if cell[0] > size_thresh and (size_thresh_upper is None or cell[0] < size_thresh_upper):
            is_pos = True if cell[1] or cell[2] > marker_thresh else False
            if is_pos:
                label = LABEL_POSITIVE
                label_border = LABEL_BORDER_POS
                num_pos += 1
            else:
                label = LABEL_NEGATIVE
                label_border = LABEL_BORDER_NEG
                num_neg += 1

            x = cell[3]
            y = cell[4]
            mask[y,x] = label_border
            seeds = [(y, x)]

            while len(seeds) > 0:
                seed = seeds.pop()
                for n in neighbors:
                    idx = (seed[0] + n[0], seed[1] + n[1])
                    if in_bounds(mask, idx) and mask[idx] == LABEL_CELL:
                        seeds.append(idx)
                        is_boundary = False
                        for n in border_neighbors:
                            idx2 = (idx[0] + n[0], idx[1] + n[1])
                            if in_bounds(mask, idx2) and mask[idx2] == LABEL_BACKGROUND:
                                is_boundary = True
                                break
                        if is_boundary:
                            mask[idx] = label_border
                        else:
                            mask[idx] = label

    num_total = num_pos + num_neg
    return {
        'num_total': num_total,
        'num_pos': num_pos,
        'num_neg': num_neg,
    }


@jit(nopython=True)
def enlarge_cell_boundaries(mask):
    """
    Enlarge cell boundaries in-place in mask by one pixel in each direction.

    Parameters
    ----------
    mask : ndarray
        2D uint8 label map.
    """

    neighbors = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == LABEL_BORDER_POS or mask[y, x] == LABEL_BORDER_NEG:
                value = LABEL_BORDER_POS2 if mask[y, x] == LABEL_BORDER_POS else LABEL_BORDER_NEG2
                for n in neighbors:
                    idx = (y + n[0], x + n[1])
                    if in_bounds(mask, idx) and mask[idx] != LABEL_BORDER_POS and mask[idx] != LABEL_BORDER_NEG:
                        mask[idx] = value

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == LABEL_BORDER_POS2:
                mask[y, x] = LABEL_BORDER_POS
            elif mask[y, x] == LABEL_BORDER_NEG2:
                mask[y, x] = LABEL_BORDER_NEG


@jit(nopython=True)
def create_final_images(overlay, mask):
    """
    Create the final overlay (in-place) and refined images from the mask.
    The 'overlay' parameter is the image on which to create the overlay,
    which will be done in-place.

    Parameters
    ----------
    overlay : ndarray
        3D uint8 array (2D image of 3 channels).
        Generally, a copy of the original input image.
    mask : ndarray
        2D uint8 label map.

    Returns
    -------
    ndarray :
        3D uint8 array (2D image of 3 channels) containing the overlaid image.
    ndarray :
        3D uint8 array (2D image of 3 channels) containing the refined segmentation image.
    """

    refined = np.zeros_like(overlay)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == LABEL_BORDER_POS:
                overlay[y, x] = (255, 0, 0)
                refined[y, x, 1] = 255
            elif mask[y, x] == LABEL_BORDER_NEG:
                overlay[y, x] = (0, 0, 255)
                refined[y, x, 1] = 255
            elif mask[y, x] == LABEL_POSITIVE:
                refined[y, x, 0] = 255
            elif mask[y, x] == LABEL_NEGATIVE:
                refined[y, x, 2] = 255

    return overlay, refined


@jit(nopython=True)
def fill_cells(mask):
    """
    For a mask with cell outlines, fill in the center of the cells in-place.
    The cell outlines must surround cell entirely, including image border pixels.

    Parameters
    ----------
    mask : ndarray
        2D uint8 label map.
    """

    for y in range(mask.shape[0]):
        for x in range(1, mask.shape[1]):
            if mask[y, x] == LABEL_UNKNOWN:
                if mask[y, x-1] == LABEL_BORDER_POS or mask[y, x-1] == LABEL_POSITIVE:
                    mask[y, x] = LABEL_POSITIVE
                else:
                    mask[y, x] = LABEL_NEGATIVE


def compute_cell_results(seg, marker, resolution, version=3,
                         seg_thresh=DEFAULT_SEG_THRESH,
                         noise_thresh=DEFAULT_NOISE_THRESH):
    """
    Perform postprocessing to compute individual cell results.

    Parameters
    ----------
    seg : Image | ndarray
        Inferred segmentation map image.
    marker : Image | ndarray
        Inferred marker image.
    resolution : string
        The resolution/magnification of the original image.  Valid values are '10x', '20x', or '40x'.
    version : int
        Version of the cell data (valid values are 3 and 4).
    seg_thresh : int
        Threshold to use in determining if a pixel should be labeled as positive/negative.
    noise_thresh : int
        Threshold for tiny noise to ignore (include only cells larger than this value).

    Returns
    -------
    dict :
        Individual cell data and other associated values.
    """

    if version not in [3, 4]:
        warnings.warn('Invalid cell data version provided, defaulting to version 3.')
        version = 3

    mask, cellsinfo, defaults = get_cells_info(seg, marker, resolution, noise_thresh, seg_thresh)

    cells = []
    for cell in cellsinfo:
        bbox, boundary = get_cell_boundary(mask, cell[3], cell[4])
        data = {
            'size': cell[0],
            'positive': cell[1],
            'marker': cell[2],
            'bbox': bbox,
            'centroid': (cell[5], cell[6]),
            'boundary': make_simple_contour(boundary),
        }
        if version == 4:
            data = encode_cell_data_v4(data)
        cells.append(data)

    results = {
        'cells': cells,
        'settings': {
            'default_marker_thresh': defaults['marker_thresh'] if 'marker_thresh' in defaults else None,
            'default_size_thresh': defaults['size_thresh'],
            'noise_thresh': noise_thresh,
            'seg_thresh': seg_thresh,
        },
        'dataVersion': version,
    }

    return results


def compute_final_results(orig, seg, marker, resolution,
                          size_thresh='default',
                          marker_thresh=None,
                          size_thresh_upper=None,
                          seg_thresh=DEFAULT_SEG_THRESH,
                          noise_thresh=DEFAULT_NOISE_THRESH):
    """
    Perform postprocessing to compute final count and image results.

    Parameters
    ----------
    orig : Image | ndarray
        Original input image.
    seg : Image | ndarray
        Inferred segmentation map image.
    marker : Image | ndarray
        Inferred marker image.
    resolution : string
        The resolution/magnification of the original image.  Valid values are '10x', '20x', or '40x'.
    size_thresh : int
        Include only cells larger than this size.
    marker_thresh : int
        Make cell positive if marker value is above this threshold (override original classification).
    size_thresh_upper : int
        Include only cells smaller than this size.
    seg_thresh : int
        Threshold to use in determining if a pixel should be labeled as positive/negative.
    noise_thresh : int
        Threshold for tiny noise to ignore (include only cells larger than this value).

    Returns
    -------
    ndarray :
        3D uint8 array (2D image of 3 channels) containing the overlaid image.
    ndarray :
        3D uint8 array (2D image of 3 channels) containing the refined segmentation image.
    dict :
        Dictionary with scoring and settings information.
    """

    mask, cellsinfo, defaults = get_cells_info(seg, marker, resolution, noise_thresh, seg_thresh)

    if size_thresh is None:
        size_thresh = 0
    elif size_thresh == 'default':
        size_thresh = defaults['size_thresh']
    if marker_thresh == 'default':
        marker_thresh = defaults['marker_thresh']

    counts = create_cell_classification(mask, cellsinfo, size_thresh, marker_thresh, size_thresh_upper)
    enlarge_cell_boundaries(mask)
    overlay, refined = create_final_images(np.array(orig), mask)

    scoring = {
        'num_total': counts['num_total'],
        'num_pos': counts['num_pos'],
        'num_neg': counts['num_neg'],
        'percent_pos': round(counts['num_pos'] / counts['num_total'] * 100, 1) if counts['num_pos'] > 0 else 0,
        'seg_thresh': seg_thresh,
        'size_thresh': size_thresh,
        'size_thresh_upper': size_thresh_upper,
        'marker_thresh': marker_thresh if marker is not None else None,
    }

    return overlay, refined, scoring


def cells_to_final_results(data, orig,
                           size_thresh='default',
                           marker_thresh=None,
                           size_thresh_upper=None):
    """
    Compute final count and image results from previously postprocessed cell results.

    Parameters
    ----------
    data : dict
        Individual cell data and associated values generated by the 'compute_cell_results' function.
    orig : Image | ndarray
        Original input image.
    size_thresh : int
        Include only cells larger than this size.
    marker_thresh : int
        Make cell positive if marker value is above this threshold (override original classification).
    size_thresh_upper : int
        Include only cells smaller than this size.

    Returns
    -------
    ndarray :
        3D uint8 array (2D image of 3 channels) containing the overlaid image.
    ndarray :
        3D uint8 array (2D image of 3 channels) containing the refined segmentation image.
    dict :
        Dictionary with scoring and settings information.
    """

    orig = np.array(orig)
    mask = np.full(orig.shape[0:2], LABEL_UNKNOWN, dtype=np.uint8)
    num_pos, num_neg = 0, 0

    if size_thresh is None:
        size_thresh = 0
    elif size_thresh == 'default':
        size_thresh = data['settings']['default_size_thresh']
    if marker_thresh == 'default':
        marker_thresh = data['settings']['default_marker_thresh']

    for cell in data['cells']:
        if data['dataVersion'] == 4:
            c = decode_cell_data_v4(cell)
        else:
            c = cell
        if c['size'] > size_thresh and (size_thresh_upper is None or c['size'] < size_thresh_upper):
            if c['positive'] or (marker_thresh is not None and c['marker'] > marker_thresh):
                num_pos += 1
                label = LABEL_BORDER_POS
            else:
                num_neg += 1
                label = LABEL_BORDER_NEG
            border = make_full_contour(c['boundary'])
            for b in border:
                mask[b[1], b[0]] = label

    mark_background(mask)
    fill_cells(mask)

    enlarge_cell_boundaries(mask)
    overlay, refined = create_final_images(np.array(orig), mask)

    num_total = num_pos + num_neg
    scoring = {
        'num_total': num_total,
        'num_pos': num_pos,
        'num_neg': num_neg,
        'percent_pos': round(num_pos / num_total * 100, 1) if num_pos > 0 else 0,
        'seg_thresh': data['settings']['seg_thresh'],
        'size_thresh': size_thresh,
        'size_thresh_upper': size_thresh_upper,
        'marker_thresh': marker_thresh,
    }

    return overlay, refined, scoring
