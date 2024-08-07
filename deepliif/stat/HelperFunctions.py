import csv
import json
import math
import os
import h5py
import numpy as np
# import staintools
from matplotlib.colors import LinearSegmentedColormap
from numba import jit
from scipy import ndimage
import cv2
import random
from skimage.feature import peak_local_max
import skimage.segmentation
from skimage.morphology import watershed as ws, remove_small_objects
import matplotlib.pyplot as plt

def remove_small_objects_from_image(red_channel, min_size=100):
    red_channel_copy = red_channel.copy()
    red_channel_copy[red_channel > 0] = 1
    red_channel_copy = red_channel_copy.astype(np.bool)
    removed_red_channel = remove_small_objects(red_channel_copy, min_size=min_size).astype(np.uint8)
    red_channel[removed_red_channel == 0] = 0
    return red_channel


def read_BC_detection_mask(img_name, data_type='test'):
    img_type = img_name.split('.')[-1]
    base_dir = '/home/parmida/Downloads/BCData'
    annotations_dir = os.path.join(base_dir, 'annotations')
    images_dir = os.path.join(base_dir, 'images')
    negative_dir = os.path.join(annotations_dir, data_type, 'negative')
    positive_dir = os.path.join(annotations_dir, data_type, 'positive')
    images_dir = os.path.join(images_dir, data_type)
    print(os.path.join(negative_dir, img_name.replace('.png', '.h5')))
    gt_file_negative = h5py.File(os.path.join(negative_dir, img_name.replace('.' + img_type, '.h5')))
    coordinates_negative = np.asarray(gt_file_negative['coordinates'])
    gt_file_positive = h5py.File(os.path.join(positive_dir, img_name.replace('.' + img_type, '.h5')))
    coordinates_positive = np.asarray(gt_file_positive['coordinates'])

    positive_mask = np.zeros((640, 640), dtype=np.uint8)
    negative_mask = np.zeros((640, 640), dtype=np.uint8)
    for coord in coordinates_positive:
        positive_mask[coord[1], coord[0]] = 255

    for coord in coordinates_negative:
        negative_mask[coord[1], coord[0]] = 255

    return positive_mask, negative_mask

def read_BC_detection_point(img_name, data_type='test'):
    img_type = img_name.split('.')[-1]
    base_dir = '/home/parmida/Downloads/BCData'
    annotations_dir = os.path.join(base_dir, 'annotations')
    images_dir = os.path.join(base_dir, 'images')
    negative_dir = os.path.join(annotations_dir, data_type, 'negative')
    positive_dir = os.path.join(annotations_dir, data_type, 'positive')
    images_dir = os.path.join(images_dir, data_type)
    print(os.path.join(negative_dir, img_name.replace('.png', '.h5')))
    gt_file_negative = h5py.File(os.path.join(negative_dir, img_name.replace('.' + img_type, '.h5')))
    coordinates_negative = np.asarray(gt_file_negative['coordinates'])
    gt_file_positive = h5py.File(os.path.join(positive_dir, img_name.replace('.' + img_type, '.h5')))
    coordinates_positive = np.asarray(gt_file_positive['coordinates'])

    return coordinates_positive, coordinates_negative



def compute_TP_FP_of_each_class(image, marked_class):
    labeled, nr_objects = ndimage.label(image > 0)
    TP = 0
    FP = 0
    for c in range(1, nr_objects):
        component = np.zeros_like(image)
        component[labeled == c] = image[labeled == c]
        component = cv2.morphologyEx(component, cv2.MORPH_DILATE, kernel=np.ones((5, 5)), iterations=1)
        TP, FP = compute_component_TP_FP(component, marked_class, TP, FP)
    return TP, FP


@jit(nopython=True)
def compute_component_TP_FP(component, marked_class, TP, FP):
    indices = np.nonzero(component)
    cell_flag = False
    for i in range(len(indices[0])):
        if marked_class[indices[0][i], indices[1][i]] > 0:
                TP += 1
                cell_flag = True
    if not cell_flag:
        FP += 1
    return TP, FP


def compute_precision_recall_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, F1


def mark_Shiraz_image_with_markers(image, immunopositive, immunonegative, immunoTIL):
    marked_image = image.copy()
    positive = cv2.morphologyEx(immunopositive, cv2.MORPH_DILATE, kernel=np.ones((5,5)))
    negative = cv2.morphologyEx(immunonegative, cv2.MORPH_DILATE, kernel=np.ones((5,5)))
    TIL = cv2.morphologyEx(immunoTIL, cv2.MORPH_DILATE, kernel=np.ones((5,5)))
    marked_image[positive > 0] = (0,0,255)
    marked_image[negative > 0] = (255,0,0)
    marked_image[TIL > 0] = (0,255,0)
    return marked_image

def read_NuClick_mask(img_name, dir_type='Train'):
    # image_dir = '/home/parmida/Pathology/IHC_Nuclick/images/Train'
    mask_dir = '/home/parmida/Pathology/IHC_Nuclick/masks/' + dir_type
    mask = np.load(os.path.join(mask_dir, img_name.replace('.png', '.npy')))
    labeled_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    final_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    labels_no = np.max(mask) + 1
    color_dict = {}
    color_dict[0] = (0, 0, 0)
    for i in range(1, labels_no):
        color_dict[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            labeled_mask[i, j] = color_dict[mask[i, j]]
            final_mask[i, j] = (0, 0, 0)
    boundaries = cv2.Canny(labeled_mask, 100, 200)
    # boundaries = cv2.dilate(boundaries, kernel=np.ones((3, 3), np.uint8))
    labeled_mask_bw = cv2.cvtColor(labeled_mask, cv2.COLOR_RGB2GRAY)
    final_mask[labeled_mask_bw > 0] = (0, 0, 255)
    # final_mask[labeled_mask_bw == 0] = (0, 0, 255)
    # final_mask[boundaries > 0] = (255, 255, 255)

    contours, hierarchy = cv2.findContours(boundaries,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(final_mask, contours, -1, (255, 255, 255), 2)
    # cv2.imshow('labeled_mask', labeled_mask)
    # cv2.imshow('boundaries', boundaries)
    # cv2.imshow('final', final_mask)
    # cv2.waitKey(0)
    boundaries[boundaries > 0] = 255
    return final_mask


def get_detection_points(seg_img):
    seg_img = cv2.resize(seg_img, (640, 640))
    det_img = np.zeros_like(seg_img)
    thresh = 50
    det_img[np.logical_and(seg_img[:, :, 2] > thresh, seg_img[:, :, 2] > seg_img[:, :, 0] + 50)] = (0, 0, 255)
    det_img[np.logical_and(seg_img[:, :, 0] > thresh, seg_img[:, :, 0] >= seg_img[:, :, 2])] = (255, 0, 0)
    det_img[seg_img[:, :, 1] > thresh] = 0
    det_img[:, :, 0] = remove_small_objects_from_image(det_img[:, :, 0], 80)
    det_img[:, :, 2] = remove_small_objects_from_image(det_img[:, :, 2], 80)
    det_img[:, :, 0] = ndimage.binary_fill_holes(det_img[:, :, 0]).astype(np.uint8) * 255
    det_img[:, :, 2] = ndimage.binary_fill_holes(det_img[:, :, 2]).astype(np.uint8) * 255
    # det_img[:, :, 0] = cv2.morphologyEx(det_img[:, :, 0], cv2.MORPH_ERODE, kernel=np.ones((3, 3)), iterations=2)
    # det_img[:, :, 2] = cv2.morphologyEx(det_img[:, :, 2], cv2.MORPH_ERODE, kernel=np.ones((3, 3)), iterations=2)
    # cv2.imshow('det_img', det_img)
    det_img = np.squeeze(det_img).astype(np.uint8)
    cells = watershed(det_img)
    final_cells = []
    positive_points = []
    negative_points = []
    seen = np.zeros((seg_img.shape[0], seg_img.shape[1]), dtype=np.uint8)
    for i in range(len(cells)):
        p1 = cells[i]
        x1, y1, c1 = int(p1[1]), int(p1[0]), int(p1[2])
        flag = False
        seen[x1][y1] = 1
        for j in range(len(cells)):
            p2 = cells[j]
            x2, y2, c2 = int(p2[1]), int(p2[0]), int(p2[2])
            if seen[x2][y2] == 0:
                if abs(x1 - x2) < 20 and abs(y1 - y2) < 20:
                    flag = True
                    # new_cell = int((x1 + x2) / 2), int((y1 + y2) / 2), int((c1 + c2)/2)
                    # final_cells.append(new_cell)
        if not flag:
            final_cells.append(p1)
            if c1 == 2:
                positive_points.append((x1, y1))
            elif c1 == 0:
                negative_points.append((x1, y1))
    return final_cells, positive_points, negative_points

def detect_circles(component, output):
    gray_blurred = cv2.blur(component, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=100,
                                        param2=20, minRadius=1, maxRadius=40)
    # circles = cv2.HoughCircles(component, cv2.HOUGH_GRADIENT, 1, 10)
    # ensure at least some circles were found
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(output, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            # cv2.circle(output, (a, b), 1, (0, 0, 255), 3)
            # cv2.imshow("Detected Circle", output)
            # cv2.waitKey(0)
            # cv2.imshow('component', component)

def watershed(pred):
    cells=[]
    for ch in range(3):
        gray=pred[:,:,ch]
        D = ndimage.distance_transform_edt(gray)
        localMax = peak_local_max(D, indices=False, min_distance=10,exclude_border=False,labels=gray)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = ws(-D, markers, mask=gray)
        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
            c = max(cnts, key=cv2.contourArea)
            ((x, y), _) = cv2.minEnclosingCircle(c)
            cells.append([x,y,ch])
    return np.array(cells)

def read_PathoNet_data(img_addr):
    print(img_addr)
    points = np.loadtxt(img_addr.replace('.jpg', '_points.txt'))
    image = cv2.imread(img_addr)
    # positive_mask = np.zeros((640, 640), dtype=np.uint8)
    # negative_mask = np.zeros((640, 640), dtype=np.uint8)
    positive_points = []
    negative_points = []
    for p in points:
        if int(p[2]) == 1:
            image[int(p[1]), int(p[0])] = (255, 0, 255)
            negative_points.append((int(p[1]), int(p[0])))
        else:
            image[int(p[1]), int(p[0])] = (0, 255, 255)
            positive_points.append((int(p[1]), int(p[0])))
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    return positive_points, negative_points


def crop_modalities(input_dir, img_name, img_types, location, size, output_dir):
    for img_type in img_types:
        image = cv2.imread(os.path.join(input_dir, img_name + img_type + '.png'))
        crop = image[location[0]:location[0] + size[0], location[1]: location[1] + size[1]]
        cv2.imwrite(os.path.join(output_dir, 'MYC_' + img_type + '.png'), crop)


def read_mask_rcnn_segmentation_masks(input_dir, image_size):
    images = os.listdir(input_dir)
    masks = {}
    for img in images:
        if '.png' in img and len(img.split('_')) > 5:
            print(img)
            splitted = img.split('_')
            image_name = ''
            for i in range(0, len(splitted) - 3):
                image_name += splitted[i] + '_'
            image_name += splitted[-3]
            cell_type = 'positive' if splitted[-2] == 1 else 'negative'
            image = cv2.imread(os.path.join(input_dir, img))
            image = cv2.resize(image, (image_size, image_size))
            image_bw = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            image_bw[image[:,:,0] > 250] = 1
            image_bw[image[:,:,1] > 250] = 1
            image_bw[image[:,:,2] > 250] = 1
            if image_name not in masks.keys():
                masks[image_name] = {'positive': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), 'negative': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), 'binary': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)}
            masks[image_name][cell_type][image_bw > 0] = 1
            masks[image_name]['binary'][image_bw > 0] = 1
    return masks


def read_mask_rcnn_detection_masks(input_dir, image_size):
    images = os.listdir(input_dir)
    masks = {}
    for img in images:
        if '_' in img and '.png' in img:
            splitted = img.split('_')
            image_name = ''
            for i in range(0, len(splitted) - 3):
                image_name += splitted[i] + '_'
            image_name += splitted[-3]
            cell_type = 'positive' if splitted[-2] == '1' else 'negative'
            image = cv2.imread(os.path.join(input_dir, img))
            image = cv2.resize(image, (image_size, image_size))
            image_bw = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            image_bw[image[:,:,0] > 250] = 1
            image_bw[image[:,:,1] > 250] = 1
            image_bw[image[:,:,2] > 250] = 1
            points = np.nonzero(image_bw)
            x = points[0]
            y = points[1]
            bounding_box = [np.min(x), np.min(y), np.max(x), np.max(y)]
            center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
            if image_name not in masks.keys():
                masks[image_name] = {'positive': [], 'negative': [], 'binary': []}
            masks[image_name][cell_type].append(center)
            masks[image_name]['binary'].append(center)
    return masks


def read_unetplusplus_unet(input_npy, npy_path):
    # results = np.load(input_dir)
    imgs = np.load(input_npy)
    with open(npy_path + 'dict_name.txt', 'r') as f:
        names = json.load(f)
    res_path = os.path.dirname(input_npy)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        print(img.shape)
        plt.imsave(os.path.join(res_path, names[str(i)] + '.png'), (img * 255).astype(np.uint8))


def read_Unet_plusplus_detection_masks(input_dir, image_size):
    images = os.listdir(input_dir)
    masks = {}
    for img in images:
        if '.png' in img:
            image = cv2.imread(os.path.join(input_dir, img))
            image = cv2.resize(image, (image_size, image_size))
            # cv2.imshow('image1', image)
            img = img.replace('.png', '')
            new_image = np.zeros_like(image)
            new_image[image[:,:,0] > 150] = (255, 0, 0)
            new_image[image[:,:,2] > 150] = (0, 0, 255)
            # cv2.imshow('image2', new_image)
            det_img = np.squeeze(new_image).astype(np.uint8)
            cells = watershed(det_img)
            final_cells = []
            positive_points = []
            negative_points = []
            seen = np.zeros((new_image.shape[0], new_image.shape[1]), dtype=np.uint8)
            for i in range(len(cells)):
                p1 = cells[i]
                x1, y1, c1 = int(p1[1]), int(p1[0]), int(p1[2])
                flag = False
                seen[x1][y1] = 1
                for j in range(len(cells)):
                    p2 = cells[j]
                    x2, y2, c2 = int(p2[1]), int(p2[0]), int(p2[2])
                    if seen[x2][y2] == 0:
                        if abs(x1 - x2) < 20 and abs(y1 - y2) < 20:
                            flag = True
                            # new_cell = int((x1 + x2) / 2), int((y1 + y2) / 2), int((c1 + c2)/2)
                            # final_cells.append(new_cell)
                if not flag:
                    final_cells.append(p1)
                    if c1 == 2:
                        positive_points.append((x1, y1))
                    elif c1 == 0:
                        negative_points.append((x1, y1))

            # for p in positive_points:
            #     new_image[p[0]-5:p[0]+5, p[1]-5:p[1]+5] = (255,0,255)
            # for p in negative_points:
            #     new_image[p[0]-5:p[0]+5, p[1]-5:p[1]+5] = (0,255,255)

            # cv2.imshow('image3', new_image)
            # cv2.waitKey(0)
            masks[img] = {'positive': [], 'negative': []}
            masks[img]['positive'] = positive_points
            masks[img]['negative'] = negative_points
            # masks[img]['binary'].append(center)
    return masks


def read_Unet_plusplus_segmentation_masks(input_dir, image_size):
    images = os.listdir(input_dir)
    masks = {}
    for img in images:
        if '.png' in img:
            image = cv2.imread(os.path.join(input_dir, img))
            image = cv2.resize(image, (image_size, image_size))
            img = img.replace('.png', '')
            new_image = np.zeros_like(image)
            new_image[image[:,:,0] > 0] = (255, 0, 0)
            new_image[image[:,:,2] > 0] = (0, 0, 255)
            masks[img] = {'positive': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), 'negative': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), 'binary': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)}
            masks[img]['positive'][image[:,:,0] > 0] = 1
            masks[img]['negative'][image[:,:,2] > 0] = 1
            masks[img]['binary'][image[:,:,0] > 0] = 1
            masks[img]['binary'][image[:,:,2] > 0] = 1
    return masks

def read_DeepLIIF_segmentation_masks(input_dir, image_size, thresh=100):
    images = os.listdir(input_dir)
    masks = {}
    for img in images:
        if '_fake_B_5.png' in img:
            image = cv2.imread(os.path.join(input_dir, img))
            image = cv2.resize(image, (image_size, image_size))
            img = img.replace('_fake_B_5.png', '')
            new_image = np.zeros_like(image)
            new_image[np.logical_and(image[:,:,0] > thresh, image[:,:,0] > image[:,:,2])] = (255, 0, 0)
            new_image[np.logical_and(image[:,:,2] > thresh, image[:,:,2] >= image[:,:,0])] = (0, 0, 255)
            # new_image[image[:,:,1] > thresh] = 0
            # new_image[image[:,:,0] > 100] = (255, 0, 0)
            # new_image[image[:,:,2] > 100] = (0, 0, 255)
            # cv2.imshow('image', image)
            # cv2.imshow('new_image', new_image)
            # cv2.waitKey(0)

            positive_mask = new_image[:, :, 0]
            negative_mask = new_image[:, :, 2]
            positive_mask = ndimage.binary_fill_holes(positive_mask, structure=np.ones((5, 5))).astype(np.uint8)
            negative_mask = ndimage.binary_fill_holes(negative_mask, structure=np.ones((5, 5))).astype(np.uint8)
            positive_mask = remove_small_objects_from_image(positive_mask, 50)
            negative_mask = remove_small_objects_from_image(negative_mask, 50)
            positive_mask[negative_mask > 0] = 0
            masks[img] = {'positive': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), 'negative': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), 'binary': np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)}
            # masks[img]['positive'][image[:,:,0] > thresh] = 1
            # masks[img]['negative'][image[:,:,2] > thresh] = 1
            # masks[img]['binary'][image[:,:,0] > thresh] = 1
            # masks[img]['binary'][image[:,:,2] > thresh] = 1
            masks[img]['positive'][positive_mask > 0] = 1
            masks[img]['negative'][negative_mask > 0] = 1
            masks[img]['binary'][positive_mask > 0] = 1
            masks[img]['binary'][negative_mask > 0] = 1
    return masks


def read_ki67_detection_points(img):
    input_dir = '/media/parmida/Work/DetectionDataset/test_Ki67'
    image_size = 512
    image = cv2.imread(os.path.join(input_dir, img))
    image = image[:,5*512:]
    image = cv2.resize(image, (image_size, image_size))
    cv2.imshow('mask', image)
    # cv2.waitKey(0)
    positive_image = np.zeros((image.shape[0], image.shape[1]))
    positive_image[image[:, :, 2] > 0] = 1
    positive_mask = get_centers_of_objects(positive_image)

    negative_image = np.zeros((image.shape[0], image.shape[1]))
    negative_image[image[:, :, 0] > 0] = 1
    negative_mask = get_centers_of_objects(negative_image)

    return positive_mask, negative_mask


def get_centers_of_objects(image):
    mask = np.zeros((image.shape[0], image.shape[1]))
    labeled, nr_objects = ndimage.label(image > 0)
    for c in range(1, nr_objects):
        component = np.zeros_like(image)
        component[labeled == c] = image[labeled == c]

        points = np.nonzero(component)
        x = points[0]
        y = points[1]
        bounding_box = [np.min(x), np.min(y), np.max(x), np.max(y)]
        center = (int((bounding_box[0] + bounding_box[2]) / 2), int((bounding_box[1] + bounding_box[3]) / 2))
        mask[center[0], center[1]] = 255
    return mask


def read_Unet_plusplus_boundary_mask_image(img_addr, image_size):
    print(img_addr)
    image = cv2.cvtColor(cv2.imread(img_addr), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    new_image = np.zeros_like(image)
    res_image = np.zeros_like(image)
    new_image[image[:,:,0] > 100] = (255, 0, 0)
    new_image[image[:,:,2] > 100] = (0, 0, 255)
    positive_mask = new_image[:,:,0]
    negative_mask = new_image[:,:,2]
    contours, hierarchy = cv2.findContours(positive_mask,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(res_image, contours, -1, (255, 0, 0), 2)
    contours, hierarchy = cv2.findContours(negative_mask,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(res_image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(img_addr.replace('.png', '_Seg.png'), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    return res_image, new_image

def read_DeepLIIF_boundary_mask_image(img_addr, image_size, thresh=120):
    image = cv2.cvtColor(cv2.imread(img_addr), cv2.COLOR_BGR2RGB)
    # cv2.imshow('image', image)
    image = cv2.resize(image, (image_size, image_size))
    new_image = np.zeros_like(image)

    new_image[np.logical_and(image[:, :, 0] > thresh, image[:, :, 0] > image[:, :, 2])] = (255, 0, 0)
    new_image[np.logical_and(image[:, :, 2] > thresh, image[:, :, 2] >= image[:, :, 0])] = (0, 0, 255)

    new_image[image[:,:,1] > 80] = 0
    # new_image[image[:,:,0] > 100] = (255, 0, 0)
    # new_image[image[:,:,2] > 100] = (0, 0, 255)
    # cv2.imshow('image', image)
    # cv2.imshow('new_image', new_image)
    # cv2.waitKey(0)

    positive_mask = new_image[:, :, 0]
    negative_mask = new_image[:, :, 2]
    positive_mask = ndimage.binary_fill_holes(positive_mask, structure=np.ones((5, 5))).astype(np.uint8)
    negative_mask = ndimage.binary_fill_holes(negative_mask, structure=np.ones((5, 5))).astype(np.uint8)
    positive_mask = remove_small_objects_from_image(positive_mask, 50)
    negative_mask = remove_small_objects_from_image(negative_mask, 50)
    # positive_mask[negative_mask > 0] = 0
    negative_mask[positive_mask > 0] = 0

    # new_image[np.logical_and(image[:,:,0] > thresh, image[:,:,0] > image[:,:,2])] = (255, 0, 0)
    # new_image[np.logical_and(image[:,:,2] > thresh, image[:,:,2] >= image[:,:,0])] = (0, 0, 255)
    # new_image[image[:,:,1] > thresh] = 0
    #
    res_image = np.zeros_like(image)
    # positive_mask = new_image[:,:,0]
    # negative_mask = new_image[:,:,2]
    # # positive_mask = cv2.morphologyEx(positive_mask, cv2.MORPH_DILATE, kernel=np.ones((3,3)))
    # # negative_mask = cv2.morphologyEx(negative_mask, cv2.MORPH_DILATE, kernel=np.ones((3,3)))
    # positive_mask = ndimage.binary_fill_holes(positive_mask, structure=np.ones((5,5))).astype(np.uint8)
    # negative_mask = ndimage.binary_fill_holes(negative_mask, structure=np.ones((5,5))).astype(np.uint8)
    # positive_mask = remove_small_objects_from_image(positive_mask, 50)
    # negative_mask = remove_small_objects_from_image(negative_mask, 50)
    # # positive_mask[negative_mask > 0] = 0
    # negative_mask[positive_mask > 0] = 0
    # negative_mask = mask.copy()
    new_image = np.zeros_like(image)
    new_image[positive_mask > 0] = (255,0,0)
    new_image[negative_mask > 0] = (0,0,255)
    # cv2.imshow('positive_mask', positive_mask*255)
    # cv2.imshow('negative_mask', negative_mask*255)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(positive_mask,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(res_image, contours, -1, (255, 0, 0), 2)
    contours, hierarchy = cv2.findContours(negative_mask,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(res_image, contours, -1, (0, 0, 255), 2)
    return res_image, new_image



def create_log_area_mask_cell_type(predicted_mask, gt_mask, index=0, colormap='bwr'):
    smooth = 0.0001
    predicted = predicted_mask[:,:,index]
    gt = gt_mask[:,:,index]
    # gt[gt_mask[:,:,1] > 0]=0
    final_mask = np.zeros((predicted.shape[0], predicted.shape[1]))

    labeled, nr_objects = ndimage.label(predicted > 0)
    labeled_gt, nr_objects_gt = ndimage.label(gt > 0)
    for c in range(1, nr_objects):
        component = np.zeros_like(predicted)
        component[labeled == c] = predicted[labeled == c]
        nonzeros = np.nonzero(component)
        component_gt_size = 0
        # cv2.imshow('component', component)
        for i in range(len(nonzeros[0])):
            if gt[nonzeros[0][i], nonzeros[1][i]] > 0 and labeled_gt[nonzeros[0][i], nonzeros[1][i]] > 0:
                label = labeled_gt[nonzeros[0][i], nonzeros[1][i]]
                component_gt = np.zeros_like(gt)
                component_gt[labeled_gt == label] = gt[labeled_gt == label]
                nonzeros_gt = np.nonzero(component_gt)
                component_gt_size = len(nonzeros_gt[0])
                # print(component_gt_size)
                break
        # if component_gt_size > 0:
        component_size = len(nonzeros[0])
        # print('component_gt_size:', component_gt_size)
        # print('component_size:', component_size)
        if component_gt_size == 0:
            value = 5
            # print('yes')
        else:
            value = np.log2((component_size+smooth)/(component_gt_size+smooth))
            value = min(value, 2) if value >= 0 else max(value, -2)
        # print((len(nonzeros[0])+smooth)/(component_gt_size+smooth))
        # if value < 0:
        #     print(value)
        #     print(min(value, 2) if value >= 0 else max(value, -2))

        final_mask[component > 0] = value
        # cv2.waitKey(0)
        # final_mask[predicted == 0] = 0
    # color_range = np.max(final_mask)
    image_log = np.zeros_like(gt_mask)
    for i in range(0, final_mask.shape[0]):
        for j in range(0, final_mask.shape[1]):
            value = final_mask[i][j]
            if value == 5:
                # print('yes!!!!!!!')
                image_log[i, j] = (255, 255, 0)
            elif -0.5 <= value <= 0.5:
                image_log[i, j] = (255, 0, 0) if colormap == 'positive' else (0, 0, 255)
            elif value > 0.5:
                image_log[i, j] = (int(127.5/value), 0, 0) if colormap == 'positive' else (0, 0, int(127.5/value))
            elif value < -0.5:
                image_log[i, j] = (255, 255 - int(127.5/abs(value)), 255 - int(127.5/abs(value))) if colormap == 'positive' else (255 - int(127.5/abs(value)), 255 - int(127.5/abs(value)), 255)
                # print((255, 255 - int(127.5/abs(value)), 255 - int(127.5/abs(value))))
    # plt.tight_layout()
    # plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
    #
    # image_log = plt.imread('temp.png')
    # image_log = cv2.resize(image_log, (512, 512))
    # image_log[predicted_mask[:, :, index] == 0] = 0
    return image_log


def create_log_area_mask(predicted_mask, gt_mask):
    log_positive = create_log_area_mask_cell_type(predicted_mask, gt_mask, index=0, colormap='positive')
    log_negative = create_log_area_mask_cell_type(predicted_mask, gt_mask, index=2, colormap='negative')
    log_positive = (log_positive[:,:,:3] * 255).astype(np.uint8)
    log_negative = (log_negative[:,:,:3] * 255).astype(np.uint8)
    log_image = np.zeros_like(predicted_mask)
    log_image[predicted_mask[:,:,0] > 0] = log_positive[predicted_mask[:,:,0] > 0]
    log_image[predicted_mask[:,:,2] > 0] = log_negative[predicted_mask[:,:,2] > 0]
    return log_image



def create_color_map_image(colormap):
    image = np.zeros((400, 100))
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            image[j, i] = (j - 200) / 100
    colormap_image = np.zeros((400, 100, 3), dtype=np.uint8)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            value = image[i][j]
            if -0.5 <= value <= 0.5:
                colormap_image[i, j] = (255, 0, 0) if colormap == 'positive' else (0, 0, 255)
            elif value > 0.5:
                colormap_image[i, j] = (int(127.5/value), 0, 0) if colormap == 'positive' else (0, 0, int(127.5/value))
            elif value < -0.5:
                colormap_image[i, j] = (255, 255 - int(127.5/abs(value)), 255 - int(127.5/abs(value))) if colormap == 'positive' else (255 - int(127.5/abs(value)), 255 - int(127.5/abs(value)), 255)
    colormap_image = cv2.rotate(colormap_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return colormap_image


def read_image_write_crop_parts(input_dir, img_name, location, crop_size, output_dir):
    types = ['', '_DAPI', '_DAPILap2', '_Hema', '_Ki67', '_Seg_Aligned_Bound']
    for image_type in types:
        image = cv2.imread(os.path.join(input_dir, img_name + image_type + '.png'))
        crop = image[location[0]: location[0] + crop_size[0], location[1]: location[1] + crop_size[1]]
        cv2.imwrite(os.path.join(output_dir, img_name + image_type + '.png'), crop)


def overlay_ki67_on_DAPI(input_DAPI, input_ki67):
    # overlaid_image = np.zeros((input_DAPI.shape[0], input_DAPI.shape[1], 3), dtype=np.uint8)
    # overlaid_image[:,:,2] = input_ki67
    # overlaid_image[:,:,0] = input_DAPI
    # overlaid_image[:,:,1] = np.floor((input_DAPI + input_ki67) / 2)
    # overlaid_image[:,:,1] = input_DAPI

    # overlaid_image = cv2.addWeighted(input_DAPI, 0.6, input_Lap2, 0.4, 1)
    # overlaid_image = cv2.addWeighted(input_DAPI, 0.9, input_ki67, 0.1, 1)
    overlaid_image = input_DAPI.copy()
    overlaid_image[input_ki67[:,:,2] >= 30] = input_ki67[input_ki67[:,:,2] >= 30]
    # overlaid_image[input_ki67[:,:,2] < 30] = input_ki67[input_ki67[:,:,2] < 30] * 0.2 + overlaid_image[input_ki67[:,:,2]] * 0.8
    return overlaid_image


def count_cell_number(image, channel=1, thresh=0):
    mask = image[:, :, channel]
    labeled, nr_objects = ndimage.label(mask > thresh)
    return nr_objects

