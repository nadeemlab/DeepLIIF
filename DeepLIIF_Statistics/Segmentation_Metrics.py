import collections

import numpy as np
import cv2
import os
from numba import jit
from skimage import measure
import time
from PostProcessSegmentationMask import positive_negative_masks


@jit(nopython=True)
def compute_metrics_gpu(mask_img, gt_img, image_size):
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            if mask_img[i, j] > 0 and gt_img[i, j] > 0:
                TP += 1
            elif mask_img[i, j] > 0 and gt_img[i, j] == 0:
                FP += 1
            elif mask_img[i, j] == 0 and gt_img[i, j] > 0:
                FN += 1
            elif mask_img[i, j] == 0 and gt_img[i, j] == 0:
                TN += 1

    smooth = 0.0001
    if TP == 0:
        if np.count_nonzero(gt_img) > 0 or FP > 0:
            IOU, precision, recall, Dice, f1, pixAcc = 0, 0, 0, 0, 0, 0
        else:
            IOU, precision, recall, Dice, f1, pixAcc = 1, 1, 1, 1, 1, 1
    else:
        IOU = (TP) / (TP + FP + FN)
        precision = (TP) / (TP + FP)
        recall = (TP) / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        Dice = (2 * TP) / (2 * TP + FP + FN)
        pixAcc = (TP + TN) / (TP + TN + FP + FN)
    return IOU, precision, recall, f1, Dice, pixAcc


def compute_metrics(mask_img, gt_img):
    smooth = 0.0001
    intesection_TP = np.logical_and(gt_img, mask_img)
    intesection_FN = np.logical_and(gt_img, 1 - mask_img)
    intesection_FP = np.logical_and(1 - gt_img, mask_img)
    intesection_TN = np.logical_and(1 - gt_img, 1 - mask_img)
    union = np.logical_or(gt_img, mask_img)

    iou_score = (np.sum(intesection_TP) + smooth) / (np.sum(union) + smooth)
    precision_score = (np.sum(intesection_TP) + smooth) / (np.sum(intesection_TP) + np.sum(intesection_FP) + smooth)
    recall_score = (np.sum(intesection_TP) + smooth) / (np.sum(intesection_TP) + np.sum(intesection_FN) + smooth)
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    dice_score = (2 * np.sum(intesection_TP) + smooth) / (2 * np.sum(intesection_TP) + np.sum(intesection_FN) + np.sum(intesection_FP) + smooth)
    pix_acc_score = (np.sum(intesection_TP) + np.sum(intesection_TN) + smooth) / (np.sum(intesection_TP) + np.sum(intesection_TN) + np.sum(intesection_FN) + np.sum(intesection_FP) + smooth)
    return iou_score, precision_score, recall_score, f1_score, dice_score, pix_acc_score


def compute_jaccard_index(set_1, set_2):
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)


def compute_aji(gt_image, mask_image):
    label_image_gt = measure.label(gt_image, background=0)
    label_image_mask = measure.label(mask_image, background=0)
    gt_labels = np.unique(label_image_gt)
    mask_labels = np.unique(label_image_mask)
    mask_components = []
    mask_marked = []
    for mask_label in mask_labels:
        if mask_label == 0:
            continue
        comp = np.zeros((gt_image.shape[0], gt_image.shape[1]), dtype=np.uint8)
        comp[label_image_mask == mask_label] = 1
        mask_components.append(comp)
        mask_marked.append(False)

    total_intersection = 0
    total_union = 0
    total_U = 0
    for gt_label in gt_labels:
        if gt_label == 0:
            continue
        comp = np.zeros((gt_image.shape[0], gt_image.shape[1]), dtype=np.uint8)
        comp[label_image_gt == gt_label] = 1
        intersection = [0, 0, 0]    # index, intersection, union
        for i in range(len(mask_components)):
            if not mask_marked[i]:
                comp_intersection = np.sum(np.logical_and(comp, mask_components[i]))
                if comp_intersection > intersection[1]:
                    union = np.sum(np.logical_or(comp, mask_components[i]))
                    intersection = [i, comp_intersection, union]
        if intersection[1] > 0:
            mask_marked[intersection[0]] = True
            total_intersection += intersection[1]
            total_union += intersection[2]
    for i in range(len(mask_marked)):
        if not mask_marked[i]:
            total_U += np.sum(mask_components[i])
    aji = total_intersection / (total_union + total_U) if (total_union + total_U) > 0 else 0
    return aji


def compute_segmentation_metrics(gt_dir, model_dir, model_name, image_size=512, thresh=100, boundary_thresh=100, small_object_size=20, raw_segmentation=True):
    info_dict = []
    metrics = collections.defaultdict(float)
    images = os.listdir(model_dir)
    counter = 0
    postfix = '_Seg.png' if raw_segmentation else '_SegRefined.png'
    for mask_name in images:
        if postfix in mask_name:
            counter += 1

            mask_image = cv2.cvtColor(cv2.imread(os.path.join(model_dir, mask_name)), cv2.COLOR_BGR2RGB)
            mask_image = cv2.resize(mask_image, (image_size, image_size))
            if not raw_segmentation:
                positive_mask = mask_image[:, :, 0]
                negative_mask = mask_image[:, :, 2]
            else:
                positive_mask, negative_mask = positive_negative_masks(mask_image, thresh, boundary_thresh, small_object_size)

            positive_mask[positive_mask > 0] = 1
            negative_mask[negative_mask > 0] = 1

            gt_img = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, mask_name.replace(postfix, '.png'))), cv2.COLOR_BGR2RGB)
            gt_img = cv2.resize(gt_img, (image_size, image_size))

            positive_gt = gt_img[:, :, 0]
            negative_gt = gt_img[:, :, 2]

            positive_gt[positive_gt > 0] = 1
            negative_gt[negative_gt > 0] = 1

            # AJI_positive = compute_aji(positive_gt, positive_mask)
            # start = time.time()
            IOU_positive, precision_positive, recall_positive, f1_positive, Dice_positive, pixAcc_positive = compute_metrics_gpu(positive_mask, positive_gt, gt_img.shape)
            # end = time.time()
            # print(end - start)
            # print('GPU: ', IOU_positive, precision_positive, recall_positive, f1_positive, Dice_positive, pixAcc_positive)
            # start = time.time()
            # IOU_positive, precision_positive, recall_positive, f1_positive, Dice_positive, pixAcc_positive = compute_metrics(positive_mask, positive_gt)
            # end = time.time()
            # print(end - start)
            # print('CPU: ', end - start, IOU_positive, precision_positive, recall_positive, f1_positive, Dice_positive, pixAcc_positive)

            # AJI_negative = compute_aji(negative_gt, negative_mask)
            # start = time.time()
            IOU_negative, precision_negative, recall_negative, f1_negative, Dice_negative, pixAcc_negative = compute_metrics_gpu(negative_mask, negative_gt, gt_img.shape)
            # end = time.time()
            # print('GPU: ', IOU_negative, precision_negative, recall_negative, f1_negative, Dice_negative, pixAcc_negative)
            # start = time.time()
            # IOU_negative, precision_negative, recall_negative, f1_negative, Dice_negative, pixAcc_negative = compute_metrics(negative_mask, negative_gt)
            # end = time.time()
            # print('CPU: ', end - start, IOU_negative, precision_negative, recall_negative, f1_negative, Dice_negative, pixAcc_negative)

            info_dict.append({'Model': model_name,
                              'image_name': mask_name,
                              'cell_type': 'Positive',
                              'precision': precision_positive * 100,
                              'recall': recall_positive * 100,
                              'f1': f1_positive * 100,
                              'Dice': Dice_positive * 100,
                              'IOU': IOU_positive * 100,
                              'PixAcc': pixAcc_positive * 100
                              # 'AJI': AJI_positive * 100
                              })

            info_dict.append({'Model': model_name,
                              'image_name': mask_name,
                              'cell_type': 'Negative',
                              'precision': precision_negative * 100,
                              'recall': recall_negative * 100,
                              'f1': f1_negative * 100,
                              'Dice': Dice_negative * 100,
                              'IOU': IOU_negative * 100,
                              'PixAcc': pixAcc_negative * 100
                              # 'AJI': AJI_negative * 100
                              })

            precision = (precision_positive * 100 + precision_negative * 100) / 2
            recall = (recall_positive * 100 + recall_negative * 100) / 2
            f1 = (f1_positive * 100 + f1_negative * 100) / 2
            Dice = (Dice_positive * 100 + Dice_negative * 100) / 2
            IOU = (IOU_positive * 100 + IOU_negative * 100) / 2
            pixAcc = (pixAcc_positive * 100 + pixAcc_negative * 100) / 2
            # AJI = (AJI_positive * 100 + AJI_negative * 100) / 2

            info_dict.append({'Model': model_name,
                              'image_name': mask_name,
                              'cell_type': 'Mean',
                              'precision': precision,
                              'recall': recall,
                              'f1': f1,
                              'Dice': Dice,
                              'IOU': IOU,
                              'PixAcc': pixAcc,
                              # 'AJI': AJI
                              })

            metrics['precision'] += precision
            metrics['precision_positive'] += precision_positive
            metrics['precision_negative'] += precision_negative

            metrics['recall'] += recall
            metrics['recall_positive'] += recall_positive
            metrics['recall_negative'] += recall_negative

            metrics['f1'] += f1
            metrics['f1_positive'] += f1_positive
            metrics['f1_negative'] += f1_negative

            metrics['Dice'] += Dice
            metrics['Dice_positive'] += Dice_positive
            metrics['Dice_negative'] += Dice_negative

            metrics['IOU'] += IOU
            metrics['IOU_positive'] += IOU_positive
            metrics['IOU_negative'] += IOU_negative

            metrics['PixAcc'] += pixAcc
            metrics['PixAcc_positive'] += pixAcc_positive
            metrics['PixAcc_negative'] += pixAcc_negative

            # metrics['AJI'] += AJI
            # metrics['AJI_positive'] += AJI_positive
            # metrics['AJI_negative'] += AJI_negative

    for key in metrics:
        metrics[key] /= counter

    return info_dict, metrics

