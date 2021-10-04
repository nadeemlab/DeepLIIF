import os
import cv2
import random
import numpy as np
from deepliif.options.processing_options import ProcessingOptions


def prepare_dataset_for_training(input_dir, dataset_dir, validation_ratio=0.2):
    """
    Preparing Data for Training:
    This function, first, creates the train and validation directories inside the given dataset directory.
    Then it reads all images in the folder and saves the pairs in the train or validation directory, based on the given validation_ratio.
    *** for training, you need to have paired data including IHC, Hematoxylin Channel, mpIF DAPI, mpIF Lap2, mpIF marker, and segmentation mask in the input directory ***

    :param input_dir: Path to the input images.
    :param dataset_dir: Path to the dataset directory. The function automatically creates the train and validation directories inside of this directory.
    :param validation_ratio: The ratio of the number of the images in the validation set to the total number of images.
    :return:
    """
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    if not os.path.exists(train_dir):
        os.makedires(train_dir)
        os.makedires(val_dir)
    images = os.listdir(input_dir)
    for img in images:
        if 'IHC' in img:
            IHC_image = cv2.resize(cv2.imread(os.path.join(input_dir, img)), (512, 512))
            Hema_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'Hematoxylin'))), (512, 512))
            DAPI_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'DAPI'))), (512, 512))
            Lap2_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'Lap2'))), (512, 512))
            Marker_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'Marker'))), (512, 512))
            Seg_image = cv2.resize(cv2.imread(os.path.join(input_dir, img.replace('IHC', 'Seg'))), (512, 512))


            save_dir = train_dir
            if random.random() < validation_ratio:
                save_dir = val_dir
            cv2.imwrite(os.path.join(save_dir, img), np.concatenate([IHC_image, Hema_image, DAPI_image, Lap2_image, Marker_image, Seg_image], 1))


if __name__ == '__main__':
    opt = ProcessingOptions().parse()   # get training options
    prepare_dataset_for_training(input_dir=opt.input_dir, dataset_dir=opt.output_dir, validation_ratio=opt.validation_ratio)

