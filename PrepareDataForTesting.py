import os
import cv2
import random
import numpy as np
from options.processing_options import ProcessingOptions



def prepare_dataset_for_testing(input_dir, dataset_dir):
    """
        Preparing Data for Testing:
        This function, first, creates the test directory inside the given dataset directory.
        Then it reads all images in the folder and saves pairs in the test directory.
        *** for testing, you only need to have IHC images in the input directory ***

        :param input_dir: Path to the input images.
        :param dataset_dir: Path to the dataset directory. The function automatically creates the train and validation directories inside of this directory.
        :return:
        """
    test_dir = os.path.join(dataset_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedires(test_dir)
    images = os.listdir(input_dir)
    for img in images:
        if 'IHC' in img:
            image = cv2.resize(cv2.imread(os.path.join(input_dir, img)), (512, 512))
            cv2.imwrite(os.path.join(test_dir, img), np.concatenate([image, image, image, image, image, image], 1))


if __name__ == '__main__':
    opt = ProcessingOptions().parse()   # get testing options
    prepare_dataset_for_testing(input_dir=opt.input_dir, dataset_dir=opt.output_dir)

