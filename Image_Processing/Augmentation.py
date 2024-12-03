import random

import numpy as np
import cv2
import scipy.ndimage as ndimage


class Augmentation:
    def __init__(self, images, tile_size=512):
        self.images = images
        self.shape = self.images[list(self.images.keys())[0]].shape
        self.rotation_angle = np.random.choice([0, 90, 180, 270], 1)[0]
        # self.zoom_value = random.randint(0, 5)
        self.alpha_affine = 0.1
        self.tile_size = tile_size

    def pipeline(self):
        """
        In this function, you can define a pipeline for augmentation
        and add the augmentation functions with the specified their order.
        :return:
        """
        self.elastic_transform()
        self.zoom()
        self.rotate()

    def zoom(self):
        """
        This function provides augmenting the pair of images by zooming into the image
        (keeping at least 75% of the original image).
        :return:
        """
        new_size = random.randint(int(self.shape[0] * 0.75), self.shape[0])
        assert self.shape[1] - new_size >= 0, f'self.shape[1] - new_size ({self.shape[1]} - {new_size})should not be negative'
        start_point = (random.randint(0, self.shape[0] - new_size), random.randint(0, self.shape[1] - new_size))
        for key in self.images.keys():
            try:
                self.images[key] = cv2.resize(self.images[key][start_point[0]: start_point[0] + new_size, start_point[1]: start_point[1] + new_size], (self.tile_size, self.tile_size))
            except Exception as e:
                print(e)

    def rotate(self):
        """
        This function randomly rotates the given image by selecting a random value form the given rotation angles.
        :param image: The input image.
        :param rotation_angle: Rotation angles specified by the user in the init function.
        :return:
        """
        for key in self.images.keys():
            try:
                self.images[key] = ndimage.rotate(self.images[key], self.rotation_angle, reshape=False)
            except Exception as e:
                print(e)

    def elastic_transform(self, random_state=None):
        """
        This function performs elastic deformation on the input image.

        Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape_size = self.shape[:2]
        self.alpha_affine = self.shape[1] * self.alpha_affine

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-self.alpha_affine, self.alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        for key in self.images.keys():
            try:
                self.images[key] = cv2.warpAffine(self.images[key], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
            except Exception as e:
                print(e)
