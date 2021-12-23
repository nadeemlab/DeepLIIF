import os.path
from deepliif.data.base_dataset import BaseDataset, get_params, get_transform
from deepliif.data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, load_size, crop_size, input_nc, output_nc, direction,  targets_no, preprocess, no_flip, phase, max_dataset_size):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, dataroot)
        self.preprocess = preprocess
        self.dir_AB = os.path.join(dataroot, phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, max_dataset_size))  # get image paths
        assert(load_size >= crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = output_nc if direction == 'BtoA' else input_nc
        self.output_nc = input_nc if direction == 'BtoA' else output_nc
        self.no_flip = no_flip
        self.targets_no = targets_no
        self.load_size = load_size
        self.crop_size = crop_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / (self.targets_no + 1))
        A = AB.crop((0, 0, w2, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.preprocess, self.load_size, self.crop_size, A.size)
        A_transform = get_transform(self.preprocess, self.load_size, self.crop_size, self.no_flip, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.preprocess, self.load_size, self.crop_size, self.no_flip, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B_Array = []
        for i in range(1, self.targets_no + 1):
            B = AB.crop((w2 * i, 0, w2 * (i + 1), h))
            B = B_transform(B)
            B_Array.append(B)

        return {'A': A, 'B': B_Array, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
