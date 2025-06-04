import os.path
from deepliif.data.base_dataset import BaseDataset, get_params, get_transform
from deepliif.data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, phase='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt.dataroot)
        self.preprocess = opt.preprocess
        self.dir_AB = os.path.join(opt.dataroot, phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(opt.load_size >= opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc
        self.no_flip = opt.no_flip
        self.modalities_no = opt.modalities_no
        self.seg_no = opt.seg_no
        self.input_no = opt.input_no
        self.seg_gen = opt.seg_gen
        self.load_size = opt.load_size
        self.crop_size = opt.crop_size
        self.model = opt.model

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
        if self.model in ['DeepLIIF','DeepLIIFKD']:
            num_img = self.modalities_no + 1 + 1 # +1 for segmentation channel, +1 for input image
        elif self.model == 'DeepLIIFExt':
            num_img = self.modalities_no * 2 + 1 if self.seg_gen else self.modalities_no + 1 # +1 for segmentation channel   
        elif self.model == 'SDG':
            num_img = self.modalities_no + self.seg_no + self.input_no
        else:
            raise Exception(f'model class {self.model} does not have corresponding implementation in deepliif/data/aligned_dataset.py')
        w2 = int(w / num_img)
        A = AB.crop((0, 0, w2, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.preprocess, self.load_size, self.crop_size, A.size)
        A_transform = get_transform(self.preprocess, self.load_size, self.crop_size, self.no_flip, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.preprocess, self.load_size, self.crop_size, self.no_flip, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B_Array = []
        if self.model in ['DeepLIIF','DeepLIIFKD']:
            for i in range(1, num_img):
                B = AB.crop((w2 * i, 0, w2 * (i + 1), h))
                B = B_transform(B)
                B_Array.append(B)

            return {'A': A, 'B': B_Array, 'A_paths': AB_path, 'B_paths': AB_path}
        elif self.model == 'DeepLIIFExt':
            for i in range(1, self.modalities_no + 1):
                B = AB.crop((w2 * i, 0, w2 * (i + 1), h))
                B = B_transform(B)
                B_Array.append(B)
            
            BS_Array = []
            if self.seg_gen:
                for i in range(self.modalities_no + 1, self.modalities_no * 2 + 1):
                    BS = AB.crop((w2 * i, 0, w2 * (i + 1), h))
                    BS = B_transform(BS)
                    BS_Array.append(BS)

            return {'A': A, 'B': B_Array, 'BS': BS_Array,'A_paths': AB_path, 'B_paths': AB_path}
        elif self.model == 'SDG':
            A_Array = []
            for i in range(self.input_no):
                A = AB.crop((w2 * i, 0, w2 * (i+1), h))
                A = A_transform(A)
                A_Array.append(A)
            for i in range(self.input_no, self.input_no + self.modalities_no + 1):
                B = AB.crop((w2 * i, 0, w2 * (i + 1), h))
                B = B_transform(B)
                B_Array.append(B)
            
            return {'A': A_Array, 'B': B_Array, 'A_paths': AB_path, 'B_paths': AB_path}
        else:
            raise Exception(f'model class {self.model} does not have corresponding implementation in deepliif/data/aligned_dataset.py')
        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
