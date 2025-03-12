import os.path
from deepliif.data.base_dataset import BaseDataset, get_params, get_transform
from deepliif.data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, phase='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc
        self.preprocess = opt.preprocess
        self.no_flip = opt.no_flip
        self.modalities_no = opt.modalities_no
        self.seg_no = opt.seg_no
        self.input_no = opt.input_no
        self.seg_gen = opt.seg_gen
        self.load_size = opt.load_size
        self.crop_size = opt.crop_size
        self.model = opt.model
        
        self.dir_A = os.path.join(opt.dataroot, phase + 'A')  # create a path '/path/to/data/trainA'
        # trainB1/trainB2/trainB3... are organized as elements of DATASET B which is a list 
        self.dirs_B = [os.path.join(opt.dataroot, phase + f'B{i}') for i in range(1,self.modalities_no+1)]  # create a list of paths ['/path/to/data/trainB',...]

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = [sorted(make_dataset(dir_B, opt.max_dataset_size)) for dir_B in self.dirs_B]    # load images from '/path/to/data/trainB', '/path/to/data/trainC', ...
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_sizes = [len(B_paths) for B_paths in self.B_paths]  # get the size of dataset B1, B2, B3, ...
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            indice_B = [index % B_size for B_size in self.B_sizes]
        else:   # randomize the index for domain B to avoid fixed pairs.
            indice_B = [random.randint(0, B_size - 1) for B_size in self.B_sizes]
        B_paths = [B_paths[index_B] for B_paths, index_B in zip(self.B_paths, indice_B)]
        
        A_img = Image.open(A_path).convert('RGB')
        B_imgs = [Image.open(B_path).convert('RGB') for B_path in B_paths]
        
        # apply image transformation
        transform_params = get_params(self.preprocess, self.load_size, self.crop_size, A_img.size)
        A_transform = get_transform(self.preprocess, self.load_size, self.crop_size, self.no_flip, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.preprocess, self.load_size, self.crop_size, self.no_flip, transform_params, grayscale=(self.output_nc == 1))
        A = A_transform(A_img)
        Bs = [B_transform(B_img) for B_img in B_imgs]
        
        return {'A': A, 'Bs': Bs, 'A_paths': A_path, 'B_paths': B_paths}

    def __len__(self):
        """Return the total number of images in the dataset.

        
        The effective size of this dataset will be the size of datasetA through which we loop and grab a random/matching image B1/B2/B3... for
        """
        return self.A_size #max(self.A_size, self.B_size)
