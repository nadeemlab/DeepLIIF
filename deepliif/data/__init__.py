"""This package includes all the modules related to data loading and preprocessing

To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py'
and define a subclass 'DummyDataset' inherited from BaseDataset.
You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""

import importlib

import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from .base_dataset import __make_power_2, BaseDataset
from .aligned_dataset import AlignedDataset

from torch.utils.data.distributed import DistributedSampler
import os


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "deepliif.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches "
                                  "%s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    """
    return CustomDatasetDataLoader(opt)


class CustomDatasetDataLoader(object):
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.batch_size = opt.batch_size
        self.max_dataset_size = opt.max_dataset_size
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        sampler = None
        if os.getenv('LOCAL_RANK') is not None or os.getenv('RANK') is not None:
            sampler = DistributedSampler(self.dataset) if len(opt.gpu_ids) > 0 else None

        # control randomness: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        import numpy as np
        import random
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        if os.getenv('DEEPLIIF_SEED',None) is None:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                sampler=sampler,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches if sampler is None else False,
                num_workers=int(opt.num_threads)
            )
        else:
            g = torch.Generator()
            g.manual_seed(0)

            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                sampler=sampler,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches if sampler is None else False,
                num_workers=int(opt.num_threads),
                worker_init_fn=seed_worker,
                generator=g
            )

        self.sampler=sampler

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return max(len(self.dataset), self.max_dataset_size or 0)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if self.max_dataset_size and i * self.batch_size >= self.max_dataset_size:
                break
            yield data


def transform(img):
    return transforms.Compose([
        transforms.Lambda(lambda i: __make_power_2(i, base=4, method=Image.BICUBIC)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(img).unsqueeze(0)

