from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request
import os
import urllib.request
import zipfile
import subprocess
import datetime

MODEL_INFO = {'latest':{'model':'DeepLIIF', # cli.py train looks for subfolder "train" under dataroot
                        'dir_input_train':['Datasets/Sample_Dataset'],
                        'dir_input_inference':['Datasets/Sample_Dataset/test_cli'],
                        'dir_model':['../checkpoints/model/DeepLIIF_Latest_Model'],
                        'modalities_no': [4],
                        'seg_gen':[True],
                        'tile_size':512},
              'ext':{'model':'DeepLIIFExt',
                     'dir_input_train':['Datasets/Sample_Dataset_ext_withseg','Datasets/Sample_Dataset_ext_noseg'],
                     'dir_input_inference':['Datasets/Sample_Dataset_ext_withseg/test_cli','Datasets/Sample_Dataset_ext_noseg/test_cli'],
                     'dir_model':['../checkpoints/deepliif_extension_LN_Tonsil_4mod_400epochs','../checkpoints/HER2_5mod_400epochs'],
                     'modalities_no':[4,5],
                     'seg_gen':[True,False],
                     'tile_size':1024},
              'sdg':{'model':'SDG',
                     'dir_input_train':['Datasets/Sample_Dataset_sdg'],
                     'dir_input_inference':['Datasets/Sample_Dataset/test_cli'],
                     'dir_model':['../checkpoints/sdg_20240104'],
                     'modalities_no': [4],
                     'seg_gen':[False],
                     'tile_size':512}}

def pytest_addoption(parser):
    parser.addoption("--model_type", action="store", default="latest")
    #parser.addoption("--model_dir", action="store", default=None)

import pytest

@pytest.fixture(scope="session")
def model_type(pytestconfig):
    return pytestconfig.getoption("model_type")

@pytest.fixture(scope="session")
def model_info(model_type):
    return MODEL_INFO[model_type]

@pytest.fixture(scope="session")
def model_dir(model_info):
    return model_info['dir_model']
