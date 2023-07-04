from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request
import os
import urllib.request
import zipfile

def pytest_addoption(parser):
    parser.addoption("--model_type", action="store", default="latest")
    parser.addoption("--model_dir", action="store", default=None)

import pytest

@pytest.fixture(scope="session")
def model_type(pytestconfig):
    return pytestconfig.getoption("model_type")

@pytest.fixture(scope="session")
def model_dir(pytestconfig):
    return pytestconfig.getoption("model_dir")


MODEL_INFO = {'latest':{'model':'DeepLIIF', # cli.py train looks for subfolder "train" under dataroot
                        'dir_input_train':'Datasets/Sample_Dataset',
                        'dir_input_inference':'Datasets/Sample_Dataset/test_cli'},
              'ext':{'model':'DeepLIIFExt',
                     'dir_input_train':'Datasets/Sample_Dataset_ext',
                     'dir_input_inference':'Datasets/Sample_Dataset_ext/test_cli'}}

@pytest.fixture(scope="session")
def model_info(model_type):
    return MODEL_INFO[model_type]