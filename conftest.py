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
    
    
URLS_MODEL = {'latest':"https://zenodo.org/record/4751737/files/DeepLIIF_Latest_Model.zip"}

@pytest.fixture(scope="session")
def model_dir_final(model_type,model_dir):
        
    if model_dir:
        fns = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
        assert len(fns) > 0
        return model_dir
    else:
        assert model_type in URLS_MODEL.keys()
        url_model = URLS_MODEL[model_type]
        td = TemporaryDirectory()
        tmp_path = Path(td.name)
        
        target_path = tmp_path / url_model.split('/')[-1]
    
        urllib.request.urlretrieve(url_model, target_path)
        
        
        with zipfile.ZipFile(target_path, 'r') as f:
            f.extractall(tmp_path)
        
        print(os.listdir(tmp_path))
        return tmp_path