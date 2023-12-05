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
                        'dir_model':['/mnts/deepliif/deepliif-2022/model/DeepLIIF_Latest_Model'],
                        'modalities_no': [4],
                        'seg_gen':[True],
                        'tile_size':512},
              'ext':{'model':'DeepLIIFExt',
                     'dir_input_train':['Datasets/Sample_Dataset_ext_withseg','Datasets/Sample_Dataset_ext_noseg'],
                     'dir_input_inference':['Datasets/Sample_Dataset_ext_withseg/test_cli','Datasets/Sample_Dataset_ext_noseg/test_cli'],
                     'dir_model':['/mnts/deepliif/deepliif-2022/model/deepliif_extension_LN_Tonsil_4mod_400epochs','/mnts/deepliif/deepliif-2022/model/HER2_5mod_400epochs'],
                     'modalities_no':[4,5],
                     'seg_gen':[True,False],
                     'tile_size':1024}}

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

#@pytest.fixture(scope="session")
#def model_dir(model_info, tmpdir_factory):
#    dirs_input = model_info['dir_input_train']
#    dirs_model = []
#    for dir_input in dirs_input:
#        print(f'Creating models on data {dir_input}...')
#        prefix = datetime.datetime.now().strftime('%Y%m%d%H%M')
#        dir_save = tmpdir_factory.mktemp(f'{prefix}_model') / f"{dir_input.split('/')[-1]}"
#        
#        fns_input = [f for f in os.listdir(dir_input + '/train') if os.path.isfile(os.path.join(dir_input + '/train', f)) and f.endswith('png')]
#        num_input = len(fns_input)
#        assert num_input > 0
#        
#        res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dir_input} --name . --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --seg-gen {"noseg" not in dir_input}',shell=True)
#        assert res.returncode == 0
#        
#        res = subprocess.run(f'python cli.py serialize --models-dir {dir_save} --output-dir {dir_save}',shell=True)
#        assert res.returncode == 0
#        
#        dirs_model.append(dir_save)
#    return model_info['dir_model']#dirs_model ##


@pytest.fixture(scope="session")
def model_dir(model_info):
    return model_info['dir_model']

    
    
URLS_CODE_VERSION = {'latest':"https://zenodo.org/record/4751737/files/DeepLIIF_Latest_Model.zip"}
