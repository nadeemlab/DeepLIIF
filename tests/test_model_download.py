import torch
import zipfile
import os
import pytest



def test_model_download(model_dir_final):
    print('final model dir:',model_dir_final)
    print(os.listdir(model_dir_final))