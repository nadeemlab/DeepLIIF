import subprocess
from util import *

def test_cli_serialize(tmp_path, model_dir, model_info):
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        dir_output = tmp_path
    
        res = subprocess.run(f'python cli.py serialize --models-dir {dir_model} --output-dir {dir_output}',shell=True)
        assert res.returncode == 0
        
        remove_contents_in_folder(tmp_path)
