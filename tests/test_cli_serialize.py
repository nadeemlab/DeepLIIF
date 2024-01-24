import subprocess
from util import *
import torch

def test_cli_serialize(tmp_path, model_dir, model_info):
    torch.cuda.nvtx.range_push("test_cli_serialize")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_serialize {dir_model}")
        dir_output = tmp_path
    
        res = subprocess.run(f'python cli.py serialize --model-dir {dir_model} --output-dir {dir_output}',shell=True)
        assert res.returncode == 0
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
