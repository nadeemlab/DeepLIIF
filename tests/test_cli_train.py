# Tests for training
# hardware config: cpu / single gpu / multi gpu
# tests (cpu, multi gpu):
#   - basic training
# tests (sinlge gpu):
#   - basic training
#   - optimizer
#   - net-g
#   - net-gs
#   - with-val

import subprocess
import os
import torch
import pytest

available_gpus = torch.cuda.device_count()

#----------------------
#---- cpu-based -------
#----------------------
def test_cli_train(tmp_path, model_info):
    torch.cuda.nvtx.range_push("test_cli_train")
    dirs_input = model_info['dir_input_train']
    for i in range(len(dirs_input)):
        torch.cuda.nvtx.range_push(f"test_cli_train {dirs_input[i]}")
        dir_save = tmp_path
        
        fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --print-freq 1 --display-freq 1 --update-html-freq 1 --save-latest-freq 1 --save-epoch-freq 1',shell=True)
        assert res.returncode == 0
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()



#----------------------
#---- gpu: single -----
#----------------------
def test_cli_train_single_gpu(tmp_path, model_info):
    if available_gpus > 0:
        torch.cuda.nvtx.range_push("test_cli_train_single_gpu")
        dirs_input = model_info['dir_input_train']
        for i in range(len(dirs_input)):
            torch.cuda.nvtx.range_push(f"test_cli_train_single_gpu {dirs_input[i]}")
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0',shell=True)
            assert res.returncode == 0
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


def test_cli_train_single_gpu_optimizer(tmp_path, model_info):
    if available_gpus > 0:
        torch.cuda.nvtx.range_push("test_cli_train_single_gpu_optimizer")
        dirs_input = model_info['dir_input_train']
        for i in range(len(dirs_input)):
            torch.cuda.nvtx.range_push(f"test_cli_train_single_gpu {dirs_input[i]}")
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --optimizer sgd',shell=True)
            assert res.returncode == 0
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


def test_cli_train_single_gpu_netg(tmp_path, model_info):
    if available_gpus > 0:
        torch.cuda.nvtx.range_push("test_cli_train_single_gpu_netg")
        dirs_input = model_info['dir_input_train']
        for i in range(len(dirs_input)):
            torch.cuda.nvtx.range_push(f"test_cli_train_single_gpu {dirs_input[i]}")
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --net-g unet_512',shell=True)
            assert res.returncode == 0
            
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --net-g unet_512_attention',shell=True)
            assert res.returncode == 0
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')
        

def test_cli_train_single_gpu_netgs(tmp_path, model_info):
    if available_gpus > 0:
        torch.cuda.nvtx.range_push("test_cli_train_single_gpu_netgs")
        dirs_input = model_info['dir_input_train']
        for i in range(len(dirs_input)):
            torch.cuda.nvtx.range_push(f"test_cli_train_single_gpu {dirs_input[i]}")
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --net-gs unet_512',shell=True)
            assert res.returncode == 0
            
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --net-gs unet_512_attention',shell=True)
            assert res.returncode == 0
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


def test_cli_train_single_gpu_withval(tmp_path, model_info):
    if available_gpus > 0:
        torch.cuda.nvtx.range_push("test_cli_train_single_gpu_withval")
        dirs_input = model_info['dir_input_train']
        for i in range(len(dirs_input)):
            torch.cuda.nvtx.range_push(f"test_cli_train_single_gpu {dirs_input[i]}")
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --with-val',shell=True)
            assert res.returncode == 0
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')

#----------------------
#---- gpu: multi ------
#----------------------
def test_cli_train_multi_gpu_dp(tmp_path, model_info):
    if available_gpus > 1:
        torch.cuda.nvtx.range_push("test_cli_train_multi_gpu_dp")
        dirs_input = model_info['dir_input_train']
        for i in range(len(dirs_input)):
            torch.cuda.nvtx.range_push(f"test_cli_train_multi_gpu_dp {dirs_input[i]}")
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --gpu-ids 1',shell=True)
            assert res.returncode == 0
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')

