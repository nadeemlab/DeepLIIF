import subprocess
import os
import torch
import pytest

available_gpus = torch.cuda.device_count()


def test_cli_train(tmp_path, model_info):
    dirs_input = model_info['dir_input_train']
    for i in range(len(dirs_input)):
        dir_save = tmp_path
        
        fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --print-freq 1 --display-freq 1 --update-html-freq 1 --save-latest-freq 1 --save-epoch-freq 1',shell=True)
        assert res.returncode == 0

# 
# def test_cli_train_batchsize2(tmp_path, model_info):
#     dirs_input = model_info['dir_input_train']
#     for i in range(len(dirs_input)):
#         dir_save = tmp_path
#         
#         fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
#         num_input = len(fns_input)
#         assert num_input > 0
#     
#         res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 2 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1',shell=True)
#         assert res.returncode == 0


def test_cli_train_single_gpu(tmp_path, model_info):
    if torch.cuda.device_count() > 0:
        dirs_input = model_info['dir_input_train']
        for i in range(len(dirs_input)):
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0',shell=True)
            assert res.returncode == 0
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


# def test_cli_train_single_gpu_batchsize2(tmp_path, model_info):
#     if torch.cuda.device_count() > 0:
#         dirs_input = model_info['dir_input_train']
#         for i in range(len(dirs_input)):
#             dir_save = tmp_path
#             
#             fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
#             num_input = len(fns_input)
#             assert num_input > 0
#         
#             res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 2 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0',shell=True)
#             assert res.returncode == 0
#     else:
#         pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')
    

def test_cli_train_multi_gpu_dp(tmp_path, model_info):
    if torch.cuda.device_count() > 0:
        dirs_input = model_info['dir_input_train']
        for i in range(len(dirs_input)):
            dir_save = tmp_path
            
            fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 1 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --gpu-ids 1',shell=True)
            assert res.returncode == 0
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


# def test_cli_train_multi_gpu_batchsize2(tmp_path, model_info):
#     if torch.cuda.device_count() > 0:
#         dirs_input = model_info['dir_input_train']
#         for i in range(len(dirs_input)):
#             dir_save = tmp_path
#             
#             fns_input = [f for f in os.listdir(dirs_input[i] + '/train') if os.path.isfile(os.path.join(dirs_input[i] + '/train', f)) and f.endswith('png')]
#             num_input = len(fns_input)
#             assert num_input > 0
#             
#             res = subprocess.run(f'python cli.py train --model {model_info["model"]} --dataroot {dirs_input[i]} --name test_local --modalities-no {model_info["modalities_no"][i]} --seg-gen {model_info["seg_gen"][i]} --batch-size 2 --num-threads 0 --checkpoints-dir {dir_save} --remote True --n-epochs 1 --n-epochs-decay 1 --gpu-ids 0 --gpu-ids 1',shell=True)
#             assert res.returncode == 0
#     else:
#         pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')
