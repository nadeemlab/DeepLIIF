import pytest
import subprocess
import os
import torch
from util import *

available_gpus = torch.cuda.device_count()
TOLERANCE = 0.0003
TOLERANCE_SEG = 0.05

subdir_testpy = 'test_latest/images'

def match_suffix(l_suffix_cli, model='DeepLIIF'):
    """
    Given a list of suffix found in cli output folder, derive a list of suffix used by testpy output
    """
    assert model in ['DeepLIIF', 'DeepLIIFExt', 'SDG'], f'Cannot derive suffix of output images for model {model}'
    
    if model == 'DeepLIIF':
        d_cli2testpy = {'Hema':'fake_B_1', 'DAPI':'fake_B_2', 'Lap2':'fake_B_3',
                        'Marker':'fake_B_4', 'Seg':'fake_B_5'}
    elif model in ['DeepLIIFExt', 'SDG']:
        d_cli2testpy = {'mod':'fake_B_', 'Seg':'fake_BS_'}
    
    # only include cli suffix that matches the specified mapping
    res_cli = []
    res_testpy = []
    for suffix_cli in l_suffix_cli:
        if model == 'DeepLIIF':
            try: 
                res_testpy.append(d_cli2testpy[suffix_cli])
                res_cli.append(suffix_cli)
            except:
                pass
        else:
            try:
                img_type = suffix_cli[:3]
                mod_idx = suffix_cli[3:]
                res_testpy.append(d_cli2testpy[img_type] + mod_idx)
                res_cli.append(suffix_cli)
            except:
                pass

    return res_cli, res_testpy

#### 0. test if test.py can run ####
def test_testpy(tmp_path, model_dir, model_info):
    torch.cuda.nvtx.range_push("test_testpy")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_testpy']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_testpy {dir_model}")
        dir_output = tmp_path
        
        fns_input = [f for f in os.listdir(dir_input+'/test') if os.path.isfile(os.path.join(dir_input+'/test', f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python test.py --checkpoints_dir {dir_model} --dataroot {dir_input} --results_dir {dir_output}',shell=True)
        assert res.returncode == 0
        
        dir_output_img = dir_output / subdir_testpy
        fns_output = [f for f in os.listdir(dir_output_img) if os.path.isfile(os.path.join(dir_output_img, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output > 0
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()

#### 1. test if functions can run ####
def test_cli_inference(tmp_path, model_dir, model_info):
    torch.cuda.nvtx.range_push("test_cli_inference")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    tile_size = model_info['tile_size']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_inference {dir_model}")
        dir_output = tmp_path
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size}',shell=True)
        assert res.returncode == 0
        
        fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output > 0
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()


def test_cli_inference_cpu(tmp_path, model_dir, model_info):
    torch.cuda.nvtx.range_push("test_cli_inference_cpu")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    tile_size = model_info['tile_size']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_inference {dir_model}")
        dir_output = tmp_path
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --gpu-ids -1',shell=True)
        assert res.returncode == 0
        
        fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output > 0
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()



def test_cli_inference_gpu_single(tmp_path, model_dir, model_info):
    if available_gpus > 0:
        torch.cuda.nvtx.range_push("test_cli_inference_gpu_single")
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        tile_size = model_info['tile_size']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            torch.cuda.nvtx.range_push(f"test_cli_inference_gpu_single {dir_model}")
            dir_output = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --gpu-ids 0',shell=True)
            assert res.returncode == 0
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


def test_cli_inference_gpu_multi(tmp_path, model_dir, model_info):
    if available_gpus > 1:
        torch.cuda.nvtx.range_push("test_cli_inference_gpu_multi")
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        tile_size = model_info['tile_size']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            torch.cuda.nvtx.range_push(f"test_cli_inference_gpu_multi {dir_model}")
            dir_output = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --gpu-ids 1 --gpu-ids 0',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')


def test_cli_inference_eager(tmp_path, model_dir, model_info):
    torch.cuda.nvtx.range_push("test_cli_inference_eager")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    tile_size = model_info['tile_size']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_inference_eager {dir_model}")
        dir_output = tmp_path
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --eager-mode',shell=True)
        assert res.returncode == 0
        
        fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output > 0
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()


def test_cli_inference_eager_cpu(tmp_path, model_dir, model_info):
    torch.cuda.nvtx.range_push("test_cli_inference_eager_cpu")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    tile_size = model_info['tile_size']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_inference_eager {dir_model}")
        dir_output = tmp_path
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --eager-mode --gpu-ids -1',shell=True)
        assert res.returncode == 0
        
        fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output > 0
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()


def test_cli_inference_eager_gpu_single(tmp_path, model_dir, model_info):
    if available_gpus > 0:
        torch.cuda.nvtx.range_push("test_cli_inference_eager_gpu_single")
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        tile_size = model_info['tile_size']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            torch.cuda.nvtx.range_push(f"test_cli_inference_eager_gpu_single {dir_model}")
            dir_output = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --eager-mode --gpu-ids 0',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


def test_cli_inference_eager_gpu_multi(tmp_path, model_dir, model_info):
    if available_gpus > 1:
        torch.cuda.nvtx.range_push("test_cli_inference_eager_gpu_multi")
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        tile_size = model_info['tile_size']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            torch.cuda.nvtx.range_push(f"test_cli_inference_eager_gpu_multi {dir_model}")
            dir_output = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --eager-mode --gpu-ids 0 --gpu-ids 1',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')


from deepliif.models import inference
def test_cli_inference_bare(tmp_path, model_dir, model_info):
    torch.cuda.nvtx.range_push("test_cli_inference_bare")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    tile_size = model_info['tile_size']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_inference_bare {dir_model}")
        overlap_size = 0
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        fn_input = fns_input[0] # take only 1 image
        
        img = Image.open(os.path.join(dir_input, fn_input))
        res = inference(img, tile_size, overlap_size, dir_model, use_torchserve=False, eager_mode=False,
                  color_dapi=False, color_marker=False, opt=None)
        
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()


#### 2. test inference with selected gpus
def test_cli_inference_selected_gpu(tmp_path, model_dir, model_info):
    if available_gpus > 1:
        torch.cuda.nvtx.range_push("test_cli_inference_selected_gpu")
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        tile_size = model_info['tile_size']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            torch.cuda.nvtx.range_push(f"test_cli_inference_selected_gpu {dir_model}")
            dir_output = tmp_path
    
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --gpu-ids 1',shell=True)
            assert res.returncode == 0
    
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')


def test_cli_inference_eager_selected_gpu(tmp_path, model_dir, model_info):
    if available_gpus > 1:
        torch.cuda.nvtx.range_push("test_cli_inference_eager_selected_gpu")
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        tile_size = model_info['tile_size']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            torch.cuda.nvtx.range_push(f"test_cli_inference_eager_selected_gpu {dir_model}")
            dir_output = tmp_path
    
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --eager-mode --gpu-ids 1',shell=True)
            assert res.returncode == 0
    
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')


#### 3. test if inference results are consistent ####
def test_cli_inference_consistency(tmp_path, model_dir, model_info):
    """
    Seg Overlaid or Seg Refined are not compared
    """
    torch.cuda.nvtx.range_push("test_cli_inference_consistency")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    tile_size = model_info['tile_size']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_inference_consistency {dir_model}")
        print('serialized x 2',dir_model, dir_input)
        dirs_output = [tmp_path / 'test1', tmp_path / 'test2']
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0    
        
        for dir_output in dirs_output:
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size}',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
        
        fns = [f for f in os.listdir(dirs_output[0]) if os.path.isfile(os.path.join(dirs_output[0], f)) and f.endswith('png')]
        fns = [x.replace('_Overlaid','').replace('_Refined','') for x in fns] # remove _Overlaid / _Refined
        l_suffix = list(set([fn[:-4].split('_')[-1] for fn in fns]))
        print('suffix:',l_suffix)       
    
        fns = list(set(['_'.join(x[:-4].split('_')[:-1]) for x in fns])) # remove suffix (e.g., fake_B_1.png), then take unique values
        print('num of input files (derived from output):',len(fns))
        print('input img name:',fns)
        
        print(f'Calculating SSIM...')
        for i, suffix in enumerate(l_suffix):
            ssim_score = calculate_ssim(dirs_output[0], dirs_output[1], fns, '_'+suffix, '_'+suffix)
            print(suffix, ssim_score)
            assert (1 - ssim_score) < TOLERANCE
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
        
    torch.cuda.nvtx.range_pop()


def test_cli_inference_eager_consistency(tmp_path, model_dir, model_info):
    """
    Seg Overlaid or Seg Refined are not compared
    """
    torch.cuda.nvtx.range_push("test_cli_inference_eager_consistency")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    tile_size = model_info['tile_size']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_inference_eager_consistency {dir_model}")
        print('eager x 2',dir_model, dir_input)
        dirs_output = [tmp_path / 'test1', tmp_path / f'test2']
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        for dir_output in dirs_output:
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --eager-mode',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
        
        fns = [f for f in os.listdir(dirs_output[0]) if os.path.isfile(os.path.join(dirs_output[0], f)) and f.endswith('png')]
        fns = [x.replace('_Overlaid','').replace('_Refined','') for x in fns] # remove _Overlaid / _Refined
        l_suffix = list(set([fn[:-4].split('_')[-1] for fn in fns]))
        print('suffix:',l_suffix)       
    
        fns = list(set(['_'.join(x[:-4].split('_')[:-1]) for x in fns])) # remove suffix (e.g., fake_B_1.png), then take unique values
        print('num of input files (derived from output):',len(fns))
        print('input img name:',fns)
        
        print(f'Calculating SSIM...')
        for i, suffix in enumerate(l_suffix):
            ssim_score = calculate_ssim(dirs_output[0], dirs_output[1], fns, '_'+suffix, '_'+suffix)
            print(suffix, ssim_score)
            assert (1 - ssim_score) < TOLERANCE
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()


def test_cli_inference_eager_serialized_consistency(tmp_path, model_dir, model_info):
    """
    Seg Overlaid or Seg Refined are not compared
    """
    torch.cuda.nvtx.range_push("test_cli_inference_eager_serialized_consistency")
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    tile_size = model_info['tile_size']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        torch.cuda.nvtx.range_push(f"test_cli_inference_eager_serialized_consistency {dir_model}")
        print('serialized vs eager',dir_model, dir_input)
        dirs_output = [tmp_path / 'test_eager', tmp_path / 'test_serialized']
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        for dir_output in dirs_output:
            if 'eager' in str(dir_output):
                res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size} --eager-mode',shell=True)
            else:
                res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --tile-size {tile_size}',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
        
        fns = [f for f in os.listdir(dirs_output[0]) if os.path.isfile(os.path.join(dirs_output[0], f)) and f.endswith('png')]
        fns = [x.replace('_Overlaid','').replace('_Refined','') for x in fns] # remove _Overlaid / _Refined
        l_suffix = list(set([fn[:-4].split('_')[-1] for fn in fns]))
        print('suffix:',l_suffix)       
    
        fns = list(set(['_'.join(x[:-4].split('_')[:-1]) for x in fns])) # remove suffix (e.g., fake_B_1.png), then take unique values
        print('num of input files (derived from output):',len(fns))
        print('input img name:',fns)
        
        print(f'Calculating SSIM...')
        for i, suffix in enumerate(l_suffix):
            ssim_score = calculate_ssim(dirs_output[0], dirs_output[1], fns, '_'+suffix, '_'+suffix)
            print(suffix, ssim_score)
            assert (1 - ssim_score) < TOLERANCE
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()


def test_cli_inference_eager_testpy_consistency(tmp_path, model_dir, model_info):
    torch.cuda.nvtx.range_push("test_cli_inference_eager_testpy_consistency")
    dirs_model = model_dir
    dirs_input_cli = model_info['dir_input_inference']
    dirs_input_testpy = model_info['dir_input_testpy']
    tile_size = model_info['tile_size']
    for dir_model, dir_input_cli, dir_input_testpy in zip(dirs_model, dirs_input_cli, dirs_input_testpy):
        torch.cuda.nvtx.range_push(f"test_cli_inference_eager_testpy_consistency {dir_model}")
        print('eager vs testpy',dir_model, dir_input_cli, dir_input_testpy)
        dirs_output = [tmp_path / 'test_eager', tmp_path / 'test_testpy']
        
        fns_input_cli = [f for f in os.listdir(dir_input_cli) if os.path.isfile(os.path.join(dir_input_cli, f)) and f.endswith('png')]
        num_input_cli = len(fns_input_cli)
        assert num_input_cli > 0
        
        fns_input_testpy = [f for f in os.listdir(dir_input_testpy+'/test') if os.path.isfile(os.path.join(dir_input_testpy+'/test', f)) and f.endswith('png')]
        num_input_testpy = len(fns_input_testpy)
        assert num_input_testpy > 0
        
        for dir_output in dirs_output:
            if dir_output.name == 'test_eager': # check subfolder name; dir_output is posixpath
                res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input_cli} --output-dir {dir_output} --tile-size {tile_size} --eager-mode',shell=True)
                assert res.returncode == 0
                
                fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
                num_output = len(fns_output)
                assert num_output > 0
            else:
                res = subprocess.run(f'python test.py --checkpoints_dir {dir_model} --dataroot {dir_input_testpy} --results_dir {dir_output}',shell=True)
                assert res.returncode == 0
                
                dir_output_img = dir_output / subdir_testpy
                fns_output = [f for f in os.listdir(dir_output_img) if os.path.isfile(os.path.join(dir_output_img, f)) and f.endswith('png')]
                num_output = len(fns_output)
                assert num_output > 0
        
        # filenames from cli inference
        fns = [f for f in os.listdir(dirs_output[0]) if os.path.isfile(os.path.join(dirs_output[0], f)) and f.endswith('png')]
        fns = [x.replace('_Overlaid','').replace('_Refined','') for x in fns] # remove _Overlaid / _Refined
        
        l_suffix_cli = list(set([fn[:-4].split('_')[-1] for fn in fns]))
        l_suffix_cli, l_suffix_testpy = match_suffix(l_suffix_cli, model=model_info['model'])
        print('suffix:',l_suffix_cli)       
    
        fns = list(set(['_'.join(x[:-4].split('_')[:-1]) for x in fns])) # remove suffix (e.g., fake_B_1.png), then take unique values
        print('num of input files (derived from output):',len(fns))
        print('input img name:',fns)
        
        print(f'Calculating SSIM...')
        for i, (suffix_cli, suffix_testpy) in enumerate(zip(l_suffix_cli, l_suffix_testpy)):
            ssim_score = calculate_ssim(dirs_output[0], dirs_output[1]/subdir_testpy, fns, '_'+suffix_cli, '_'+suffix_testpy)
            print(suffix_cli, ssim_score)
            if 'seg' in suffix_cli.lower():
                assert (1 - ssim_score) < TOLERANCE_SEG
            else:
                assert (1 - ssim_score) < TOLERANCE
        
        remove_contents_in_folder(tmp_path)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
