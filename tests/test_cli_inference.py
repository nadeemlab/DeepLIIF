import pytest
import subprocess
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import cv2
import numpy as np
import torch
import shutil

available_gpus = torch.cuda.device_count()
TOLERANCE = 0.0001

def calculate_ssim(dir_a,dir_b,fns,suffix_a,suffix_b,verbose_freq=50):
    print(suffix_a, suffix_b)
    score = 0
    count = 0
    
    for fn in fns:
        path_a = f"{dir_a}/{fn}{suffix_a}.png"
        assert os.path.exists(path_a), f'path {path_a} does not exist'
        img_a = cv2.imread(path_a)
        # print(img_a.shape)
        
        if suffix_b == 'combine': 
            paths_b = [f"{dir_b}/{fn}fake_BS_{i}.png" for i in range(1,5)]
            imgs_b = []
            for path_b in paths_b:
                assert os.path.exists(path_b), f'path {path_b} does not exist'
                imgs_b.append(cv2.imread(path_b))
            img_b = np.mean(imgs_b,axis=(0))
        else:
            path_b = f"{dir_b}/{fn}{suffix_b}.png"
            assert os.path.exists(path_b), f'path {path_b} does not exist'
            img_b = cv2.imread(path_b)
        # print(img_b.shape)
        
        score += ssim(img_a,img_b, data_range=img_a.max() - img_b.min(), multichannel=True, channel_axis=2)
        count +=1

        #if count % verbose_freq == 0 or count == len(fns):
        #    print(f"{count}/{len(fns)}, running mean SSIM {score / count}")
    return score/count


# https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
def remove_contents_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#### 1. test if functions can run ####
def test_cli_inference(tmp_path, model_dir, model_info):
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        dir_output = tmp_path
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output}',shell=True)
        assert res.returncode == 0
        
        fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output > 0
        
        remove_contents_in_folder(tmp_path)



def test_cli_inference_gpu_single(tmp_path, model_dir, model_info):
    if available_gpus > 0:
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        tile_size = model_info['tile_size']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
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
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


def test_cli_inference_gpu_multi(tmp_path, model_dir, model_info):
    if available_gpus > 1:
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            dir_output = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --gpu-ids 1 --gpu-ids 0',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')


def test_cli_inference_eager(tmp_path, model_dir, model_info):
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        dir_output = tmp_path
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --eager-mode',shell=True)
        assert res.returncode == 0
        
        fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
        num_output = len(fns_output)
        assert num_output > 0
        
        remove_contents_in_folder(tmp_path)


def test_cli_inference_eager_gpu_single(tmp_path, model_dir, model_info):
    if available_gpus > 0:
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            dir_output = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --eager-mode --gpu-ids 0',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
    else:
        pytest.skip(f'Detected {available_gpus} (< 1) available GPUs. Skip.')


def test_cli_inference_eager_gpu_multi(tmp_path, model_dir, model_info):
    if available_gpus > 1:
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            dir_output = tmp_path
            
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
            
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --eager-mode --gpu-ids 0 --gpu-ids 1',shell=True)
            assert res.returncode == 0
            
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')


from deepliif.models import inference
def test_cli_inference_bare(tmp_path, model_dir, model_info):
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        tile_size = 512
        overlap_size = 0
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        fn_input = fns_input[0] # take only 1 image
        
        img = Image.open(os.path.join(dir_input, fn_input))
        res = inference(img, tile_size, overlap_size, dir_model, use_torchserve=False, eager_mode=False,
                  color_dapi=False, color_marker=False, opt=None)
        


#### 2. test inference with selected gpus
def test_cli_inference_selected_gpu(tmp_path, model_dir, model_info):
    if available_gpus > 1:
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            dir_output = tmp_path
    
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --gpu-ids 1',shell=True)
            assert res.returncode == 0
    
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')


def test_cli_inference_eager_selected_gpu(tmp_path, model_dir, model_info):
    if available_gpus > 1:
        dirs_model = model_dir
        dirs_input = model_info['dir_input_inference']
        for dir_model, dir_input in zip(dirs_model, dirs_input):
            dir_output = tmp_path
    
            fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
            num_input = len(fns_input)
            assert num_input > 0
    
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --eager-mode --gpu-ids 1',shell=True)
            assert res.returncode == 0
    
            fns_output = [f for f in os.listdir(dir_output) if os.path.isfile(os.path.join(dir_output, f)) and f.endswith('png')]
            num_output = len(fns_output)
            assert num_output > 0
            
            remove_contents_in_folder(tmp_path)
    else:
        pytest.skip(f'Detected {available_gpus} (< 2) available GPUs. Skip.')


#### 3. test if inference results are consistent ####
def test_cli_inference_consistency(tmp_path, model_dir, model_info):
    """
    Seg Overlaid or Seg Refined are not compared
    """
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        print('serialized x 2',dir_model, dir_input)
        dirs_output = [tmp_path / 'test1', tmp_path / 'test2']
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0    
        
        for dir_output in dirs_output:
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output}',shell=True)
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


def test_cli_inference_eager_consistency(tmp_path, model_dir, model_info):
    """
    Seg Overlaid or Seg Refined are not compared
    """
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        print('eager x 2',dir_model, dir_input)
        dirs_output = [tmp_path / 'test1', tmp_path / f'test2']
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        
        for dir_output in dirs_output:
            res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --eager-mode',shell=True)
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


def test_cli_inference_eager_serialized_consistency(tmp_path, model_dir, model_info):
    """
    Seg Overlaid or Seg Refined are not compared
    """
    dirs_model = model_dir
    dirs_input = model_info['dir_input_inference']
    for dir_model, dir_input in zip(dirs_model, dirs_input):
        print('serialized vs eager',dir_model, dir_input)
        dirs_output = [tmp_path / 'test_eager', tmp_path / 'test_serialized']
        
        fns_input = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f)) and f.endswith('png')]
        num_input = len(fns_input)
        assert num_input > 0
        for dir_output in dirs_output:
            if 'eager' in str(dir_output):
                res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output} --eager-mode',shell=True)
            else:
                res = subprocess.run(f'python cli.py test --model-dir {dir_model} --input-dir {dir_input} --output-dir {dir_output}',shell=True)
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
      
