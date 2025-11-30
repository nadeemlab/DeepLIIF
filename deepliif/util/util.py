"""This module contains simple helper functions """
import os
from time import time
from functools import wraps

import torch
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim


def timeit(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        print(f'{f.__name__} {time() - ts}')

        return result

    return wrap


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    x, y, nc = image_numpy.shape
    
    if nc > 3:
        if nc % 3 == 0:
            nc_img = 3
            no_img = nc // nc_img
            
        elif nc % 2 == 0:
            nc_img = 2
            no_img = nc // nc_img
        else:
            nc_img = 1
            no_img = nc // nc_img
        print(f'image (numpy) has {nc}>3 channels, inferred to have {no_img} images each with {nc_img} channel(s)')
        l_image_numpy = np.dsplit(image_numpy,[nc_img*i for i in range(1,no_img)])
        image_numpy = np.concatenate(l_image_numpy, axis=1) # stack horizontally
        
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor_to_pil(t):
    return Image.fromarray(tensor2im(t))


def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min())


def check_multi_scale(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    max_ssim = (512, 0)
    for tile_size in range(100, 1000, 100):
        image_ssim = 0
        tile_no = 0
        for i in range(0, img2.shape[0], tile_size):
            for j in range(0, img2.shape[1], tile_size):
                if i + tile_size <= img2.shape[0] and j + tile_size <= img2.shape[1]:
                    tile = img2[i: i + tile_size, j: j + tile_size]
                    tile = cv2.resize(tile, (img1.shape[0], img1.shape[1]))
                    tile_ssim = calculate_ssim(img1, tile)
                    image_ssim += tile_ssim
                    tile_no += 1
        if tile_no > 0:
            image_ssim /= tile_no
            if max_ssim[1] < image_ssim:
                max_ssim = (tile_size, image_ssim)
    return max_ssim[0]


import subprocess
import os
from threading import Thread , Timer
import sched, time

# modified from https://stackoverflow.com/questions/67707828/how-to-get-every-seconds-gpu-usage-in-python
def get_gpu_memory(gpu_id=0):
    """
    Currently collects gpu memory info for a given gpu id.
    """
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(subprocess.check_output(COMMAND.split(),stderr=subprocess.STDOUT))[1:]
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    
    #assert len(memory_use_values)==1, f"get_gpu_memory::memory_use_values should have only 1 value, now has {len(memory_use_values)} (memory_use_values)"
    return memory_use_values[gpu_id]

class HardwareStatus():
    def __init__(self):
        self.gpu_mem = []
        self.timer = None
    
    def get_status_every_sec(self, gpu_id=0):
        """
            This function calls itself every 1 sec and appends the gpu_memory.
        """
        self.timer = Timer(1.0, self.get_status_every_sec)
        self.timer.start()
        self.gpu_mem.append(get_gpu_memory(gpu_id))
        # print('self.gpu_mem',self.gpu_mem)
    
    def stop_timer(self):
        self.timer.cancel()


def get_mod_id_seg(dir_model):
    # assume we already know there are seg models - this check is intended to be done prior to calling this function
    fns = [fn for fn in os.listdir(dir_model) if fn.endswith('.pth') and 'net_G' in fn]
    
    if len(fns) == 0: # typically this means the directory only contains serialized models
        fns = [fn for fn in os.listdir(dir_model) if fn.endswith('.pt') and fn.startswith('G')]
        model_names = [fn[1:-3] for fn in fns] # 1[:-3] drop ".pt" and the starting G
    else:
        model_names = [fn[:-4].split('_')[2][1:] for fn in fns] # [1:] drop "G"
    
    if len(fns) == 0:
        raise Exception('Cannot find any model file ending with .pt or .pth in directory',dir_model)
    
    model_name_seg = max(model_names, key=len)
    return model_name_seg[0]

def get_input_id(dir_model):
    # assume we already know there are seg models - this check is intended to be done prior to calling this function
    fns = [fn for fn in os.listdir(dir_model) if fn.endswith('.pth') and 'net_G' in fn]
    
    if len(fns) == 0: # typically this means the directory only contains serialized models
        fns = [fn for fn in os.listdir(dir_model) if fn.endswith('.pt') and fn.startswith('G')]
        model_names_seg = [fn[2:-3] for fn in fns] # [2:] drop "GS"/"G5"
    else:
        model_names_seg = [fn[:-4].split('_')[2][2:] for fn in fns] # [2:] drop "GS"/"G5"
    
    if len(fns) == 0:
        raise Exception('Cannot find any model file ending with .pt or .pth in directory',dir_model)
      
    if '0' in model_names_seg:
        return '0'
    else:
        return '1'
    
def init_input_and_mod_id(opt):
    """
    Used by model classes to initialize input id and mod id under different situations.
    """
    mod_id_seg = None
    input_id = None
    
    if not opt.continue_train and opt.is_train:
        if hasattr(opt,'mod_id_seg'): # use mod id seg from train opt file if available
            mod_id_seg = opt.mod_id_seg
        elif not hasattr(opt,'modalities_names'): # backward compatible with models trained before this param was introduced
            mod_id_seg = opt.modalities_no + 1 # for original DeepLIIF, modalities_no is 4 and the seg mod id is 5
        else:
            mod_id_seg = 'S'
        
        if opt.model in ['DeepLIIF','DeepLIIFKD']:
            input_id = '0'
    else: # for both continue train and test, we load existing models, so need to obtain seg mod id and input id from filenames
        if hasattr(opt, 'mod_id_seg'):
            mod_id_seg = opt.mod_id_seg
        else:
            # for contiue-training, extract mod id seg from existing files if not available
            mod_id_seg = get_mod_id_seg(os.path.join(opt.checkpoints_dir, opt.name))
        
        if opt.model in ['DeepLIIF','DeepLIIFKD']:
            input_id = get_input_id(os.path.join(opt.checkpoints_dir, opt.name))
        
    return mod_id_seg, input_id
