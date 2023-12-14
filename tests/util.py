from skimage.metrics import structural_similarity as ssim
from PIL import Image
import cv2
import numpy as np
import shutil
import os

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
