import sys
import os

# add the DeepLIIF root directory to Python path in order to call the image processing script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import json
import cv2
import argparse
import copy
import random
from PIL import Image, ImageOps

DEFAULT_OPTIONS = {'concateToWideFormat':
                      {'input_img':
                          {'input_mod':[['_ihc.png','{input_dir}']],
                           'trans_mod':[['_ihc_Marker.png','{input_dir}']],
                           'seg_mod':[['_seg.png','{input_dir}']]},
                       'output_dir':'{output_dir}',
                       'process_mod':{}, # {<index in ALL mod>:<process type>}; currently support either cleanStains or black
                       }, 
                   'trainTestSplit':
                      {'train_ratio':0.8,
                       'output_dir':'{output_dir}_traintest{train_ratio}',
                       'img_names_test':['H25-007118-2A-2_ss12460_192056__01_1408_0',
                                         'S25-059017-1A-9_ss12022_110221__04_1408_0',
                                         'S25-059017-1A-9_ss12022_110221__05_1408_492',
                                         'S25-059017-1A-9_ss12022_110221__05_470_0',
                                         'S25-059017-1A-9_ss12022_110221__05_470_492',
                                         'S25-059017-1A-9_ss12022_110221__05_940_0',
                                         'S25-059017-1A-9_ss12022_110221__06_940_492',
                                         'S25-059017-1A-9_ss12022_110221__07_470_0',
                                         'S25-059017-1A-9_ss12022_110221.svs_67539_52871_0_0',
                                         'S25-059017-1A-9_ss12022_110221.svs_70632_54451_512_512',
                                         'S25-059017-1A-9_ss12022_110221.svs_77617_38373_0_0',
                                         'S25-059017-1A-9_ss12022_110221.svs_77617_38373_512_512',
                                         # 2025-11-26
                                         'C25-36104-1B-19_SS12017_221646.svs_107712_33574_0_0',
                                         'H25-006735-1A-16_ss12460_121724.svs_154023_18488_0_0',
                                         'H25-009692-1A-38_SS12257_081703.svs_17884_17594_0_0',
                                         'H25-009720-2A-19_ss12014_094308.svs_83004_32682_0_0',
                                         'S25-083086-1A-12_ss12250_155859.svs_92498_21089_0_0',
                                         'S25-091121-1A-12_ss12460_085450.svs_113014_32242_0_0',
                                         'S25-091707-1F-6_ss12013_002257.svs_144710_35652_0_0'], # if this list is provided, random split with train ratio will be ignored
                       'process_mod':{"1":'cleanStains'}, # {<index in ALL mod>:<process type>}; currently support either cleanStains or black
                       'tile_size':(512,512),
                       'create_gt':True, # whether to take the last tile and put into subfolder val_cli_gt
                       'input_no':1, # note that input_no > 1 will disable create_gt as this indicates DeepLIIFExt model for which we do not use val_cli_gt
                       }, 
                   'augmentTrainSet':
                      {'output_dir':'{output_dir}_aug', # if trainTestSplit is part of the pipeline, then augmenTrainSet writes the aug img to the same train subfolder instead of a new folder {output_dir}_aug
                       'aug_no':9,
                       'modality_types':['IHC','Marker','Seg'],
                       },
                   'calculateValMetric': # this step only applies to DeepLIIF and DeepLIIFEXt models at the moment
                      {'model':'DeepLIIF',
                       'tile_size':512},
                   'addExistingData':
                      {'output_dir':'{output_dir}_plus',
                       'dir_add':['/mnts/deepliif/deepliif-2022/All_Gen_BC_Lung_Bladder_Dataset']}
                                                          }

class prepareDataPipeline():
    def __init__(self, input_dir, output_dir, steps, seg_gen, options):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.steps = [step for step in steps.split(',')]
        #self.opt = options
        self.seg_gen = seg_gen
            
        self.opt = {k:v for k,v in DEFAULT_OPTIONS.items() if k in self.steps}
        self.opt = modify_opt(self.opt, options)
        if self.seg_gen == False:
            if 'concateToWideFormat' in steps:
                del opt['concateToWideFormat']['input_img']['seg_mod']
    
    def run(self):
        for i,step in enumerate(self.steps):
            print(f'Starting step {i+1}/{len(self.steps)}: {step}')
            #continue
            print(self.opt[step])
            getattr(self,step)(self.opt[step])
    
    def _get_tile_size(self,path_img):
        img = Image.open(path_img)
        self.tile_size = img.size
        print('Image shape (w x h):', img.size)
        return img.size
        
    def concateToWideFormat(self, opt):
        # process input config list
        l_input_config = opt['input_img']['input_mod'] + opt['input_img']['trans_mod'] + opt['input_img']['seg_mod']
        print(l_input_config)
        for i in range(len(l_input_config)):
            l_input_config[i][1] = l_input_config[i][1].format(input_dir=self.input_dir)
        
        output_dir= opt['output_dir'].format(output_dir=self.output_dir)
        
        # obtain common image names
        img_names_common = set()
        for i, (suffix, input_dir) in enumerate(l_input_config):
            print(suffix,input_dir)
            img_names = sorted([fn.replace(suffix,'') for fn in os.listdir(input_dir.format(input_dir=self.input_dir)) if fn.endswith(suffix)])
            if i == 0:
                img_names_common = set(img_names)
            else:
                img_names_common = img_names_common.intersection(set(img_names))
        img_names_common = list(img_names_common)
        print(f'Found {len(img_names_common)} common image names to proceed with')
        self.img_names = img_names_common
        
        # obtain tile size
        path_img = os.path.join(l_input_config[0][1],f'{img_names_common[0]}{l_input_config[0][0]}')
        w, h = self._get_tile_size(path_img)
        
        # concate tiles
        for i, img_name in enumerate(img_names_common):
            img = Image.new('RGB', (w*len(l_input_config), h))
            
            for i, (suffix, input_dir) in enumerate(l_input_config):
                path_img = os.path.join(input_dir, f'{img_name}{suffix}')
                
                if str(i) in opt['process_mod'].keys():
                    if opt['process_mod'][i] == 'black':
                        img_tile = Image.new('RGB', (w,h)) # default is black
                        img.paste(img_tile,(w*i,0))
                    elif opt['process_mod'][i] == 'cleanStains':
                       img = clean_stains_for_image(img, d_mod_to_modify={int(i):'red'})
                    else:
                        raise Exception(f"Not implemented: process type {opt['process_mod'][i]}")
                else:
                  img_tile = Image.open(path_img)
                  img.paste(img_tile,(w*i,0))
            
            img.save(os.path.join(output_dir,f'{img_name}.png'))
            
            if i % 100 == 0 or i == len(img_names_common)-1:
                print(f'Saved {i+1}/{len(img_names_common)} concatenated images to directory {self.output_dir}')
        
        # now this output directory becomes the input for the next step in the pipeline
        self.input_dir = output_dir 
    
    
    def trainTestSplit(self,opt):
        output_dir = opt['output_dir'].format(output_dir=self.output_dir,train_ratio=opt['train_ratio'])
        output_dir_train = os.path.join(output_dir,'train')
        output_dir_test = os.path.join(output_dir,'val')
        output_dir_test_cli = os.path.join(output_dir,'val_cli')
        ensure_exists(output_dir_train)
        ensure_exists(output_dir_test)
        ensure_exists(output_dir_test_cli)
        self.train_dir = output_dir_train
        self.val_dir = output_dir_test
        
        if self.seg_gen:
            output_dir_test_cli_gt = os.path.join(output_dir,'val_cli_gt')
            ensure_exists(output_dir_test_cli_gt)
        else:
            output_dir_test_cli_gt = None
        
        if not hasattr(self,'img_names') or self.img_names is None:
            img_names = [fn[:-4] for fn in os.listdir(self.input_dir) if fn.endswith('.png')]
            self.img_names = img_names
        
        # assign img_names to train/test set
        if len(opt['img_names_test']) == 0:
            train_ratio = opt['train_ratio']
            random.seed(0)
            print('Train-test split with train ratio',train_ratio)
            self.img_names_train = random.sample(self.img_names,k=int(len(self.img_names)*train_ratio))
            self.img_names_test = [e for e in self.img_names if e not in self.img_names_train]
        else:
            print('Train-test split with provided test set of size',len(opt['img_names_test']))
            self.img_names_test = [e for e in self.img_names if e in opt['img_names_test']]
            self.img_names_train = [e for e in self.img_names if e not in self.img_names_test]
        print('Train vs test set size:', len(self.img_names_train), len(self.img_names_test))
        
        w, h = opt['tile_size'] if not hasattr(self,'tile_size') else self.tile_size
        for img_name in self.img_names:
            img = Image.open(os.path.join(self.input_dir,f'{img_name}.png'))
            if len(opt['process_mod']) > 0:
                for i,process_type in opt['process_mod'].items(): # i (key) from opt['process_mod'] is a string
                    if opt['process_mod'][i] == 'black':
                        img_tile = Image.new('RGB', (w,h)) # default is black
                        img.paste(img_tile,(w*int(i),0))
                    elif opt['process_mod'][i] == 'cleanStains':
                        img = clean_stains_for_image(img, d_mod_to_modify={int(i):'red'})
                    else:
                        raise Exception(f"Not implemented: process type {opt['process_mod'][i]}")
            
            if img_name in self.img_names_test:
                img.save(os.path.join(output_dir_test,f'{img_name}.png'))
            else:
                img.save(os.path.join(output_dir_train,f'{img_name}.png'))
            
            img_input = img.crop((0, 0, w*opt['input_no'], h))
            img_input.save(os.path.join(output_dir_test_cli,f'{img_name}.png'))
            
            if output_dir_test_cli_gt is not None:
                img_seg = img.crop((img.size[0]-w, 0, img.size[0], h)) # the last tile
                img_seg.save(os.path.join(output_dir_test_cli_gt,f'{img_name}_SegRefined.png'))
                
        self.data_dir = os.path.dirname(self.train_dir)
        
    def augmentTrainSet(self,opt):
        if hasattr(self,'train_dir'):
            input_dir = self.train_dir
            #foldername = 'train'#os.path.basename(self.output_dir)
            output_dir = input_dir#os.path.join(self.data_dir,opt['output_dir'].format(output_dir=foldername))
        else:
            input_dir = self.input_dir
            output_dir = opt['output_dir'].format(output_dir=self.output_dir)
        
        from Image_Processing.Image_Processing_Helper_Functions import augment_set
        augment_set(input_dir=input_dir, 
                    output_dir=output_dir, 
                    aug_no=opt['aug_no'], modality_types=opt['modality_types'])
        
        # now this output directory becomes the input for the next step in the pipeline
        #self.input_dir = output_dir
    
    
    def addExistingData(self,opt):
        # for data dir, input dir, and add dir, we assume it is a dataset folder containing subfolders like train and val
        input_dir = self.data_dir if hasattr(self,'data_dir') else self.input_dir
        output_dir = opt['output_dir'].format(output_dir=self.output_dir)
        add_dirs = opt['dir_add']
        
        shutil.copytree(input_dir, output_dir,  dirs_exist_ok=True)
        for copy_dir in [input_dir]+add_dirs:
            print(f'Copying content in {copy_dir} to {output_dir}...')
            shutil.copytree(copy_dir, output_dir, dirs_exist_ok=True)
        
        self.val_dir = os.path.join(output_dir,'val')
        self.output_dir = output_dir
        
        
    def calculateValMetric(self,opt):
        input_dir = self.val_dir if hasattr(self,'val_dir') else self.input_dir
        tile_size = self.tile_size[0] if hasattr(self,'tile_size') else opt['tile_size']

        from deepliif.stat import get_cell_count_metrics
        get_cell_count_metrics(input_dir, dir_save=input_dir, model=opt['model'], tile_size=tile_size)



def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

def modify_opt(d_base, d_update):
    """
    modify options by applying user-passed updates recursively
    """
    d_res = copy.deepcopy(d_base)
    
    for key, value in d_update.items():
        if key in d_res and isinstance(d_res[key], dict) and isinstance(value, dict):
            # recurse dicts
            d_res[key] = modify_opt(d_res[key], value)
        else:
            # replace value (including lists)
            d_res[key] = copy.deepcopy(value)
    
    return d_res

from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation

def clean_stains(patch_to_modify, patch_seg, keep_only='red', keep_boundaries=True, 
                 dilate_structure=np.ones((3, 3))):
    d_color_name2rgb = {'red':[255, 0, 0], # positive
                    'blue':[0, 0, 255], # negative
                    'green':[0, 255, 0], # boundaries
                   }
    
    mask_color_keep = np.all(patch_seg == d_color_name2rgb[keep_only], axis=2)
    if keep_boundaries:
        mask_color_boundaries = np.all(patch_seg == d_color_name2rgb['green'], axis=2)
        
        # dilate target regions slightly to find green pixels adjacent to the target cells
        mask_color_keep_dilated = binary_dilation(mask_color_keep, structure=dilate_structure)

        # find green pixels that are adjacent to target color pixels
        mask_color_boundaries_keep = mask_color_boundaries & mask_color_keep_dilated 

        # final keep mask consists of target cells and _their_ green boundaries
        mask_color_keep = mask_color_keep | mask_color_boundaries_keep

    patch_to_modify[~mask_color_keep] = [0, 0, 0] # use black pixels if not in the keep mask
    return patch_to_modify


def clean_stains_for_image(img, d_mod_to_modify={1:'red'}):
    """
    img: PIL image
    d_mod_to_modify: key is index, starting from 0 referring to the very first patch in the wide image
                     (the first input patch), value is the color to keep
    """
    image_array = np.array(img)

    tile_size, w, c = image_array.shape
    total_no = w // tile_size
    seg_no = 1
    input_no = 1
    modalities_no = total_no - input_no - seg_no
    assert seg_no in [1, modalities_no], 'seg_no should either be 1 or the same as the number of the translation modalities'

    # Extract patches
    l_patch = [image_array[:, i*tile_size:(i+1)*tile_size] for i in range(total_no)]
    l_patch_modified = []
    for i,patch in enumerate(l_patch):
        if i not in d_mod_to_modify.keys():
            l_patch_modified.append(patch)
        else:
            if seg_no == 1:
                patch_seg = l_patch[-1]
            else:
                idx_patch_in_modalities = idx_patch_in_seg = i + 1 - input_no # +1 because index starts from 0
                idx_patch_seg_in_img = input_no + modalities_no + idx_patch_in_seg - 1 # -1 because index starts from 0
                patch_seg = l_patch[idx_patch_seg_in_img]

            patch_modified = clean_stains(patch, patch_seg, keep_only=d_mod_to_modify[i])
            l_patch_modified.append(patch_modified)

    image_array_modified = np.concatenate(l_patch_modified, axis=1)
    img_modified = Image.fromarray(image_array_modified.astype(np.uint8))
    
    return img_modified

def clean_stains_for_file(path_img, d_mod_to_modify={1:'red'}):
    img = Image.open(path_img)
    return clean_stains_for_image(img, d_mod_to_modify)
    
def clean_stains_for_directory(dir_img, dir_img_output=None, d_mod_to_modify={1:'red'}):
    if dir_img_output is None:
        dir_img_output = os.path.join(os.path.dirname(directory),'cleaned')
        
    fns = [fn for fn in os.listdir(dir_img) if fn.endswith('png')]
    for i,fn in enumerate(fns):
        img_modified = clean_stains_for_file(os.path.join(dir_img,fn), d_mod_to_modify)
        img_modified.save(os.path.join(dir_img_output,fn))
        
        if i > 0 and i % 100 == 0:
            print(i,'/',len(fns))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Image processing pipeline')
    parser.add_argument('--input-dir',required=True,type=str,help='path to input images')
    parser.add_argument('--output-dir',type=str,help='path to output images')
    parser.add_argument('--seg-gen', type=lambda x: x.lower() == 'true',default=True,help='True (Translation and Segmentation), False (Only Translation). Default: True')
    parser.add_argument('--steps',type=str,default='concateToWideFormat,trainTestSplit,augmentTrainSet,addExistingData,calculateValMetric',
                       help='steps to run for the input images, separated by comma')
    parser.add_argument('--options',type=str, default=None,help='configuration for pipeline steps as json string')
    
    args = parser.parse_args()
    
    if args.options:
        try:
            options = json.loads(args.options)
        except Exception as e:
            raise Exception('Failed to load option file:',e)
    else:
        options = {}
    
    ensure_exists(args.output_dir)
    pipeline = prepareDataPipeline(args.input_dir, args.output_dir, args.steps, args.seg_gen, options)
    pipeline.run()
