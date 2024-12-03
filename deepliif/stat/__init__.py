import os
import json
from PIL import Image
from ..models import postprocess
import re

def get_cell_count_metrics(dir_img, dir_save=None, model = None,
                           tile_size=512, single_tile=False,
                           use_marker = False,
                           save_individual = False):
    """
    Obtain cell count metrics through postprocess functions.
    Currenlty implemented only for ground truth tiles.
    
    dir_img: directory to load images for calculation
    dir_save: directory to save the results out
    model: model type (DeepLIIF, DeepLIIFExt, SDG)
    tile_size: tile size used for postprocess calculation
    single_tile: True if the images are single-tile images; use False if the 
                 images contain a row of multiple tiles like those used in
                 training or validation
    use_marker: whether to use the marker image (if Truem, assumes the marker
                image is the second last tile (single_tile=False) or has a 
                suffix of "_4.png" (single_tile=True))
    """
    dir_save = dir_save if dir_img is None else dir_save
    
    if model is None:
        if 'sdg' in dir_img:
            model = 'SDG'
        elif 'ext' in dir_img:
            model = 'DeepLIIFExt'
        else:
            model = 'DeepLIIF'
    
    if single_tile:
        fns = [x for x in os.listdir(dir_img) if x.endswith('_5.png') or x.endswith('_4.png')]
        fns = list(set([x[:-6] for x in fns])) # fns do not have extention
    else:
        fns = [x for x in os.listdir(dir_img) if x.endswith('.png')] # fns have extension
    
    d_metrics = {}
    count = 0
    for fn in fns:
        if single_tile:
            img_gt = Image.open(os.path.join(dir_img,fn+'_5.png'))
            img_marker = Image.open(os.path.join(dir_img,fn+'_4.png'))
            img_input = Image.open(os.path.join(dir_img.replace('/gt','/input'),fn+'.png'))
            k = fn
        else:
            img = Image.open(os.path.join(dir_img,fn))
            w, h = img.size
            
            # assume in the row of tiles, the first is the input and the last is the ground truth
            img_input = img.crop((0,0,h,h))
            img_gt = img.crop((w-h,0,w,h))
            img_marker = img.crop((w-h*2,0,w-h,h)) # the second last is marker, if marker is included
            k = re.sub('\..*?$','',fn) # remove extension
    
        images = {'Seg':img_gt}
        if use_marker:
            images['Marker'] = img_marker
        
        post_images, scoring = postprocess(img_input, images, tile_size, model)
        d_metrics[k] = scoring
        
        if save_individual:
            with open(os.path.join(
                dir_save,
                k+'.json'
            ), 'w') as f:
                json.dump(scoring, f, indent=2)
        
        count += 1
        
        if count % 100 == 0 or count == len(fns):
            print(count,'/',len(fns))

    with open(os.path.join(
                dir_save,
                'metrics.json'
            ), 'w') as f:
                json.dump(d_metrics, f, indent=2)      
