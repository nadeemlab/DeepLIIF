import os

from PIL import Image
import json
from ..models import postprocess
import re

def get_cell_count_metrics(dir_seg, 
                           dir_input = None,
                           dir_save = None, 
                           model = "DeepLIIF",
                           tile_size = 512, 
                           single_tile = False,
                           use_marker = False,
                           suffix_seg = '5',
                           suffix_marker = '4',
                           save_individual = False):
    """
    Obtain cell count metrics through postprocess functions.
    Currenlty implemented only for ground truth tiles.
    Eligible for data used in model type:
      DeepLIIF (with segmentation task)
      DeepLIIFKD (with segmentation task)
    
    dir_seg: directory to find segmentation (and marker if specified) images
    dir_input: directory to find input tile images, only used when single_tile 
               is True and the filenames are expected to be <img name>.png; if 
               not specified, default to dir_seg
    dir_save: directory to save the results out; if not specified, default to
              dir_seg
    model: model type for postprocess function to understand how to handle the 
           data (DeepLIIF, DeepLIIFExt)
    tile_size: tile size used for postprocess calculation
    single_tile: True if the images are single-tile images, and image name 
                 should follow <img name>_<mod suffix>.png; use False if the 
                 images consisting of a row of multiple tiles like those used in
                 training or validation (in this case the segmentation tile is
                 the last one or several tiles)
    use_marker: whether to use the marker image (if True, assumes the marker
                image is the second last tile (single_tile=False) or has a 
                suffix of <suffix_marker> (single_tile=True))
    suffix_seg: filename suffix for segmentation images if single_tile is True
    suffix_marker: filename suffix for marker images if single_tile is True
    save_individual: save cell count statistics for each individual image
    """
    dir_save = dir_save if dir_seg is None else dir_save
    
    if single_tile:
        fns = [x for x in os.listdir(dir_seg) if x.endswith(f'_{suffix_seg}.png') or x.endswith(f'_{suffix_marker}.png')]
        fns = list(set(['_'.join(x.split('_')[:-1]) for x in fns])) # fns do not have extention and mod suffix
    else:
        fns = [x for x in os.listdir(dir_seg) if x.endswith('.png')] # fns have extension
    
    d_metrics = {}
    count = 0
    for fn in fns:
        if single_tile:
            dir_input = dir_input if dir_seg is None else dir_input
            img_gt = Image.open(os.path.join(dir_seg, fn + f'_{suffix_seg}.png'))
            img_marker = Image.open(os.path.join(dir_seg, fn + f'_{suffix_marker}.png'))
            img_input = Image.open(os.path.join(dir_input,fn+'.png'))
            k = fn
        else:
            img = Image.open(os.path.join(dir_seg,fn))
            w, h = img.size
            
            # assume in the row of tiles, the first is the input and the last is the ground truth
            img_input = img.crop((0,0,h,h))
            img_gt = img.crop((w-h,0,w,h))
            img_marker = img.crop((w-h*2,0,w-h,h)) # the second last is marker, if marker is included
            k = os.path.splitext(fn)[0] #re.sub('\..*?$','',fn) # remove extension
    
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
