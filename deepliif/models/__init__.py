"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See CycleGAN_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""
import base64
import os
import itertools
import importlib
from functools import lru_cache
from io import BytesIO
import json
import math

import requests
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np
from dask import delayed, compute
import openslide

from deepliif.util import *
from deepliif.util.util import tensor_to_pil
from deepliif.data import transform
from deepliif.postprocessing import compute_final_results, compute_cell_results
from deepliif.postprocessing import encode_cell_data_v4, decode_cell_data_v4
from deepliif.options import Options, print_options

from .base_model import BaseModel

# import for init purpose, not used in this script
from .DeepLIIF_model import DeepLIIFModel
from .DeepLIIFExt_model import DeepLIIFExtModel


@lru_cache
def get_opt(model_dir, mode='test'):
    """
    mode: test or train, currently only functions used for inference utilize get_opt so it
          defaults to test
    """
    if mode == 'train':
        opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode=mode)
    elif mode == 'test':
        try:
            opt = Options(path_file=os.path.join(model_dir,'test_opt.txt'), mode=mode)
        except:
            opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode=mode)
        opt.use_dp = False
        opt.gpu_ids = list(range(torch.cuda.device_count()))
    return opt


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "deepliif.models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
                and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from deepliif.models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance


def load_torchscript_model(model_pt_path, device):
    net = torch.jit.load(model_pt_path, map_location=device)
    net = disable_batchnorm_tracking_stats(net)
    net.eval()
    return net



def load_eager_models(opt, devices=None):
    # create a model given model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)

    nets = {}
    if devices:
        model_names = list(devices.keys())
    else:
        model_names = model.model_names
    
    for name in model_names:#model.model_names:
        if isinstance(name, str):
            if '_' in name:
                net = getattr(model, 'net' + name.split('_')[0])[int(name.split('_')[-1]) - 1]
            else:
                net = getattr(model, 'net' + name)
    
            if opt.phase != 'train':
                net.eval()
                net = disable_batchnorm_tracking_stats(net)
            
            # SDG models when loaded are still DP.. not sure why
            if isinstance(net, torch.nn.DataParallel):
                net = net.module

            nets[name] = net
            if devices:
                nets[name].to(devices[name])
            
    return nets


@lru_cache
def init_nets(model_dir, eager_mode=False, opt=None, phase='test'):
    """
    Init DeepLIIF networks so that every net in
    the same group is deployed on the same GPU
    
    opt_args: to overwrite opt arguments in train_opt.txt, typically used in inference stage
              for example, opt_args={'phase':'test'}
    """ 
    if opt is None:
        opt = get_opt(model_dir, mode=phase)
    opt.use_dp = False
    
    if opt.model == 'DeepLIIF':
        net_groups = [
            ('G1', 'G52'),
            ('G2', 'G53'),
            ('G3', 'G54'),
            ('G4', 'G55'),
            ('G51',)
        ]
    elif opt.model in ['DeepLIIFExt','SDG']:
        if opt.seg_gen:
            net_groups = [(f'G_{i+1}',f'GS_{i+1}') for i in range(opt.modalities_no)]
        else:
            net_groups = [(f'G_{i+1}',) for i in range(opt.modalities_no)]
    elif opt.model == 'CycleGAN':
        if opt.BtoA:
            net_groups = [(f'GB_{i+1}',) for i in range(opt.modalities_no)]
        else:
            net_groups = [(f'GA_{i+1}',) for i in range(opt.modalities_no)]
    else:
        raise Exception(f'init_nets() not implemented for model {opt.model}')

    number_of_gpus_all = torch.cuda.device_count()
    number_of_gpus = min(len(opt.gpu_ids),number_of_gpus_all)

    if number_of_gpus > 0:
        mapping_gpu_ids = {i:idx for i,idx in enumerate(opt.gpu_ids)}
        chunks = [itertools.chain.from_iterable(c) for c in chunker(net_groups, number_of_gpus)]
        # chunks = chunks[1:]
        devices = {n: torch.device(f'cuda:{mapping_gpu_ids[i]}') for i, g in enumerate(chunks) for n in g}
        # devices = {n: torch.device(f'cuda:{i}') for i, g in enumerate(chunks) for n in g}
    else:
        devices = {n: torch.device('cpu') for n in itertools.chain.from_iterable(net_groups)}

    if eager_mode:
        return load_eager_models(opt, devices)

    return {
        n: load_torchscript_model(os.path.join(model_dir, f'{n}.pt'), device=d)
        for n, d in devices.items()
    }


def compute_overlap(img_size, tile_size):
    w, h = img_size
    if round(w / tile_size) == 1 and round(h / tile_size) == 1:
        return 0

    return tile_size // 4


def run_torchserve(img, model_path=None, eager_mode=False, opt=None, seg_only=False):
    """
    eager_mode: not used in this function; put in place to be consistent with run_dask
           so that run_wrapper() could call either this function or run_dask with
           same syntax
    opt: same as eager_mode
    seg_only: same as eager_mode
    """
    buffer = BytesIO()
    torch.save(transform(img.resize((opt.scale_size, opt.scale_size))), buffer)

    torchserve_host = os.getenv('TORCHSERVE_HOST', 'http://localhost')
    res = requests.post(
        f'{torchserve_host}/wfpredict/deepliif',
        json={'img': base64.b64encode(buffer.getvalue()).decode('utf-8')}
    )

    res.raise_for_status()

    def deserialize_tensor(bs):
        return torch.load(BytesIO(base64.b64decode(bs.encode())), map_location=torch.device('cpu'))

    return {k: tensor_to_pil(deserialize_tensor(v)) for k, v in res.json().items()}


def run_dask(img, model_path, eager_mode=False, opt=None, seg_only=False):
    model_dir = os.getenv('DEEPLIIF_MODEL_DIR', model_path)
    nets = init_nets(model_dir, eager_mode, opt)
    use_dask = True if opt.norm != 'spectral' else False
    
    if opt.input_no > 1 or opt.model == 'SDG':
        l_ts = [transform(img_i.resize((opt.scale_size,opt.scale_size))) for img_i in img]
        ts = torch.cat(l_ts, dim=1)
    else:
        ts = transform(img.resize((opt.scale_size, opt.scale_size)))
    

    if use_dask:
        @delayed
        def forward(input, model):
            with torch.no_grad():
                return model(input.to(next(model.parameters()).device))
    else: # some train settings like spectral norm some how in inference mode is not compatible with dask
        def forward(input, model):
            with torch.no_grad():
                return model(input.to(next(model.parameters()).device))
    
    if opt.model == 'DeepLIIF':
        weights = {
            'G51': 0.25, # IHC
            'G52': 0.25, # Hema
            'G53': 0.25, # DAPI
            'G54': 0.00, # Lap2
            'G55': 0.25, # Marker
        }

        seg_map = {'G1': 'G52', 'G2': 'G53', 'G3': 'G54', 'G4': 'G55'}
        if seg_only:
            seg_map = {k: v for k, v in seg_map.items() if weights[v] != 0}
        
        lazy_gens = {k: forward(ts, nets[k]) for k in seg_map}
        if 'G4' not in seg_map:
            lazy_gens['G4'] = forward(ts, nets['G4'])
        gens = compute(lazy_gens)[0]
        
        lazy_segs = {v: forward(gens[k], nets[v]).to(torch.device('cpu')) for k, v in seg_map.items()}
        if not seg_only or weights['G51'] != 0:
            lazy_segs['G51'] = forward(ts, nets['G51']).to(torch.device('cpu'))
        segs = compute(lazy_segs)[0]
    
        seg = torch.stack([torch.mul(segs[k], weights[k]) for k in segs.keys()]).sum(dim=0)
    
        if seg_only:
            res = {'G4': tensor_to_pil(gens['G4'])} if 'G4' in gens else {}
        else:
            res = {k: tensor_to_pil(v) for k, v in gens.items()}
            res.update({k: tensor_to_pil(v) for k, v in segs.items()})
        res['G5'] = tensor_to_pil(seg)
    
        return res
    elif opt.model in ['DeepLIIFExt','SDG','CycleGAN']:
        if opt.model == 'CycleGAN':
            seg_map = {f'GB_{i+1}':None for i in range(opt.modalities_no)} if opt.BtoA else {f'GA_{i+1}':None for i in range(opt.modalities_no)}
        else:
            seg_map = {'G_' + str(i): 'GS_' + str(i) for i in range(1, opt.modalities_no + 1)}
        
        if use_dask:
            lazy_gens = {k: forward(ts, nets[k]) for k in seg_map}
            gens = compute(lazy_gens)[0]
        else:
            gens = {k: forward(ts, nets[k]) for k in seg_map}
        
        res = {k: tensor_to_pil(v) for k, v in gens.items()}
    
        if opt.seg_gen:
            if use_dask:
                lazy_segs = {v: forward(torch.cat([ts.to(torch.device('cpu')), gens[next(iter(seg_map))].to(torch.device('cpu')), gens[k].to(torch.device('cpu'))], 1), nets[v]).to(torch.device('cpu')) for k, v in seg_map.items()}
                segs = compute(lazy_segs)[0]
            else:
                segs = {v: forward(torch.cat([ts.to(torch.device('cpu')), gens[next(iter(seg_map))].to(torch.device('cpu')), gens[k].to(torch.device('cpu'))], 1), nets[v]).to(torch.device('cpu')) for k, v in seg_map.items()}
            res.update({k: tensor_to_pil(v) for k, v in segs.items()})
    
        return res
    else:
        raise Exception(f'run_dask() not fully implemented for {opt.model}')


def is_empty(tile):
    thresh = 15
    if isinstance(tile, list): # for pair of tiles, only mark it as empty / no need for prediction if ALL tiles are empty
        return all([True if np.max(image_variance_rgb(t)) < thresh else False for t in tile])
    else:
        return True if np.max(image_variance_rgb(tile)) < thresh else False


def run_wrapper(tile, run_fn, model_path, eager_mode=False, opt=None, seg_only=False):
    if opt.model == 'DeepLIIF':
        if is_empty(tile):
            if seg_only:
                return {
                    'G4': Image.new(mode='RGB', size=(512, 512), color=(10, 10, 10)),
                    'G5': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
                }
            else :
                return {
                    'G1': Image.new(mode='RGB', size=(512, 512), color=(201, 211, 208)),
                    'G2': Image.new(mode='RGB', size=(512, 512), color=(10, 10, 10)),
                    'G3': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
                    'G4': Image.new(mode='RGB', size=(512, 512), color=(10, 10, 10)),
                    'G5': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
                    'G51': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
                    'G52': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
                    'G53': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
                    'G54': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
                    'G55': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
                }
        else:
            return run_fn(tile, model_path, eager_mode, opt, seg_only)
    elif opt.model in ['DeepLIIFExt', 'SDG']:
        if is_empty(tile):
            res = {'G_' + str(i): Image.new(mode='RGB', size=(512, 512)) for i in range(1, opt.modalities_no + 1)}
            res.update({'GS_' + str(i): Image.new(mode='RGB', size=(512, 512)) for i in range(1, opt.modalities_no + 1)})
            return res
        else:
            return run_fn(tile, model_path, eager_mode, opt)
    elif opt.model in ['CycleGAN']:
        if is_empty(tile):
            net_names = ['GB_{i+1}' for i in range(opt.modalities_no)] if opt.BtoA else [f'GA_{i+1}' for i in range(opt.modalities_no)]
            res = {net_name: Image.new(mode='RGB', size=(512, 512)) for net_name in net_names}
            return res
        else:
            return run_fn(tile, model_path, eager_mode, opt)
    else:
        raise Exception(f'run_wrapper() not implemented for model {opt.model}')


def inference(img, tile_size, overlap_size, model_path, use_torchserve=False,
              eager_mode=False, color_dapi=False, color_marker=False, opt=None,
              return_seg_intermediate=False, seg_only=False):
    if not opt:
        opt = get_opt(model_path)
        #print_options(opt)

    run_fn = run_torchserve if use_torchserve else run_dask

    if opt.model == 'SDG':
        # SDG could have multiple input images/modalities, hence the input could be a rectangle.
        # We split the input to get each modality image then create tiles for each set of input images.
        w, h = int(img.width / opt.input_no), img.height
        orig = [img.crop((w * i, 0, w * (i+1), h)) for i in range(opt.input_no)]
    else:
        # Otherwise expect a single input image, which is used directly.
        orig = img

    tiler = InferenceTiler(orig, tile_size, overlap_size)
    for tile in tiler:
        tiler.stitch(run_wrapper(tile, run_fn, model_path, eager_mode, opt, seg_only))
    results = tiler.results()

    if opt.model == 'DeepLIIF':
        if seg_only:
            images = {'Seg': results['G5']}
            if 'G4' in results:
                images.update({'Marker': results['G4']})
        else:
            images = {
                'Hema': results['G1'],
                'DAPI': results['G2'],
                'Lap2': results['G3'],
                'Marker': results['G4'],
                'Seg': results['G5'],
            }
        
        if return_seg_intermediate and not seg_only:
            images.update({'IHC_s':results['G51'],
                          'Hema_s':results['G52'],
                          'DAPI_s':results['G53'],
                          'Lap2_s':results['G54'],
                          'Marker_s':results['G55'],})
        
        if color_dapi and not seg_only:
            matrix = (       0,        0,        0, 0,
                      299/1000, 587/1000, 114/1000, 0,
                      299/1000, 587/1000, 114/1000, 0)
            images['DAPI'] = images['DAPI'].convert('RGB', matrix)
        if color_marker and not seg_only:
            matrix = (299/1000, 587/1000, 114/1000, 0,
                      299/1000, 587/1000, 114/1000, 0,
                             0,        0,        0, 0)
            images['Marker'] = images['Marker'].convert('RGB', matrix)
        return images

    elif opt.model == 'DeepLIIFExt':
        images = {f'mod{i}': results[f'G_{i}'] for i in range(1, opt.modalities_no + 1)}
        if opt.seg_gen:
            images.update({f'Seg{i}': results[f'GS_{i}'] for i in range(1, opt.modalities_no + 1)})
        return images

    elif opt.model == 'SDG':
        images = {f'mod{i}': results[f'G_{i}'] for i in range(1, opt.modalities_no + 1)}
        return images

    else:
        #raise Exception(f'inference() not implemented for model {opt.model}')
        return results # return result images with default key names (i.e., net names)


def postprocess(orig, images, tile_size, model, seg_thresh=150, size_thresh='default', marker_thresh=None, size_thresh_upper=None):
    if model == 'DeepLIIF':
        resolution = '40x' if tile_size > 384 else ('20x' if tile_size > 192 else '10x')
        overlay, refined, scoring = compute_final_results(
            orig, images['Seg'], images.get('Marker'), resolution,
            size_thresh, marker_thresh, size_thresh_upper, seg_thresh)
        processed_images = {}
        processed_images['SegOverlaid'] = Image.fromarray(overlay)
        processed_images['SegRefined'] = Image.fromarray(refined)
        return processed_images, scoring

    elif model in ['DeepLIIFExt','SDG']:
        resolution = '40x' if tile_size > 768 else ('20x' if tile_size > 384 else '10x')
        processed_images = {}
        scoring = {}
        for img_name in list(images.keys()):
            if 'Seg' in img_name:
                seg_img = images[img_name]
                overlay, refined, score = compute_final_results(
                    orig, images[img_name], None, resolution,
                    size_thresh, marker_thresh, size_thresh_upper, seg_thresh)
    
                processed_images[img_name + '_Overlaid'] = Image.fromarray(overlay)
                processed_images[img_name + '_Refined'] = Image.fromarray(refined)
                scoring[img_name] = score
        return processed_images, scoring

    else:
        raise Exception(f'postprocess() not implemented for model {model}')


def infer_modalities(img, tile_size, model_dir, eager_mode=False,
                     color_dapi=False, color_marker=False, opt=None,
                     return_seg_intermediate=False, seg_only=False):
    """
    This function is used to infer modalities for the given image using a trained model.
    :param img: The input image.
    :param tile_size: The tile size.
    :param model_dir: The directory containing serialized model files.
    :return: The inferred modalities and the segmentation mask.
    """
    if opt is None:
        opt = get_opt(model_dir)
        opt.use_dp = False
        #print_options(opt)
    
    # for those with multiple input modalities, find the correct size to calculate overlap_size
    input_no = opt.input_no if hasattr(opt, 'input_no') else 1
    img_size = (img.size[0] / input_no, img.size[1]) # (width, height)

    images = inference(
        img,
        tile_size=tile_size,
        #overlap_size=compute_overlap(img_size, tile_size),
        overlap_size=tile_size//16,
        model_path=model_dir,
        eager_mode=eager_mode,
        color_dapi=color_dapi,
        color_marker=color_marker,
        opt=opt,
        return_seg_intermediate=return_seg_intermediate,
        seg_only=seg_only
    )
    
    if not hasattr(opt,'seg_gen') or (hasattr(opt,'seg_gen') and opt.seg_gen): # the first condition accounts for old settings of deepliif; the second refers to deepliifext models
        post_images, scoring = postprocess(img, images, tile_size, opt.model)
        images = {**images, **post_images}
        return images, scoring
    else:
        return images, None


def infer_results_for_wsi(input_dir, filename, output_dir, model_dir, tile_size, region_size=20000):
    """
    This function infers modalities and segmentation mask for the given WSI image. It

    :param input_dir: The directory containing the WSI.
    :param filename: The WSI name.
    :param output_dir: The directory for saving the inferred modalities.
    :param model_dir: The directory containing the serialized model files.
    :param tile_size: The tile size.
    :param region_size: The size of each individual region to be processed at once.
    :return:
    """
    basename, _ = os.path.splitext(filename)
    results_dir = os.path.join(output_dir, basename)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    size_x, size_y, size_z, size_c, size_t, pixel_type = get_information(os.path.join(input_dir, filename))
    rescale = (pixel_type != 'uint8')
    print(filename, size_x, size_y, size_z, size_c, size_t, pixel_type)

    results = {}
    scoring = None

    # javabridge already set up from previous call to get_information()
    with bioformats.ImageReader(os.path.join(input_dir, filename)) as reader:
        start_x, start_y = 0, 0

        while start_x < size_x:
            while start_y < size_y:
                print(start_x, start_y)
                region_XYWH = (start_x, start_y, min(region_size, size_x - start_x), min(region_size, size_y - start_y))
                region = reader.read(XYWH=region_XYWH, rescale=rescale)
                img = Image.fromarray((region * 255).astype(np.uint8)) if rescale else Image.fromarray(region)

                region_modalities, region_scoring = infer_modalities(img, tile_size, model_dir)
                if region_scoring is not None:
                    if scoring is None:
                        scoring = {
                            'num_pos': region_scoring['num_pos'],
                            'num_neg': region_scoring['num_neg'],
                        }
                    else:
                        scoring['num_pos'] += region_scoring['num_pos']
                        scoring['num_neg'] += region_scoring['num_neg']

                for name, img in region_modalities.items():
                    if name not in results:
                        results[name] = np.zeros((size_y, size_x, 3), dtype=np.uint8)
                    results[name][region_XYWH[1]: region_XYWH[1] + region_XYWH[3],
                    region_XYWH[0]: region_XYWH[0] + region_XYWH[2]] = np.array(img)
                start_y += region_size
            start_y = 0
            start_x += region_size

    # write_results_to_pickle_file(os.path.join(results_dir, "results.pickle"), results)
    # read_results_from_pickle_file(os.path.join(results_dir, "results.pickle"))

    for name, img in results.items():
        write_big_tiff_file(os.path.join(results_dir, f'{basename}_{name}.ome.tiff'), img, tile_size)

    if scoring is not None:
        scoring['num_total'] = scoring['num_pos'] + scoring['num_neg']
        scoring['percent_pos'] = round(scoring['num_pos'] / scoring['num_total'] * 100, 1) if scoring['num_pos'] > 0 else 0
        with open(os.path.join(results_dir, f'{basename}.json'), 'w') as f:
            json.dump(scoring, f, indent=2)

    javabridge.kill_vm()


def get_wsi_resolution(filename):
    """
    Use OpenSlide to get the resolution (magnification) of the slide
    and the corresponding tile size to use by default for DeepLIIF.
    If it cannot be found, return (None, None) instead.

    Parameters
    ----------
    filename : str
        Full path to the file.

    Returns
    -------
    str :
        Magnification (objective power) as found by OpenSlide.
    int :
        Corresponding tile size for DeepLIIF.
    """
    try:
        image = openslide.OpenSlide(filename)
        mag = image.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        tile_size = round((float(mag) / 40) * 512)
        return mag, tile_size
    except Exception as e:
        return None, None


def infer_cells_for_wsi(filename, model_dir, tile_size, region_size=20000, version=3, print_log=False):
    """
    Perform inference on a slide and get the results individual cell data.

    Parameters
    ----------
    filename : str
        Full path to the file.
    model_dir : str
        Full path to the directory with the DeepLIIF model files.
    tile_size : int
        Size of tiles to extract and perform inference on.
    region_size : int
        Maximum size to split the slide for processing.
    version : int
        Version of cell data to return (3 or 4).
    print_log : bool
        Whether or not to print updates while processing.

    Returns
    -------
    dict :
        Individual cell data and associated values.
    """

    def print_info(*args):
        if print_log:
            print(*args, flush=True)

    resolution = '40x' if tile_size > 384 else ('20x' if tile_size > 192 else '10x')

    size_x, size_y, size_z, size_c, size_t, pixel_type = get_information(filename)
    rescale = (pixel_type != 'uint8')
    print_info('Info:', size_x, size_y, size_z, size_c, size_t, pixel_type)

    num_regions_x = math.ceil(size_x / region_size)
    num_regions_y = math.ceil(size_y / region_size)
    stride_x = math.ceil(size_x / num_regions_x)
    stride_y = math.ceil(size_y / num_regions_y)
    print_info('Strides:', stride_x, stride_y)

    data = None
    default_marker_thresh, count_marker_thresh = 0, 0
    default_size_thresh, count_size_thresh = 0, 0

    # javabridge already set up from previous call to get_information()
    with bioformats.ImageReader(filename) as reader:
        start_x, start_y = 0, 0

        while start_y < size_y:
            while start_x < size_x:
                region_XYWH = (start_x, start_y, min(stride_x, size_x-start_x), min(stride_y, size_y-start_y))
                print_info('Region:', region_XYWH)

                region = reader.read(XYWH=region_XYWH, rescale=rescale)
                print_info(region.shape, region.dtype)
                img = Image.fromarray((region * 255).astype(np.uint8)) if rescale else Image.fromarray(region)
                print_info(img.size, img.mode)

                images = inference(
                    img,
                    tile_size=tile_size,
                    overlap_size=tile_size//16,
                    model_path=model_dir,
                    eager_mode=False,
                    color_dapi=False,
                    color_marker=False,
                    opt=None,
                    return_seg_intermediate=False,
                    seg_only=True,
                )
                region_data = compute_cell_results(images['Seg'], images.get('Marker'), resolution, version=version)

                if start_x != 0 or start_y != 0:
                    for i in range(len(region_data['cells'])):
                        cell = decode_cell_data_v4(region_data['cells'][i]) if version == 4 else region_data['cells'][i]
                        for j in range(2):
                            cell['bbox'][j] = (cell['bbox'][j][0] + start_x, cell['bbox'][j][1] + start_y)
                        cell['centroid'] = (cell['centroid'][0] + start_x, cell['centroid'][1] + start_y)
                        for j in range(len(cell['boundary'])):
                            cell['boundary'][j] = (cell['boundary'][j][0] + start_x, cell['boundary'][j][1] + start_y)
                        region_data['cells'][i] = encode_cell_data_v4(cell) if version == 4 else cell

                if data is None:
                    data = region_data
                else:
                    data['cells'] += region_data['cells']

                if region_data['settings']['default_marker_thresh'] is not None and region_data['settings']['default_marker_thresh'] != 0:
                    default_marker_thresh += region_data['settings']['default_marker_thresh']
                    count_marker_thresh += 1
                if region_data['settings']['default_size_thresh'] != 0:
                    default_size_thresh += region_data['settings']['default_size_thresh']
                    count_size_thresh += 1

                start_x += stride_x

            start_x = 0
            start_y += stride_y

    javabridge.kill_vm()

    if count_marker_thresh == 0:
        count_marker_thresh = 1
    if count_size_thresh == 0:
        count_size_thresh = 1
    data['settings']['default_marker_thresh'] = round(default_marker_thresh / count_marker_thresh)
    data['settings']['default_size_thresh'] = round(default_size_thresh / count_size_thresh)

    return data
