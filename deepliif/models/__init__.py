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
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""
import base64
import os
import itertools
import importlib
from functools import lru_cache
from io import BytesIO

import requests
import torch
from PIL import Image
import numpy as np
from dask import delayed, compute

from deepliif.util import generate_tiles, stitch, Tile, chunker, calculate_background_mean_value, \
    calculate_background_area, adjust_background_tile
from deepliif.util.util import tensor_to_pil
from deepliif.data import transform
from deepliif.postprocessing import adjust_marker, adjust_dapi, compute_IHC_scoring, \
    overlay_final_segmentation_mask, create_final_segmentation_mask_with_boundaries, create_basic_segmentation_mask

from .base_model import BaseModel
from .DeepLIIF_model import DeepLIIFModel
from .networks import get_norm_layer, ResnetGenerator, UnetGenerator


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
    return torch.jit.load(model_pt_path, map_location=device)


def read_model_params(file_addr):
    with open(file_addr) as f:
        lines = f.readlines()
    param_dict = {}
    for line in lines:
        if ':' in line:
            key = line.split(':')[0].strip()
            val = line.split(':')[1].split('[')[0].strip()
            param_dict[key] = val
    print(param_dict)
    return param_dict


def load_eager_models(model_dir, devices):
    input_nc = 3
    output_nc = 3
    ngf = 64
    norm = 'batch'
    use_dropout = True
    padding_type = 'zero'

    files = os.listdir(model_dir)
    for f in files:
        if 'train_opt.txt' in f:
            param_dict = read_model_params(os.path.join(model_dir, f))
            input_nc = int(param_dict['input_nc'])
            output_nc = int(param_dict['output_nc'])
            ngf = int(param_dict['ngf'])
            norm = param_dict['norm']
            use_dropout = False if param_dict['no_dropout'] == 'True' else True
            padding_type = param_dict['padding']

    norm_layer = get_norm_layer(norm_type=norm)

    nets = {}
    for n in ['G1', 'G2', 'G3', 'G4']:
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, padding_type=padding_type)
        net.load_state_dict(torch.load(
            os.path.join(model_dir, f'latest_net_{n}.pth'),
            map_location=devices[n]
        ))
        nets[n] = net

    for n in ['G51', 'G52', 'G53', 'G54', 'G55']:
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        net.load_state_dict(torch.load(
            os.path.join(model_dir, f'latest_net_{n}.pth'),
            map_location=devices[n]
        ))
        nets[n] = net

    return nets


@lru_cache
def init_nets(model_dir, eager_mode=False):
    """
    Init DeepLIIF networks so that every net in
    the same group is deployed on the same GPU
    """
    net_groups = [
        ('G1', 'G52'),
        ('G2', 'G53'),
        ('G3', 'G54'),
        ('G4', 'G55'),
        ('G51',)
    ]

    number_of_gpus = torch.cuda.device_count()
    # number_of_gpus = 0
    if number_of_gpus:
        chunks = [itertools.chain.from_iterable(c) for c in chunker(net_groups, number_of_gpus)]
        # chunks = chunks[1:]
        devices = {n: torch.device(f'cuda:{i}') for i, g in enumerate(chunks) for n in g}
    else:
        devices = {n: torch.device('cpu') for n in itertools.chain.from_iterable(net_groups)}

    if eager_mode:
        return load_eager_models(model_dir, devices)

    return {
        n: load_torchscript_model(os.path.join(model_dir, f'{n}.pt'), device=d)
        for n, d in devices.items()
    }


def compute_overlap(img_size, tile_size):
    w, h = img_size
    if round(w / tile_size) == 1 and round(h / tile_size) == 1:
        return 0

    return tile_size // 4


def run_torchserve(img, model_path=None):
    buffer = BytesIO()
    torch.save(transform(img.resize((512, 512))), buffer)

    torchserve_host = os.getenv('TORCHSERVE_HOST', 'http://localhost')
    res = requests.post(
        f'{torchserve_host}/wfpredict/deepliif',
        json={'img': base64.b64encode(buffer.getvalue()).decode('utf-8')}
    )

    res.raise_for_status()

    def deserialize_tensor(bs):
        return torch.load(BytesIO(base64.b64decode(bs.encode())), map_location=torch.device('cpu'))

    return {k: tensor_to_pil(deserialize_tensor(v)) for k, v in res.json().items()}


def run_dask(img, model_path):
    model_dir = os.getenv('DEEPLIIF_MODEL_DIR', model_path)
    nets = init_nets(model_dir)

    ts = transform(img.resize((512, 512)))

    @delayed
    def forward(input, model):
        with torch.no_grad():
            return model(input.to(next(model.parameters()).device))

    seg_map = {'G1': 'G52', 'G2': 'G53', 'G3': 'G54', 'G4': 'G55'}

    lazy_gens = {k: forward(ts, nets[k]) for k in seg_map}
    gens = compute(lazy_gens)[0]

    lazy_segs = {v: forward(gens[k], nets[v]).to(torch.device('cpu')) for k, v in seg_map.items()}
    lazy_segs['G51'] = forward(ts, nets['G51']).to(torch.device('cpu'))
    segs = compute(lazy_segs)[0]

    seg_weights = [0.25, 0.25, 0.25, 0, 0.25]
    seg = torch.stack([torch.mul(n, w) for n, w in zip(segs.values(), seg_weights)]).sum(dim=0)

    res = {k: tensor_to_pil(v) for k, v in gens.items()}
    res['G5'] = tensor_to_pil(seg)

    return res


def is_empty(tile, mean_background_val):
    # return True if np.mean(np.array(tile) - np.array(mean_background_val)) < 40 else False
    return True if calculate_background_area(tile) > 98 else False


def run_wrapper(tile, run_fn, model_path, mean_background_val):
    if is_empty(tile, mean_background_val):
        return {
            'G1': Image.new(mode='RGB', size=(512, 512), color=(201, 211, 208)),
            'G2': Image.new(mode='RGB', size=(512, 512), color=(10, 10, 10)),
            'G3': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0)),
            'G4': Image.new(mode='RGB', size=(512, 512), color=(10, 10, 10)),
            'G5': Image.new(mode='RGB', size=(512, 512), color=(0, 0, 0))
        }
    else:
        return run_fn(tile, model_path)


def inference(img, tile_size, overlap_size, model_path, use_torchserve=False):

    mean_background_val = calculate_background_mean_value(img)
    tiles = list(generate_tiles(img, tile_size, overlap_size, mean_background_val))

    run_fn = run_torchserve if use_torchserve else run_dask
    # res = [Tile(t.i, t.j, run_fn(t.img, model_path)) for t in tiles]
    res = [Tile(t.i, t.j, run_wrapper(t.img, run_fn, model_path, mean_background_val)) for t in tiles]

    def get_net_tiles(n):
        return [Tile(t.i, t.j, t.img[n]) for t in res]

    images = {}

    images['Hema'] = stitch(get_net_tiles('G1'), tile_size, overlap_size).resize(img.size)

    # images['DAPI'] = stitch(
    #     [Tile(t.i, t.j, adjust_background_tile(dt.img))
    #      for t, dt in zip(tiles, get_net_tiles('G2'))],
    #     tile_size, overlap_size).resize(img.size)
    # dapi_pix = np.array(images['DAPI'])
    # dapi_pix[:, :, 0] = 0
    # images['DAPI'] = Image.fromarray(dapi_pix)

    images['DAPI'] = stitch(get_net_tiles('G2'), tile_size, overlap_size).resize(img.size)
    dapi_pix = np.array(images['DAPI'].convert('L').convert('RGB'))
    dapi_pix[:, :, 0] = 0
    images['DAPI'] = Image.fromarray(dapi_pix)
    images['Lap2'] = stitch(get_net_tiles('G3'), tile_size, overlap_size).resize(img.size)
    images['Marker'] = stitch(get_net_tiles('G4'), tile_size, overlap_size).resize(img.size)
    marker_pix = np.array(images['Marker'].convert('L').convert('RGB'))
    marker_pix[:, :, 2] = 0
    images['Marker'] = Image.fromarray(marker_pix)

    # images['Marker'] = stitch(
    #     [Tile(t.i, t.j, kt.img)
    #      for t, kt in zip(tiles, get_net_tiles('G4'))],
    #     tile_size, overlap_size).resize(img.size)

    images['Seg'] = stitch(get_net_tiles('G5'), tile_size, overlap_size).resize(img.size)

    return images


def postprocess(img, seg_img, thresh=80, noise_objects_size=20, small_object_size=50):
    mask_image = create_basic_segmentation_mask(np.array(img), np.array(seg_img),
                                                thresh, noise_objects_size, small_object_size)
    images = {}
    images['SegOverlaid'] = Image.fromarray(overlay_final_segmentation_mask(np.array(img), mask_image))
    images['SegRefined'] = Image.fromarray(create_final_segmentation_mask_with_boundaries(np.array(mask_image)))

    all_cells_no, positive_cells_no, negative_cells_no, IHC_score = compute_IHC_scoring(mask_image)
    scoring = {
        'num_total': all_cells_no,
        'num_pos': positive_cells_no,
        'num_neg': negative_cells_no,
        'percent_pos': IHC_score
    }

    return images, scoring
