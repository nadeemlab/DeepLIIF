import os
import json
from collections import namedtuple

import torch
import numpy as np
from PIL import Image
from dask import delayed, compute

from deepliif.options.processing_options import ProcessingOptions
from deepliif.preprocessing import allowed_file, generate_tiles, Tile, transform
from deepliif.models import init_nets
from deepliif.postprocessing import stitch, adjust_marker, adjust_dapi, compute_IHC_scoring, \
    overlay_final_segmentation_mask, create_final_segmentation_mask_with_boundaries, create_basic_segmentation_mask
from deepliif.util import util

model_dir = os.getenv('DEEPLIIF_MODEL_DIR', './model-server/DeepLIIF_Latest_Model/')


@util.timeit
def run_models(nets, img):
    ts = transform(img.resize((512, 512)))

    @delayed
    def forward(input, model):
        with torch.no_grad():
            return model(input.to(next(model.parameters()).device))

    gen_map = {
        'hema': 'G1',
        'dapi': 'G2',
        'lap2': 'G3',
        'ki67': 'G4',
    }

    lazy_gens = {k: forward(ts, nets[v]) for k, v in gen_map.items()}
    gens = compute(lazy_gens)[0]

    seg_map = {
        'hema': 'G52',
        'dapi': 'G53',
        'lap2': 'G54',
        'ki67': 'G55',
    }

    lazy_segs = {f'{k}_seg': forward(gens[k], nets[v]).to(torch.device('cpu')) for k, v in seg_map.items()}
    lazy_segs['original_seg'] = forward(ts, nets['G51']).to(torch.device('cpu'))
    segs = compute(lazy_segs)[0]

    seg_weights = [0.25, 0.15, 0.25, 0.1, 0.25]
    seg = torch.stack([
        torch.mul(segs['original_seg'], seg_weights[0]),
        torch.mul(segs['hema_seg'], seg_weights[1]),
        torch.mul(segs['dapi_seg'], seg_weights[3]),
        torch.mul(segs['lap2_seg'], seg_weights[2]),
        torch.mul(segs['ki67_seg'], seg_weights[4]),
    ]).sum(dim=0)

    return {
        'g1': util.tensor_to_pil(gens['hema']),
        'g2': util.tensor_to_pil(gens['dapi']),
        'g3': util.tensor_to_pil(gens['lap2']),
        'g4': util.tensor_to_pil(gens['ki67']),
        'g5': util.tensor_to_pil(seg)
    }


def inference(img, tile_size, overlap_size):
    nets = init_nets(model_dir)

    tiles = list(generate_tiles(img, tile_size, overlap_size))

    res = [Tile(t.i, t.j, run_models(nets, t.img)) for t in tiles]

    images = {}

    hema_tiles = [Tile(t.i, t.j, t.img['g1']) for t in res]

    images['Hema'] = stitch(hema_tiles, tile_size, overlap_size).resize(img.size)

    dapi_tiles = [Tile(t.i, t.j, t.img['g2']) for t in res]
    post_dapi_tiles = [
        Tile(t.i, t.j, adjust_dapi(dt.img, t.img))
        for t, dt in zip(tiles, dapi_tiles)
    ]
    images['DAPI'] = stitch(post_dapi_tiles, tile_size, overlap_size).resize(img.size)

    lap2_tiles = [Tile(t.i, t.j, t.img['g3']) for t in res]
    images['DAPILap2'] = stitch(lap2_tiles, tile_size, overlap_size).resize(img.size)

    ki67_tiles = [Tile(t.i, t.j, t.img['g4']) for t in res]
    post_ki67_tiles = [
        Tile(t.i, t.j, adjust_marker(kt.img, t.img))
        for t, kt in zip(tiles, ki67_tiles)
    ]
    images['Ki67'] = stitch(post_ki67_tiles, tile_size, overlap_size).resize(img.size)

    seg_tiles = [Tile(t.i, t.j, t.img['g5']) for t in res]
    seg_img = stitch(seg_tiles, tile_size, overlap_size).resize(img.size)
    images['Seg'] = seg_img

    mask_image = create_basic_segmentation_mask(np.array(img), np.array(seg_img))

    images['SegOverlaid'] = Image.fromarray(overlay_final_segmentation_mask(np.array(img), mask_image))

    refined_image = create_final_segmentation_mask_with_boundaries(np.array(img))
    images['SegRefined'] = Image.fromarray(refined_image)

    all_cells_no, positive_cells_no, negative_cells_no, IHC_score = compute_IHC_scoring(mask_image)
    scoring = {
        'num_total': all_cells_no,
        'num_pos': positive_cells_no,
        'num_neg': negative_cells_no,
        'percent_pos': IHC_score
    }

    return images, scoring


def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


def compute_overlap(img_size, tile_size):
    w, h = img_size
    if round(w / tile_size) == 1 and round(h / tile_size) == 1:
        return 0

    return tile_size // 4


if __name__ == '__main__':
    opt = ProcessingOptions().parse()

    output_dir = opt.output_dir or opt.input_dir
    ensure_exists(output_dir)

    image_files = [fn for fn in os.listdir(opt.input_dir) if allowed_file(fn)]

    for filename in image_files:

        img = Image.open(os.path.join(opt.input_dir, filename))

        images, scoring = inference(
            img,
            tile_size=opt.tile_size,
            overlap_size=compute_overlap(img.size, opt.tile_size)
        )

        for name, i in images.items():
            i.save(os.path.join(
                output_dir,
                filename.replace('.' + filename.split('.')[-1], f'_{name}.png')
            ))

        print(json.dumps(scoring, indent=2))
