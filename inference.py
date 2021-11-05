import os

from PIL import Image

from deepliif.options.processing_options import ProcessingOptions
from deepliif.preprocessing import allowed_file
from deepliif.models import init_nets
from deepliif.postprocessing import stitch, overlay, refine, adjust_marker, adjust_dapi, compute_IHC_scoring, \
    create_final_segmentation_mask,\
    overlay_final_segmentation_mask, \
    create_final_segmentation_mask_with_boundaries
from deepliif.preprocessing import generate_tiles, Tile, transform

import torch
import numpy as np
from deepliif.util import util


def infer_images(input_dir, output_dir, filename, tile_size, overlap_size):
    noise_objects_size = 0
    model_dir = os.getenv('DEEPLIIF_MODEL_DIR', 'DeepLIIF/checkpoints/DeepLIIF_Latest_Model/')
    nets = init_nets(model_dir)

    device = torch.device('cpu') if os.getenv('DEEPLIIF_PROC') == 'cpu' else torch.device('cuda')

    def segment_tile(t, hema, dpi, lap2, ki67):
        with torch.no_grad():
            # Segmentation mask generator from IHC input image
            ihc_mask = nets['G51'].to(device)(transform(t.img).to(device))
            # Segmentation mask generator from Hematoxylin input image
            hema_mask = nets['G52'].to(device)(hema.img.to(device))
            # Segmentation mask generator from mpIF DAPI input image
            dpi_mask = nets['G53'].to(device)(dpi.img.to(device))
            # Segmentation mask generator from mpIF Lap2 input image
            lap2_mask = nets['G54'].to(device)(lap2.img.to(device))
            # Segmentation mask generator from Ki67 input image
            ki67_mask = nets['G55'].to(device)(ki67.img.to(device))

        seg_weights = [0.25, 0.15, 0.25, 0.1, 0.25]

        return Tile(t.i, t.j, torch.stack([
            torch.mul(ihc_mask, seg_weights[0]),
            torch.mul(hema_mask, seg_weights[1]),
            torch.mul(dpi_mask, seg_weights[2]),
            torch.mul(lap2_mask, seg_weights[3]),
            torch.mul(ki67_mask, seg_weights[4])
        ]).sum(dim=0))

    overlap_size = int(tile_size / 4)
    img = Image.open(os.path.join(input_dir, filename))
    w, h = img.size
    if abs(w - tile_size) < overlap_size and abs(h - tile_size) < overlap_size:
        overlap_size = 0

    # generate the tiles and resize them to the
    # nets input size 512x512
    tiles = [Tile(t.i, t.j, t.img.resize((512, 512)))
             for t in generate_tiles(img, tile_size, overlap_size)]

    def eval_net(net):
        def eval_tile(t):
            with torch.no_grad():
                return Tile(t.i, t.j, nets[net].to(device)(transform(t.img).to(device)))

        return [eval_tile(t) for t in tiles]

    def stitch_tensor_tiles(ts):
        return stitch(
            [Tile(t.i, t.j, util.tensor_to_pil(t.img)) for t in ts],
            tile_size,
            overlap_size
        ).resize(img.size)

    def stitch_pil_tiles(ts):
        return stitch(
            ts,
            tile_size,
            overlap_size
        ).resize(img.size)

    hema_tiles = eval_net('G1')
    util.save_image(
        np.array(stitch_tensor_tiles(hema_tiles)),
        os.path.join(output_dir, filename.replace('.' + filename.split('.')[-1], '_Hema.png'))
    )

    dpi_tiles = eval_net('G2')
    dpi_pil_tiles = [Tile(dpi_tiles[tile_no].i, dpi_tiles[tile_no].j,
                          adjust_dapi(util.tensor_to_pil(dpi_tiles[tile_no].img),
                                           tiles[tile_no].img))
                     for tile_no in range(len(dpi_tiles))]
    util.save_image(
        np.array(stitch_pil_tiles(dpi_pil_tiles)),
        os.path.join(output_dir, filename.replace('.' + filename.split('.')[-1], '_DAPI.png'))
    )

    lap2_tiles = eval_net('G3')
    util.save_image(
        np.array(stitch_tensor_tiles(lap2_tiles)),
        os.path.join(output_dir, filename.replace('.' + filename.split('.')[-1], '_Lap2.png'))
    )

    ki67_tiles = eval_net('G4')
    ki67_pil_tiles = [Tile(ki67_tiles[tile_no].i, ki67_tiles[tile_no].j,
                          adjust_marker(util.tensor_to_pil(ki67_tiles[tile_no].img),
                                             tiles[tile_no].img))
                      for tile_no in range(len(ki67_tiles))]
    marker_image = np.array(stitch_pil_tiles(ki67_pil_tiles))
    util.save_image(
        marker_image,
        os.path.join(output_dir, filename.replace('.' + filename.split('.')[-1], '_Marker.png'))
    )

    seg_img = stitch_tensor_tiles(
        [segment_tile(*ts) for ts in zip(tiles, hema_tiles, dpi_tiles, lap2_tiles, ki67_tiles)]
    )
    util.save_image(
        np.array(seg_img),
        os.path.join(output_dir, filename.replace('.' + filename.split('.')[-1], '_Seg.png'))
    )

    mask_image = create_final_segmentation_mask(np.array(img), np.array(seg_img), np.array(marker_image))

    util.save_image(
        np.array(Image.fromarray(overlay_final_segmentation_mask(np.array(img), mask_image))),
        os.path.join(output_dir, filename.replace('.' + filename.split('.')[-1], '_SegOverlaid.png'))
    )

    util.save_image(
        np.array(Image.fromarray(create_final_segmentation_mask_with_boundaries(mask_image))),
        os.path.join(output_dir, filename.replace('.' + filename.split('.')[-1], '_SegRefined.png'))
    )

    all_cells_no, positive_cells_no, negative_cells_no, IHC_score = compute_IHC_scoring(mask_image)
    print('image name:', filename.split('.')[0],
          'number of all cells:', all_cells_no,
          'number of positive cells:', positive_cells_no,
          'number of negative cells:', negative_cells_no,
          'IHC Score:', IHC_score)


if __name__ == '__main__':
    opt = ProcessingOptions().parse()
    for img_name in os.listdir(opt.input_dir):
        if allowed_file(img_name):
            output_addr = opt.output_dir if opt.output_dir != '' else opt.input_dir
            infer_images(opt.input_dir, output_addr, img_name, opt.tile_size, opt.overlap_size)
