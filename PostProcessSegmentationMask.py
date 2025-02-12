import os.path
import sys
import json

import cv2
import numpy as np
import scipy.ndimage as ndi

from deepliif.postprocessing import compute_results


def post_process_segmentation_mask(input_dir, seg_thresh=150, size_thresh='auto'):
    images = os.listdir(input_dir)
    image_extensions = ['.png', '.jpg', '.tif', '.tiff']

    for img in images:
        seg_file = None

        if '_fake_B_5.png' in img:
            orig_file = os.path.join(input_dir, img.replace('_fake_B_5', '_real_A'))
            seg_file = os.path.join(input_dir, img)
            overlaid_file = os.path.join(input_dir, img.replace('_fake_B_5', '_SegOverlaid'))
            refined_file = os.path.join(input_dir, img.replace('_fake_B_5', '_SegRefined'))
            score_file = os.path.join(input_dir, img.replace('_fake_B_5.png', '.json'))
        elif '_Seg.png' in img:
            orig_img_ext = None
            for img_ext in image_extensions:
                if os.path.exists(os.path.join(input_dir, img.replace('_Seg.png', img_ext))):
                    orig_img_ext = img_ext
                    break
            orig_file = os.path.join(input_dir, img.replace('_Seg.png', orig_img_ext)) if orig_img_ext is not None else None
            seg_file = os.path.join(input_dir, img)
            overlaid_file = os.path.join(input_dir, img.replace('_Seg', '_SegOverlaid'))
            refined_file = os.path.join(input_dir, img.replace('_Seg', '_SegRefined'))
            score_file = os.path.join(input_dir, img.replace('_Seg.png', '.json'))

        if seg_file is not None:
            if orig_file is not None:
                orig_image = cv2.cvtColor(cv2.imread(orig_file), cv2.COLOR_BGR2RGB)
            else:
                orig_image = cv2.cvtColor(cv2.imread(seg_file), cv2.COLOR_BGR2RGB)
            seg_image = cv2.cvtColor(cv2.imread(seg_file), cv2.COLOR_BGR2RGB)
            overlaid, refined, scoring = compute_results(orig_image, seg_image, None, '40x', seg_thresh, size_thresh)
            if orig_file is not None:
                cv2.imwrite(overlaid_file, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
            cv2.imwrite(refined_file, cv2.cvtColor(refined, cv2.COLOR_RGB2BGR))
            if scoring is not None:
                with open(score_file, 'w') as f:
                    json.dump(scoring, f, indent=2)


if __name__ == '__main__':
    base_dir = sys.argv[1]
    segmentation_thresh = 150
    size_thresh = 'auto'
    if len(sys.argv) > 2:
        segmentation_thresh = int(sys.argv[2])
    if len(sys.argv) > 3:
        size_thresh = int(sys.argv[3])

    post_process_segmentation_mask(base_dir, segmentation_thresh, size_thresh)
