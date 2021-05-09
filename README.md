### DeepLIIF: Deep-Learning Inferred Multiplex Immunofluoresence for IHC Quantification

Reporting biomarkers assessed by routine immunohistochemical (IHC) staining of tissue is broadly used in diagnostic pathology laboratories for patient care. To date, clinical reporting is predominantly qualitative or semi-quantitative. By creating a multitask deep learning framework referred to as DeepLIIF, we are presenting a single step solution to nuclear segmentation and quantitative single-cell IHC scoring. Leveraging a unique de novo dataset of co-registered IHC and multiplex immunoflourescence (mpIF) data generated from the same tissue section, we simultaneously segment and translate low-cost and prevalent IHC slides to more expensive-yet-informative mpIF images. Moreover, a nuclear-pore marker, LAP2beta, is co-registered to improve cell segmentation and protein expression quantification on IHC slides. By formulating the IHC quantification as cell instance segmentation/classification rather than cell detection problem, we show that our model trained on clean IHC Ki67 data can generalize to more noisy and artifact-ridden images as well as other nuclear and non-nuclear markers such as CD3, CD8, BCL2, BCL6, MYC, MUM1, CD10 and TP53. We thoroughly evaluate our method on publicly available bench-mark datasets as well as against pathologists’ semi-quantitative scoring.

© This code is made available for non-commercial academic purposes.

## Preprocessing:
```
python preprocessing.py **--input_di**r /path/to/input/images **--output_dir** /path/to/output/images **--tile_size** size_of_each_cropped_tile **--overlap_size** overlap_size_between_crops **--resize_self** if_True_resizes_to_the_closest_rectangle_if_False_resize_size_need_to_be_set **--resize_size**
```

## Postprocessing:
```
python postprocessing.py **--input_dir** /path/to/preprocessed/images **--output_dir** /path/to/output/images **--input_orig_dir** /path/to/original/input/images --tile_size size_of_each_cropped_tile **--overlap_size** overlap_size_between_crops **--resize_self** if_True_resizes_to_the_closest_rectangle_if_False_resize_size_need_to_be_set **--resize_size**
```

## Training:
```
python train.py **--dataroot** /path/to/input/images **--name** model_name **--model** DeepLIIF **--netG** resnet_9blocks 
```

## Testing:
```
python test.py **--dataroot** /path/to/input/images **--name** model_name **--model** DeepLIIF **--netG** resnet_9blocks 
```

## Demo:
Change the DeepLIIF_path to the path of DeepLIIF project. Set input_path to the directory containing the input images and python_run_path to the path of python run file. It saves the modalities to the input directory next to each image.

## More options?
You can find more options in the options/base_option.py and option/train_options.py and option/test_options.py

## Issues
Please report all issues on the public forum.

## License
© [Nadeem Lab](http://www.nadeemlab.org) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes. 

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our paper:
```
@article{ghahremani2021deepliif,
  title={DeepLIIF: Deep Learning-Inferred Multiplex ImmunoFluorescence for IHC Quantification},
  author={Ghahremani, Parmida and Li, Yanyun and Kaufman, Arie and Vanguri, Rami and Greenwald, Noah and Angelo, Michael and Hollmann, Travis J and Nadeem, Saad},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
