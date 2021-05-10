### DeepLIIF: Deep-Learning Inferred Multiplex Immunofluoresence for IHC Quantification

![picture alt](./images/overview.png "Title is optional")

Reporting biomarkers assessed by routine immunohistochemical (IHC) staining of tissue is broadly used in diagnostic pathology laboratories for patient care. To date, clinical reporting is predominantly qualitative or semi-quantitative. By creating a multitask deep learning framework referred to as DeepLIIF, we are presenting a single step solution to nuclear segmentation and quantitative single-cell IHC scoring. Leveraging a unique de novo dataset of co-registered IHC and multiplex immunoflourescence (mpIF) data generated from the same tissue section, we simultaneously segment and translate low-cost and prevalent IHC slides to more expensive-yet-informative mpIF images. Moreover, a nuclear-pore marker, LAP2beta, is co-registered to improve cell segmentation and protein expression quantification on IHC slides. By formulating the IHC quantification as cell instance segmentation/classification rather than cell detection problem, we show that our model trained on clean IHC Ki67 data can generalize to more noisy and artifact-ridden images as well as other nuclear and non-nuclear markers such as CD3, CD8, BCL2, BCL6, MYC, MUM1, CD10 and TP53. We thoroughly evaluate our method on publicly available bench-mark datasets as well as against pathologists’ semi-quantitative scoring.

© This code is made available for non-commercial academic purposes.

## Prerequisites:
```
NVIDIA GPU (Tested on NVIDIA QUADRO RTX 6000)
CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification)
Python 3
Pytorch>=0.4.0
torchvision>=0.2.1
dominate>=2.3.1
visdom>=0.1.8.3
```

## Dataset:
All image pairs must be 512x512 and paired together in 3072x512 images (6 images of size 512x512 stitched together horizontally). Data needs to be arranged in the following order:
```
XXX_Dataset 
    ├── test
    ├── val
    └── train
```

## Pre-processing:
Using this function, you can prepare images for testing purposes. This function breaks large images into tiles of size 512x512 and create a 3072x512 image for each tile. Then, stores the generated tiles in the given directory.
```
python preprocessing.py **--input_dir** /path/to/input/images **--output_dir** /path/to/output/images **--tile_size** size_of_each_cropped_tile **--overlap_size** overlap_size_between_crops **--resize_self** if_True_resizes_to_the_closest_rectangle_if_False_resize_size_need_to_be_set **--resize_size**
```

## Post-processing:
This function is used for testing purposes. This function can be used to post-process the results of the model. It saves the generated images with the proper postfixes in the given directory. If the input images are the tiles from a larger image, it also stitches them to create the inferred images for the original images. 
```
python postprocessing.py **--input_dir** /path/to/preprocessed/images **--output_dir** /path/to/output/images **--input_orig_dir** /path/to/original/input/images --tile_size size_of_each_cropped_tile **--overlap_size** overlap_size_between_crops **--resize_self** if_True_resizes_to_the_closest_rectangle_if_False_resize_size_need_to_be_set **--resize_size**
```

## Training:
To train a model:
```
python train.py **--dataroot** /path/to/input/images **--name** Model_Name **--model** DeepLIIF **--netG** resnet_9blocks 
```
* To view training losses and results, open the URL http://localhost:8097. For cloud servers replace localhost with your IP.
* To epoch-wise intermediate training results, /DeepLIIF/checkpoints/Model_Name/web/index.html
* Trained models will be by default save in /DeepLIIF/checkpoints/Model_Name.

## Testing:
To test the model:
```
python test.py **--dataroot** /path/to/input/images **--name** Model_Name **--model** DeepLIIF **--netG** resnet_9blocks 
```
* The test results will be by default saved to /DeepLIIF/results/Model_Name/test_latest/images.
* Pretrained models can be downloaded here. Place the pretrained model in /DeepLIIF/checkpoints/deepLIIF_model and set the Model_Name as deepLIIF_model.

## Demo:
Change the DeepLIIF_path to the path of DeepLIIF project. Set input_path to the directory containing the input images and python_run_path to the path of python run file. It saves the modalities to the input directory next to each image.

## More options?
You can find more options in:
* **/DeepLIIF/options/base_option.py** for basic options for training and testing purposes. 
* **/DeepLIIF/options/train_options.py** for advanced training options.
* **/DeepLIIF/options/test_options.py** for advanced testing options.
* **/DeepLIIF/options/processing_options.py** for advanced pre/post-processing options.

## Issues
Please report all issues on the public forum.

## License
© This code is made available under the GPLv3 License and is available for non-commercial academic purposes. 

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
