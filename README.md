### DeepLIIF: Deep-Learning Inferred Multiplex Immunofluoresence for IHC Quantification

Reporting biomarkers assessed by routine immunohistochemical (IHC) staining of tissue is broadly used in diagnostic pathology laboratories for patient care. To date, clinical reporting is predominantly qualitative or semi-quantitative. By creating a multitask deep learning framework referred to as DeepLIIF, we are presenting a single step solution to nuclear segmentation and quantitative single-cell IHC scoring. Leveraging a unique de novo dataset of co-registered IHC and multiplex immunoflourescence (mpIF) data generated from the same tissue section, we simultaneously segment and translate low-cost and prevalent IHC slides to more expensive-yet-informative mpIF images. Moreover, a nuclear-pore marker, LAP2beta, is co-registered to improve cell segmentation and protein expression quantification on IHC slides. By formulating the IHC quantification as cell instance segmentation/classification rather than cell detection problem, we show that our model trained on clean IHC Ki67 data can generalize to more noisy and artifact-ridden images as well as other nuclear and non-nuclear markers such as CD3, CD8, BCL2, BCL6, MYC, MUM1, CD10 and TP53. We thoroughly evaluate our method on publicly available bench-mark datasets as well as against pathologists’ semi-quantitative scoring.

© This code is made available for non-commercial academic purposes.

![overview_image](./images/overview.png "Title is optional")
<figcaption></figcaption>

## Prerequisites:
```del
NVIDIA GPU (Tested on NVIDIA QUADRO RTX 6000)
CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification)
Python 3
Pytorch>=0.4.0
torchvision>=0.2.1
dominate>=2.3.1
visdom>=0.1.8.3
```

## Dataset:
All image pairs must be 512x512 and paired together in 3072x512 images (6 images of size 512x512 stitched together horizontally). 
For testing purpose, for any unavailable image in the pairing set, just put the original IHC image or a blank image in the final paired image in place of the missing image in the pair (example images are provided in the Dataset folder). 
Data needs to be arranged in the following order:
```
XXX_Dataset 
    ├── test
    ├── val
    └── train
```
We have provided two simple functions in the DeepLIIF/PrepareDataset.py file for preparing data for testing and training purposes.
* **To prepare data for training**, you need to have the paired data including IHC, Hematoxylin Channel, mpIF DAPI, mpIF Lap2, mpIF marker, and segmentation mask in the input directory.
The script gets the address of directory containing paired data and the address of the dataset directory.
It, first, creates the train and validation directories inside the given dataset directory.
Then it reads all images in the folder and saves the pairs in the train or validation directory, based on the given validation_ratio.
```
python PrepareDataForTraining.py --input_dir /path/to/input/images
                                 --output_dir /path/to/dataset/directory
                                 --validation_ratio The ratio of the number of the images in the validation set to the total number of images.
```

* **To prepare data for testing**, you only need to have IHC images in the input directory.
The function gets the address of directory containing the IHC data and the address of the dataset directory.
It, first, creates the test directory inside the given dataset directory.
Then it reads the IHC images in the folder and saves a pair in the test directory.
```
python PrepareDataForTesting.py --input_dir /path/to/input/images
                                --output_dir /path/to/dataset/directory
```

## Training:
To train a model:
```
python train.py --dataroot /path/to/input/images 
                --name Model_Name 
                --model DeepLIIF
```
* To view training losses and results, open the URL http://localhost:8097. For cloud servers replace localhost with your IP.
* To epoch-wise intermediate training results, DeepLIIF/checkpoints/Model_Name/web/index.html
* Trained models will be by default save in DeepLIIF/checkpoints/Model_Name.

## Testing:
To test the model:
```
python test.py --dataroot /path/to/input/images 
               --name Model_Name 
               --model DeepLIIF
```
* The test results will be by default saved to DeepLIIF/results/Model_Name/test_latest/images.
* Pretrained models can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4). Place the pretrained model in DeepLIIF/checkpoints/DeepLIIF_Model and set the Model_Name as DeepLIIF_Model.
* To test the model on large tissues, we have provided two scripts for pre-processing (breaking tissue into smaller tiles) and post-processing (stitching the tiles to create the corresponding inferred images to the original tissue). A brief tutorial on how to use these scripts is given.


## Pre-processing:
Using this script, you can prepare large tissue for testing purposes. It breaks large images into tiles of size 512x512 and create a 3072x512 image for each tile. Then, stores the generated tiles in the given directory.
```
python preprocessing.py --input_dir /path/to/input/images 
                        --output_dir /path/to/output/images 
                        --tile_size size_of_each_cropped_tile 
                        --overlap_size overlap_size_between_crops 
                        --resize_self if_True_resizes_to_the_closest_rectangle_dividable_by_tile_size_if_False_resize_size_need_to_be_set 
                        --resize_size
```

## Post-processing:
This script is used for testing purposes. It can be used to post-process the results of the model tested on the images tiled by pre-processing script. 
It stitches the generated tiles with the given overlap_size to create the inferred images for the original images and saves the generated images with the proper postfixes in the given directory.
```
python postprocessing.py --input_dir /path/to/preprocessed/images 
                         --output_dir /path/to/output/images 
                         --input_orig_dir /path/to/original/input/images 
                         --tile_size size_of_each_cropped_tile 
                         --overlap_size overlap_size_between_crops 
                         --resize_self if_True_resizes_to_the_closest_rectangle_dividable_by_tile_size_if_False_resize_size_need_to_be_set 
                         --resize_size resizing_size
```

## Demo:
Change the DeepLIIF_path to the path of DeepLIIF project. 
Set input_path to the directory containing the input images and python_run_path to the python executable (path to installed python directory). 
It saves the modalities to the input directory next to each image.

## Docker File:
You can use the docker file to create the docker image for running the model.
First, you need to install the [Docker Engine](https://docs.docker.com/engine/install/ubuntu/).
After installing the Docker, you need to follow these steps:
* Download the pretrained model and place them in DeepLIIF/checkpoints/DeepLIIF_Model.
* Change XXX of the **WORKDIR** line in the **DockerFile** to the directory containing the DeepLIIF project. 
* To create a docker image from the docker file:
```
docker build -t DeepLIIF_Image .
```
The image is then used as a base. You can copy and use it to run an application. The application needs an isolated environment in which to run, referred to as a container.
* To create and run a container:
```
docker run --gpus all --name DeepLIIF_Container -it DeepLIIF_Image
```
When you run a container from the image, a conda environment is activated in the specified WORKDIR.
You can easily run the preprocessing.py, postprocessing.py, train.py and test.py in the activated environment and copy the results from the docker container to the host.
* A quick sample of testing the pre-trained model on the sample images:
```
python preprocessing.py --input_dir Sample_Data/ --output_dir Sample_Data_Tiled/test/
python test.py --dataroot Sample_Data_Tiled/ --name DeepLIIF_Model --model DeepLIIF
python postprocessing.py --input_dir Sample_Data/ --output_dir Sample_Data_Tiled/test/ --input_orig_dir Sample_Data/
```

## More options?
You can find more options in:
* **DeepLIIF/options/base_option.py** for basic options for training and testing purposes. 
* **DeepLIIF/options/train_options.py** for advanced training options.
* **DeepLIIF/options/test_options.py** for advanced testing options.
* **DeepLIIF/options/processing_options.py** for advanced pre/post-processing options.

## Issues
Please report all issues on the public forum.

## License
© DeepLIIF code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 

## Acknowledgments
* This code is inspired by [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

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
