# Prerequisites
```
NVIDIA GPU (Tested on NVIDIA QUADRO RTX 6000)
CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification)
Python 3
Pytorch>=0.4.0
torchvision>=0.2.1
dominate>=2.3.1
visdom>=0.1.8.3
```

# Dataset
All image pairs must be 512x512 and paired together in 3072x512 images (6 images of size 512x512 stitched together horizontally). 
For testing purpose, for any unavailable image in the pairing set, just put the original IHC image or a blank image in the final paired image in place of the missing image in the pair (example images are provided in the Dataset folder). For testing, you can skip stitching the images and only put the original IHC image in the test folder, but make sure to set the --dataset_mode to 'single'. 
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

# Training:
To train a model:
```
python train.py --dataroot /path/to/input/images 
                --name Model_Name 
                --model DeepLIIF
```
* To view training losses and results, open the URL http://localhost:8097. For cloud servers replace localhost with your IP.
* To epoch-wise intermediate training results, DeepLIIF/checkpoints/Model_Name/web/index.html
* Trained models will be by default save in DeepLIIF/checkpoints/Model_Name.
* Training datasets can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

# Testing:
To test the model:
```
python test.py --dataroot /path/to/input/images 
               --name Model_Name 
               --model DeepLIIF
```
* The test results will be by default saved to DeepLIIF/results/Model_Name/test_latest/images.
* The latest version of the pretrained models can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).
* Place the pretrained model in DeepLIIF/checkpoints/DeepLIIF_Latest_Model and set the Model_Name as DeepLIIF_Latest_Model.
* To test the model on large tissues, we have provided two scripts for pre-processing (breaking tissue into smaller tiles) and post-processing (stitching the tiles to create the corresponding inferred images to the original tissue). A brief tutorial on how to use these scripts is given.
* Testing datasets can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

# Demo:
Change the DeepLIIF_path to the path of DeepLIIF project. 
Set input_path to the directory containing the input images and python_run_path to the python executable (path to installed python directory). 
It saves the modalities to the input directory next to each image.

# Docker File:
You can use the docker file to create the docker image for running the model.
First, you need to install the [Docker Engine](https://docs.docker.com/engine/install/ubuntu/).
After installing the Docker, you need to follow these steps:
* Download the pretrained model and place them in DeepLIIF/checkpoints/DeepLIIF_Latest_Model.
* Change XXX of the **WORKDIR** line in the **DockerFile** to the directory containing the DeepLIIF project. 
* To create a docker image from the docker file:
```
docker build -t deepliif_image .
```
The image is then used as a base. You can copy and use it to run an application. The application needs an isolated environment in which to run, referred to as a container.
* To create and run a container:
```
docker run --gpus all --name deepliif_container -it deepliif_image
```
When you run a container from the image, a conda environment is activated in the specified WORKDIR.
You can easily run the preprocessing.py, postprocessing.py, train.py and test.py in the activated environment and copy the results from the docker container to the host.
* A quick sample of testing the pre-trained model on the sample images:
```
python preprocessing.py --input_dir Sample_Large_Tissues/ --output_dir Sample_Large_Tissues_Tiled/test/
python test.py --dataroot Sample_Large_Tissues_Tiled/ --name DeepLIIF_Latest_Model --model DeepLIIF
python postprocessing.py --output_dir Sample_Large_Tissues/ --input_dir results/DeepLIIF_Latest_Model/test_latest/images/ --input_orig_dir Sample_Large_Tissues/
```

# Google CoLab:
If you don't have access to GPU or appropriate hardware, we have also created [Google CoLab project](https://colab.research.google.com/drive/12zFfL7rDAtXfzBwArh9hb0jvA38L_ODK?usp=sharing) for your convenience. 
Please follow the steps in the provided notebook to install the requirements and run the training and testing scripts.
All the libraries and pretrained models have already been set up there. 
The user can directly run DeepLIIF on their images using the instructions given in the Google CoLab project. 

# Registration:
To register the denovo stained mpIF and IHC images, you can use the registration framework in the 'Registration' directory. Please refer to the README file provided in the same directory for more details.

# Contributing Training Data:
To train DeepLIIF, we used a dataset of lung and bladder tissues containing IHC, hematoxylin, mpIF DAPI, mpIF Lap2, and mpIF Ki67 of the same tissue scanned using ZEISS Axioscan. These images were scaled and co-registered with the fixed IHC images using affine transformations, resulting in 1667 co-registered sets of IHC and corresponding multiplex images of size 512x512. We randomly selected 709 sets for training, 358 sets for validation, and 600 sets for testing the model. We also randomly selected and segmented 41 images of size 640x640 from recently released [BCDataset](https://sites.google.com/view/bcdataset) which contains Ki67 stained sections of breast carcinoma with Ki67+ and Ki67- cell centroid annotations (for cell detection rather than cell instance segmentation task). We split these tiles into 164 images of size 512x512; the test set varies widely in the density of tumor cells and the Ki67 index. You can find this dataset [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

We are also creating a self-configurable version of DeepLIIF which will take as input any co-registered H&E/IHC and multiplex images and produce the optimal output. If you are generating or have generated H&E/IHC and multiplex staining for the same slide (denovo staining) and would like to contribute that data for DeepLIIF, we can perform co-registration, whole-cell multiplex segmentation, train the DeepLIIF model and release back to the community with full credit to the contributors.

# More options?
You can find more options in:
* **DeepLIIF/options/base_option.py** for basic options for training and testing purposes. 
* **DeepLIIF/options/train_options.py** for advanced training options.
* **DeepLIIF/options/test_options.py** for advanced testing options.
* **DeepLIIF/options/processing_options.py** for advanced pre/post-processing options.
* **DeepLIIF/options/Registration_App.py** for registering pathology crops and slides.
