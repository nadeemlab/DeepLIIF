![log_image](./images/DeepLIIF_logo.png)


### DeepLIIF: Deep-Learning Inferred Multiplex Immunofluorescence for IHC Image Quantification

[Read Link](https://www.biorxiv.org/content/10.1101/2021.05.01.442219v3) | [Google CoLab Demo](https://colab.research.google.com/drive/12zFfL7rDAtXfzBwArh9hb0jvA38L_ODK?usp=sharing) | [Docker](#docker-file)

*Reporting biomarkers assessed by routine immunohistochemical (IHC) staining of tissue is broadly used in diagnostic pathology laboratories for patient care. To date, clinical reporting is predominantly qualitative or semi-quantitative. By creating a multitask deep learning framework referred to as DeepLIIF, we present a single-step solution to stain deconvolution/separation, cell segmentation, and quantitative single-cell IHC scoring. Leveraging a unique de novo dataset of co-registered IHC and multiplex immunofluorescence (mpIF) staining of the same slides, we segment and translate low-cost and prevalent IHC slides to more expensive-yet-informative mpIF images, while simultaneously providing the essential ground truth for the superimposed brightfield IHC channels. Moreover, a new nuclear-envelop stain, LAP2beta, with high (>95%) cell coverage is introduced to improve cell delineation/segmentation and protein expression quantification on IHC slides. By simultaneously translating input IHC images to clean/separated mpIF channels and performing cell segmentation/classification, we show that our model trained on clean IHC Ki67 data can generalize to more noisy and artifact-ridden images as well as other nuclear and non-nuclear markers such as CD3, CD8, BCL2, BCL6, MYC, MUM1, CD10, and TP53. We thoroughly evaluate our method on publicly available benchmark datasets as well as against pathologists' semi-quantitative scoring.*

© This code is made available for non-commercial academic purposes.

![overview_image](./images/overview.png)**Figure1**. *Overview of DeepLIIF pipeline and sample input IHCs (different brown/DAB markers -- BCL2, BCL6, CD10, CD3/CD8, Ki67) with corresponding DeepLIIF-generated hematoxylin/mpIF modalities and classified (positive (red) and negative (blue) cell) segmentation masks. (a) Overview of DeepLIIF. Given an IHC input, our multitask deep learning framework simultaneously infers corresponding Hematoxylin channel, mpIF DAPI, mpIF protein expression (Ki67, CD3, CD8, etc.), and the positive/negative protein cell segmentation, baking explainability and interpretability into the model itself rather than relying on coarse activation/attention maps. In the segmentation mask, the red cells denote cells with positive protein expression (brown/DAB cells in the input IHC), whereas blue cells represent negative cells (blue cells in the input IHC). (b) Example DeepLIIF-generated hematoxylin/mpIF modalities and segmentation masks for different IHC markers. DeepLIIF, trained on clean IHC Ki67 nuclear marker images, can generalize to noisier as well as other IHC nuclear/cytoplasmic marker images.*

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

## Synthetic Data Generation:
The DeepLIIF_Model_V1 suffered from its inability to separate IHC positive cells in some large clusters, resulting from the absence of clustered positive cells in our training data. To infuse more information about the clustered positive cells into our model, we present a novel approach for the synthetic generation of IHC images using co-registered data. 
We design a GAN-based model that receives the Hematoxylin channel, the mpIF DAPI image, and the segmentation mask and generates the corresponding IHC image. The model converts the Hematoxylin channel to gray-scale to infer more helpful information such as the texture and discard unnecessary information such as color. The Hematoxylin image guides the network to synthesize the background of the IHC image by preserving the shape and texture of the cells and artifacts in the background. The DAPI image assists the network in identifying the location, shape, and texture of the cells to better isolate the cells from the background. The segmentation mask helps the network specify the color of cells based on the type of the cell (positive cell: a brown hue, negative: a blue hue).

In the next step, we generate synthetic IHC images with more clustered positive cells. To do so, we change the segmentation mask by choosing a percentage of random negative cells in the segmentation mask (called as Neg-to-Pos) and converting them into positive cells. Some samples of the synthesized IHC images along with the original IHC image are shown in Figure 2.

![IHC_Gen_image](./images/IHC_Gen3.png)**Figure2**. *Overview of synthetic IHC image generation. (a) A training sample of the IHC-generator model. (b) Some samples of synthesized IHC images using the trained IHC-Generator model. The Neg-to-Pos shows the percentage of the negative cells in the segmentation mask converted to positive cells.*

We created a new dataset using the original IHC images and synthetic IHC images. We synthesize each image in the dataset two times by setting the Neg-to-Pos parameter to %50 and %70. We re-trained our network with the new dataset. You can find the new trained model [here](https://zenodo.org/record/4751737/files/DeepLIIF_Latest_Model.zip?download=1).

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
* Training datasets can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

## Testing:
To test the model:
```
python test.py --dataroot /path/to/input/images 
               --name Model_Name 
               --model DeepLIIF
```
* The test results will be by default saved to DeepLIIF/results/Model_Name/test_latest/images.
* Pretrained models can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4). **You can find two pre-trained models: (1) DeepLIIF_Model_V1: This is the one reported in the paper, and (2) DeepLIIF_Model_V2: This is the improved version which includes the BCDataset and the new generated synthetic IHC data (mentioned above) in training.**
* Place the pretrained model in DeepLIIF/checkpoints/DeepLIIF_Model and set the Model_Name as DeepLIIF_Model.
* To test the model on large tissues, we have provided two scripts for pre-processing (breaking tissue into smaller tiles) and post-processing (stitching the tiles to create the corresponding inferred images to the original tissue). A brief tutorial on how to use these scripts is given.
* Testing datasets can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

## Pre-processing Large Tissues:
Using this script, you can prepare large tissue for testing purposes. It breaks large images into tiles of size 512x512 and create a 3072x512 image for each tile. Then, stores the generated tiles in the given directory.
```
python preprocessing.py --input_dir /path/to/input/images 
                        --output_dir /path/to/output/images 
                        --tile_size size_of_each_cropped_tile 
                        --overlap_size overlap_size_between_crops 
                        --resize_self if_True_resizes_to_the_closest_rectangle_dividable_by_tile_size_if_False_resize_size_need_to_be_set 
                        --resize_size
```

## Post-processing Large Tissues:
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

## Post-processing Segmentation Mask:
Using the post-processing script, you can overlay the classified boundaries around the cells over the original IHC image. It draws a blue boundary around negative cells and a red boundary around positive cells. It also refines the generated segmentation mask by removing noise. It saves the overlaid image and the refined mask in the same directory.
```
python PostProcessSegmentationMask.py /path/to/output/images/generated/by/model/
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

## Google CoLab:
If you don't have access to GPU or appropriate hardware, we have also created [Google CoLab project](https://colab.research.google.com/drive/12zFfL7rDAtXfzBwArh9hb0jvA38L_ODK?usp=sharing) for your convenience. 
Please follow the steps in the provided notebook to install the requirements and run the training and testing scripts.
All the libraries and pretrained models have already been set up there. 
The user can directly run DeepLIIF on their images using the instructions given in the Google CoLab project. 

## Registration:
To register the denovo stained mpIF and IHC images, you can use the registration framework in the 'Registration' directory. Please refer to the README file provided in the same directory for more details.

## More options?
You can find more options in:
* **DeepLIIF/options/base_option.py** for basic options for training and testing purposes. 
* **DeepLIIF/options/train_options.py** for advanced training options.
* **DeepLIIF/options/test_options.py** for advanced testing options.
* **DeepLIIF/options/processing_options.py** for advanced pre/post-processing options.
* **DeepLIIF/options/Registration_App.py** for registering pathology crops and slides.

## Issues
Please report all issues on the public forum.

## License
© [Nadeem Lab](https://nadeemlab.org/) - DeepLIIF code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 

## Acknowledgments
* This code is inspired by [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Reference
If you find our work useful in your research or if you use parts of this code, please cite our paper:
```
@article{ghahremani2021deepliif,
  title={DeepLIIF: Deep Learning-Inferred Multiplex ImmunoFluorescence for IHC Image Quantification},
  author={Ghahremani, Parmida and Li, Yanyun and Kaufman, Arie and Vanguri, Rami and Greenwald, Noah and Angelo, Michael and Hollmann, Travis J and Nadeem, Saad},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
