# Training

## Training Dataset:
For training, all image sets must be 512x512 and combined together in 3072x512 images (six images of size 512x512 stitched
together horizontally).
The data need to be arranged in the following order:
```
XXX_Dataset 
    ├── train
    └── val
```
We have provided a simple function in the CLI for preparing data for training.

* **To prepare data for training**, you need to have the image dataset for each image (including IHC, Hematoxylin Channel, mpIF DAPI, mpIF Lap2, mpIF marker, and segmentation mask) in the input directory.
Each of the six images for a single image set must have the same naming format, with only the name of the label for the type of image differing between them.  The label names must be, respectively: IHC, Hematoxylin, DAPI, Lap2, Marker, Seg.
The command takes the address of the directory containing image set data and the address of the output dataset directory.
It first creates the train and validation directories inside the given output dataset directory.
It then reads all of the images in the input directory and saves the combined image in the train or validation directory, based on the given `validation_ratio`.
```
deepliif prepare-training-data --input-dir /path/to/input/images
                               --output-dir /path/to/output/images
                               --validation-ratio 0.2
```

## Training:
To train a model:
```
deepliif train --dataroot /path/to/input/images 
                --name Model_Name 
```
* To view training losses and results, open the URL http://localhost:8097. For cloud servers replace localhost with your IP.
* Epoch-wise intermediate training results are in `DeepLIIF/checkpoints/Model_Name/web/index.html`.
* Trained models will be by default be saved in `DeepLIIF/checkpoints/Model_Name`.
* Training datasets can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

## Synthetic Data Generation:
The first version of DeepLIIF model suffered from its inability to separate IHC positive cells in some large clusters,
resulting from the absence of clustered positive cells in our training data. To infuse more information about the
clustered positive cells into our model, we present a novel approach for the synthetic generation of IHC images using
co-registered data. 
We design a GAN-based model that receives the Hematoxylin channel, the mpIF DAPI image, and the segmentation mask and
generates the corresponding IHC image. The model converts the Hematoxylin channel to gray-scale to infer more helpful
information such as the texture and discard unnecessary information such as color. The Hematoxylin image guides the
network to synthesize the background of the IHC image by preserving the shape and texture of the cells and artifacts in
the background. The DAPI image assists the network in identifying the location, shape, and texture of the cells to
better isolate the cells from the background. The segmentation mask helps the network specify the color of cells based 
on the type of the cell (positive cell: a brown hue, negative: a blue hue).

In the next step, we generate synthetic IHC images with more clustered positive cells. To do so, we change the 
segmentation mask by choosing a percentage of random negative cells in the segmentation mask (called as Neg-to-Pos) and 
converting them into positive cells. Some samples of the synthesized IHC images along with the original IHC image are 
shown in Figure 2.

![IHC_Gen_image](./images/IHC_Gen3.png)**Figure 2**. *Overview of synthetic IHC image generation. (a) A training sample 
of the IHC-generator model. (b) Some samples of synthesized IHC images using the trained IHC-Generator model. The 
Neg-to-Pos shows the percentage of the negative cells in the segmentation mask converted to positive cells.*

We created a new dataset using the original IHC images and synthetic IHC images. We synthesize each image in the dataset 
two times by setting the Neg-to-Pos parameter to %50 and %70. We re-trained our network with the new dataset. You can 
find the new trained model [here](https://zenodo.org/record/4751737/files/DeepLIIF_Latest_Model.zip?download=1).

## Registration:
To register the de novo stained mpIF and IHC images, you can use the registration framework in the 'Registration' 
directory. Please refer to the README file provided in the same directory for more details.

## Contributing Training Data:
To train DeepLIIF, we used a dataset of lung and bladder tissues containing IHC, hematoxylin, mpIF DAPI, mpIF Lap2, and 
mpIF Ki67 of the same tissue scanned using ZEISS Axioscan. These images were scaled and co-registered with the fixed IHC 
images using affine transformations, resulting in 1667 co-registered sets of IHC and corresponding multiplex images of 
size 512x512. We randomly selected 709 sets for training, 358 sets for validation, and 600 sets for testing the model. 
We also randomly selected and segmented 41 images of size 640x640 from recently released [BCDataset](https://sites.google.com/view/bcdataset) 
which contains Ki67 stained sections of breast carcinoma with Ki67+ and Ki67- cell centroid annotations (for cell 
detection rather than cell instance segmentation task). We split these tiles into 164 images of size 512x512; the test 
set varies widely in the density of tumor cells and the Ki67 index. You can find this dataset [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

We are also creating a self-configurable version of DeepLIIF which will take as input any co-registered H&E/IHC and 
multiplex images and produce the optimal output. If you are generating or have generated H&E/IHC and multiplex staining 
for the same slide (de novo staining) and would like to contribute that data for DeepLIIF, we can perform 
co-registration, whole-cell multiplex segmentation via [ImPartial](https://github.com/nadeemlab/ImPartial), train the 
DeepLIIF model and release back to the community with full credit to the contributors.

## Serialize Model
The installed `deepliif` uses Dask to perform inference on the input IHC images.
Before running the `test` command, the model files must be serialized using Torchscript.
To serialize the model files:
```
deepliif serialize --models-dir /path/to/input/model/files
                   --output-dir /path/to/output/model/files
```
* By default, the model files are expected to be located in `DeepLIIF/model-server/DeepLIIF_Latest_Model`.
* By default, the serialized files will be saved to the same directory as the input model files.

## Testing:
To test the model:
```
deepliif test --input-dir /path/to/input/images 
              --output-dir /path/to/output/images 
              --tile-size 512
```
* The latest version of the pretrained models can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).
* Before running test on images, the model files must be serialized as described above.
* The serialized model files are expected to be located in `DeepLIIF/model-server/DeepLIIF_Latest_Model`.
* The test results will be saved to the specified output directory, which defaults to the input directory.
* The default tile size is 512.
* Testing datasets can be downloaded [here](https://zenodo.org/record/4751737#.YKRTS0NKhH4).

If you prefer, it is possible to run the model using Torchserve.
Please see below for instructions on how to deploy the model with Torchserve and for an example of how to run the inference.