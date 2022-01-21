<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="./images/DeepLIIF_logo.png" width="50%">
    <h3 align="center"><strong>Deep-Learning Inferred Multiplex Immunofluorescence for IHC Image Quantification</strong></h3>
    <p align="center">
    <a href="https://doi.org/10.1101/2021.05.01.442219">Read Link</a>
    |
    <a href="https://deepliif.org/">AWS Cloud Deployment</a>
    |
    <a href="https://colab.research.google.com/drive/12zFfL7rDAtXfzBwArh9hb0jvA38L_ODK?usp=sharing">Google CoLab Demo</a>
    |
    <a href="#docker-file">Docker</a>
    |
    <a href="https://github.com/nadeemlab/DeepLIIF/issues">Report Bug</a>
    |
    <a href="https://github.com/nadeemlab/DeepLIIF/issues">Request Feature</a>
  </p>
</p>

*Reporting biomarkers assessed by routine immunohistochemical (IHC) staining of tissue is broadly used in diagnostic 
pathology laboratories for patient care. To date, clinical reporting is predominantly qualitative or semi-quantitative. 
By creating a multitask deep learning framework referred to as DeepLIIF, we present a single-step solution to stain 
deconvolution/separation, cell segmentation, and quantitative single-cell IHC scoring. Leveraging a unique de novo 
dataset of co-registered IHC and multiplex immunofluorescence (mpIF) staining of the same slides, we segment and 
translate low-cost and prevalent IHC slides to more expensive-yet-informative mpIF images, while simultaneously 
providing the essential ground truth for the superimposed brightfield IHC channels. Moreover, a new nuclear-envelop 
stain, LAP2beta, with high (>95%) cell coverage is introduced to improve cell delineation/segmentation and protein 
expression quantification on IHC slides. By simultaneously translating input IHC images to clean/separated mpIF channels 
and performing cell segmentation/classification, we show that our model trained on clean IHC Ki67 data can generalize to 
more noisy and artifact-ridden images as well as other nuclear and non-nuclear markers such as CD3, CD8, BCL2, BCL6, 
MYC, MUM1, CD10, and TP53. We thoroughly evaluate our method on publicly available benchmark datasets as well as against 
pathologists' semi-quantitative scoring.*

© This code is made available for non-commercial academic purposes.

![overview_image](./images/overview.png)**Figure 1**. *Overview of DeepLIIF pipeline and sample input IHCs (different 
brown/DAB markers -- BCL2, BCL6, CD10, CD3/CD8, Ki67) with corresponding DeepLIIF-generated hematoxylin/mpIF modalities 
and classified (positive (red) and negative (blue) cell) segmentation masks. (a) Overview of DeepLIIF. Given an IHC 
input, our multitask deep learning framework simultaneously infers corresponding Hematoxylin channel, mpIF DAPI, mpIF 
protein expression (Ki67, CD3, CD8, etc.), and the positive/negative protein cell segmentation, baking explainability 
and interpretability into the model itself rather than relying on coarse activation/attention maps. In the segmentation 
mask, the red cells denote cells with positive protein expression (brown/DAB cells in the input IHC), whereas blue cells 
represent negative cells (blue cells in the input IHC). (b) Example DeepLIIF-generated hematoxylin/mpIF modalities and 
segmentation masks for different IHC markers. DeepLIIF, trained on clean IHC Ki67 nuclear marker images, can generalize 
to noisier as well as other IHC nuclear/cytoplasmic marker images.*

## Pre-requisites
1. Python 3.8
2. Docker

## Installing `deepliif`

DeepLIIF can be `pip` installed:
```shell
$ python3.8 -m venv venv
$ source venv/bin/activate
(venv) $ pip install git+https://github.com/nadeemlab/DeepLIIF.git
```

The package is composed of two parts:
1. A library that implements the core functions used to train and test DeepLIIF models. 
2. A CLI to run common batch operations including training, batch testing and Torchscipt models serialization.

You can list all available commands:

```
(venv) $ deepliif --help
Usage: deepliif [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  prepare-testing-data   Preparing data for testing
  prepare-training-data  Preparing data for training
  serialize              Serialize DeepLIIF models using Torchscript
  test                   Test trained models
  train                  General-purpose training script for multi-task...
```

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


## Docker
We provide a Dockerfile that can be used to run the DeepLIIF models inside a container.
First, you need to install the [Docker Engine](https://docs.docker.com/engine/install/ubuntu/).
After installing the Docker, you need to follow these steps:
* Download the pretrained model and place them in DeepLIIF/checkpoints/DeepLIIF_Latest_Model.
* Change XXX of the **WORKDIR** line in the **DockerFile** to the directory containing the DeepLIIF project. 
* To create a docker image from the docker file:
```
docker build -t cuda/deepliif .
```
The image is then used as a base. You can copy and use it to run an application. The application needs an isolated 
environment in which to run, referred to as a container.
* To create and run a container:
```
 docker run -it -v `pwd`:`pwd` -w `pwd` cuda/deepliif deepliif test --input-dir Sample_Large_Tissues
```
When you run a container from the image, the `deepliif` CLI will be available.
You can easily run any CLI command in the activated environment and copy the results from the docker container to the host.

## Google CoLab:
If you don't have access to GPU or appropriate hardware, we have also created [Google CoLab project](https://colab.research.google.com/drive/12zFfL7rDAtXfzBwArh9hb0jvA38L_ODK?usp=sharing) 
for your convenience. 
Please follow the steps in the provided notebook to install the requirements and run the training and testing scripts.
All the libraries and pretrained models have already been set up there. 
The user can directly run DeepLIIF on their images using the instructions given in the Google CoLab project. 

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


## Deploying DeepLIIF with Torchserve

This section describes how to run DeepLIIF's inference using [Torchserve](https://github.com/pytorch/serve) workflows.
Workflows con be composed by both PyTorch models and Python functions that can be connected through a DAG.
For DeepLIIF there are 4 main stages (see Figure 3): 
* `Pre-process` deserialize the image from the request and return a tensor created from it.
* `G1-4` run the ResNets to generate the Hematoxylin, DAPI, LAP2 and Ki67 masks.
* `G51-5` run the UNets and apply `Weighted Average` to generate the Segmentation image.
* `Aggregate` aggregate and serialize the results and return to user.

![DeepLIIF Torchserve workflow](./images/deepliif_torchserve_workflow.png)
**Figure 3**. *Composition of DeepLIIF nets into a Torchserve workflow*

In practice, users need to call this workflow for each tile generated from the original image.  
A common use case scenario would be:
1. Load an IHC image and generate the tiles.
3. For each tile
    1. Resize to 512x512 and transform to tensor.
    2. Serialize the tensor and use the inference API to generate all the masks
    3. Deserialize the results
4. Stitch back the results and apply post-processing operations 

The next sections show how to deploy the model server.

### Pre-requisites
1. Install Torchserve and torch-model-archiver following [these instructions](https://github.com/pytorch/serve#install-torchserve-and-torch-model-archiver).
In MacOS, navigate to the `model-server` directory:

```shell
cd model-server
python3 -m venv venv
source venv/bin/activate
pip install torch torchserve torch-model-archiver torch-workflow-archiver 
```

2. Download and unzip the latest version of the DeepLIIF models from [zenodo](https://zenodo.org/record/4751737#.YXsTuS2cZhF).

```shell
wget https://zenodo.org/record/4751737/files/DeepLIIF_Latest_Model.zip
unzip DeepLIIF_Latest_Model.zip
```

### Package models and workflow
In order to run the DeepLIIF nets using Torchserve, they first need to be archived as MAR files.
In this section we will create the model artifacts and archive them in the model store.
First, inside `model-server` create a directory to store the models.

```shell
mkdir model-store
```

For every ResNet (`G1`, `G2`, `G3`, `G4`) run replacing the name of the net: 

```shell
torch-model-archiver --force --model-name <Gx> \
    --model-file resnet.py \
    --serialized-file ./DeepLIIF_Latest_Model/latest_net_<Gx>.pth \
    --export-path model-store \
    --handler net_handler.py \
    --requirements-file model_requirements.txt
```

and for the UNets (`G51`, `G52`, `G53`, `G54`, `G54`) switch the model file from `resnet.py` to `unet.py`:

```shell
torch-model-archiver --force --model-name <G5x> \
    --model-file unet.py \
    --serialized-file ./DeepLIIF_Latest_Model/latest_net_<G5x>.pth \
    --export-path model-store \
    --handler net_handler.py \
    --requirements-file model_requirements.txt
```

Once all the models have been packaged and made available in the model store,
they can be composed into a workflow archive. 
Finally, create the archive for the workflow represented in Figure 3.

```shell
torch-workflow-archiver -f --workflow-name deepliif \
    --spec-file deepliif_workflow.yaml \
    --handler deepliif_workflow_handler.py \
    --export-path model-store
```

### Run the server
Once all artifacts are available in the model store, run the model server.

```shell
torchserve --start --ncs \
    --model-store model-store \
    --workflow-store model-store  \
    --ts-config config.properties
```

An additional step is needed to register the `deepliif` workflow on the server.

```shell
curl -X POST "http://127.0.0.1:8081/workflows?url=deepliif.war"
```

### Run inference using Python

The snippet below shows an example of how to cosume the Torchserve workflow API using Python. 

```python
import base64
import requests
from io import BytesIO

import torch

from deepliif.preprocessing import transform

def deserialize_tensor(bs):
    return torch.load(BytesIO(base64.b64decode(bs.encode())))

def serialize_tensor(ts):
    buffer = BytesIO()
    torch.save(ts, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

TORCHSERVE_HOST = 'http://127.0.0.1:8080'

img = load_tile()

ts = transform(img.resize((512, 512)))

res = requests.post(
    f'{TORCHSERVE_HOST}/wfpredict/deepliif',
    json={'img': serialize_tensor(ts)}
)

res.raise_for_status()

masks = {k: deserialize_tensor(v) for k, v in res.json().items()}
```

## Issues
Please report all issues on the public forum.


## License
© [Nadeem Lab](https://nadeemlab.org/) - DeepLIIF code is distributed under **Apache 2.0 with Commons Clause** license, 
and is available for non-commercial academic purposes. 

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
