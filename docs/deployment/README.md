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

## Dask deployment

By default, DeepLIIF networks are deployed using a combination of TorchScript and Dask. Torchscript is used to
serialize and optimize the models starting from a pre-trained model checkpoint and the Python code that
describes the models. For more details check out the [Serialize Model](https://nadeemlab.github.io/DeepLIIF/testing/#serialize-model)
section on the documentation.

Models parallelization and interdependencies are expressed using Dask [Delayed](https://docs.dask.org/en/stable/delayed.html)
functions that allow us to build a computational graph with minimal code annotations. The concrete implementation
can be found on the `run_dask()` function under the `deepliif.models` module.

## Torchserve deployment

This section describes how to run DeepLIIF's inference using [Torchserve](https://github.com/pytorch/serve) workflows.
Workflows con be composed by both PyTorch models and Python functions that can be connected through a DAG.
For DeepLIIF there are 4 main stages (see Figure 3): 

* `Pre-process` deserialize the image from the request and return a tensor created from it.
* `G1-4` run the ResNets to generate the Hematoxylin, DAPI, LAP2 and Ki67 masks.
* `G51-5` run the UNets and apply `Weighted Average` to generate the Segmentation image.
* `Aggregate` aggregate and serialize the results and return to user.

![DeepLIIF Torchserve workflow](./images/deepliif_torchserve_workflow.png)
*Composition of DeepLIIF nets into a Torchserve workflow.*

In practice, users need to call this workflow for each tile generated from the original image.  
A common use case scenario would be:

1. Load an IHC image and generate the tiles.
2. For each tile:
    1. Resize to 512x512 and transform to tensor.
    2. Serialize the tensor and use the inference API to generate all the masks.
    3. Deserialize the results.
3. Stitch back the results and apply post-processing operations.

The next sections show how to deploy the model server.

### Prerequisites

1\. Install Torchserve and torch-model-archiver following [these instructions](https://github.com/pytorch/serve#install-torchserve-and-torch-model-archiver).
In MacOS, navigate to the `model-server` directory:

```shell
cd model-server
python3 -m venv venv
source venv/bin/activate
pip install torch torchserve torch-model-archiver torch-workflow-archiver 
```

2\. Download and unzip the latest version of the DeepLIIF models from [zenodo](https://zenodo.org/record/4751737#.YXsTuS2cZhF).

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