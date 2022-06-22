# Cloud Deployment
If you don't have access to GPU or appropriate hardware and don't want to install ImageJ, we have also created a [cloud-native DeepLIIF deployment](https://deepliif.org) with a user-friendly interface to upload images, visualize, interact, and download the final results.

DeepLIIF can also be accessed programmatically through an endpoint by posting a multipart-encoded request
containing the original image file:

```
POST /api/infer

Parameters

img (required)
file: image to run the models on

resolution
string: resolution used to scan the slide (10x, 20x, 40x), defaults to 20x 

pil
boolean: if true, use PIL.Image.open() to load the image, instead of python-bioformats

slim
boolean: if true, return only the segmentation result image
```

For example, in Python:

```python
import os
import json
import base64
from io import BytesIO

import requests
from PIL import Image

# Use the sample images from the main DeepLIIF repo
images_dir = './Sample_Large_Tissues'
filename = 'ROI_1.png'

res = requests.post(
    url='https://deepliif.org/api/infer',
    files={
        'img': open(f'{images_dir}/{filename}', 'rb')
    },
    # optional param that can be 10x, 20x (default) or 40x
    params={
        'resolution': '20x'
    }
)

data = res.json()

def b64_to_pil(b):
    return Image.open(BytesIO(base64.b64decode(b.encode())))

for name, img in data['images'].items():
    output_filepath = f'{images_dir}/{os.path.splitext(filename)[0]}_{name}.png'
    with open(output_filepath, 'wb') as f:
        b64_to_pil(img).save(f, format='PNG')

print(json.dumps(data['scoring'], indent=2))
```

## Auto-scaling the service

DeepLIIFs underlying infrastructure is completely defined using [Pulumi](https://www.pulumi.com) stacks.
Behind the scenes, we use containers to deploy both the web application and the API on top of an
ECS cluster with an auto-scaling group that runs on G4dns (GPU) machines.

Under stress, the system will autoscale both the compute capacity and the service availability to accommodate
the incoming requests without affecting the overall performance. The current auto-scaling policy monitors the
number of requests per target on the application load balancer.
