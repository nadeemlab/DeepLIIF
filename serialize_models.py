# Serialize DeepLIIF models to Torchscript
import torch
from PIL import Image

from deepliif.models import init_nets
from deepliif.preprocessing import transform


def trace_nets():
    models_dir = './model-server/DeepLIIF_Latest_Model'

    with Image.open('./Sample_Large_Tissues/ROI_7.png') as img:
        sample = transform(img.resize((512, 512)))

    for name, net in init_nets(models_dir).items():
        print(f'Serializing {name}')
        traced_net = torch.jit.trace(net, sample)
        traced_net.save(f'{models_dir}/{name}.pt')


if __name__ == '__main__':
    trace_nets()
