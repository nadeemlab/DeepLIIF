import base64
from io import BytesIO

import torch


def preprocess(data, context):
    if data:
        return [torch.load(BytesIO(base64.b64decode(data[0]['body']['img'])))]


def weighted_average(data, context):
    if data:
        seg_weights = [0.25, 0.15, 0.25, 0.1, 0.25]

        def to_tensor(bs):
            return torch.load(BytesIO(bs)).to(torch.device('cuda'))

        return [torch.stack([
            torch.mul(to_tensor(data[0]['g51']), seg_weights[0]),
            torch.mul(to_tensor(data[0]['g52']), seg_weights[1]),
            torch.mul(to_tensor(data[0]['g53']), seg_weights[2]),
            torch.mul(to_tensor(data[0]['g54']), seg_weights[3]),
            torch.mul(to_tensor(data[0]['g55']), seg_weights[4]),
        ]).sum(dim=0)]


def aggregate(data, context):
    if data:
        def serialize_tensor(ts):
            return base64.b64encode(BytesIO(ts).getvalue()).decode()

        response = {
            'g1': serialize_tensor(data[0]['g1']),
            'g2': serialize_tensor(data[0]['g2']),
            'g3': serialize_tensor(data[0]['g3']),
            'g4': serialize_tensor(data[0]['g4']),
            'g5': serialize_tensor(data[0]['weighted_average']),
        }

        return [response]
