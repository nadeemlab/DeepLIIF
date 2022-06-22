from io import BytesIO

import torch
from ts.torch_handler.base_handler import BaseHandler


class NetHandler(BaseHandler):

    def preprocess(self, data):
        # TODO: temporarily set the model in training mode,
        # instead of eval mode until model is fixed.
        self.model.train()
        return torch.load(BytesIO(data[0]['body'])).to(self.device)

    def postprocess(self, inference_output):
        return [inference_output]
