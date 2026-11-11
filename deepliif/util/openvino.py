import nncf
import torch.nn as nn

# Disable all inplace operations
# unet has ReLU(inplace=True) layers but torch.jit.trace (used internally by openvino's convert_model()) cannot correctly trace inplace operations
# after disabling inplace operations, the mismatch rate between pytorch and openvino
# dropped from 40-50% to ~0.1% and the max absolute difference from 2 to ~2e-5
def disable_inplace(module):
    for child in module.children():
        if isinstance(child, nn.ReLU):
            child.inplace = False
        disable_inplace(child)


def transform_fn(data_item):
    """
    Extract the model's input from the data item.
    The data item here is the data item that is returned from the data source per iteration.
    This function should be passed when the data item cannot be used as model's input.
    """
    return data_item['A'] # input image of size [1,3,512,512]


def ovdict_to_numpy(ovdict):
    """
    Converts OpenVINO OVDict predictions into Numpy.
    In our case, each ovdict contains just one key so we do not
    have to worry about the output tensor format.
    """
    assert len(ovdict) == 1, f'The OVDict object contains {len(ovdict)} keys. This function only handles 1-key OVDict.'
    return ovdict[0]


def ovdict_to_tensor(ovdict):
    """
    Converts OpenVINO OVDict predictions into PyTorch tensor.
    In our case, each ovdict contains just one key so we do not
    have to worry about the output tensor format.
    """
    assert len(ovdict) == 1, f'The OVDict object contains {len(ovdict)} keys. This function only handles 1-key OVDict.'
    for key, value in ovdict.items():
        return torch.from_numpy(np.array(value))
