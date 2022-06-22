# Installation

## Prerequisites
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
  prepare-training-data  Preparing data for training
  serialize              Serialize DeepLIIF models using Torchscript
  test                   Test trained models
  train                  General-purpose training script for multi-task...
```