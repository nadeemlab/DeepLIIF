#!/usr/bin/env python

from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from callbacks import Callback
from fabric_model import FabricModel
from edtcallback import EDTLoggerCallback
import torch
import os


## Define model and extract training parameters
def get_max_worker():
    import argparse
    parser = argparse.ArgumentParser(description='EDT Example')
    parser.add_argument('--numWorker', type=int, default='16', help='input the max number ')
    parser.add_argument('--gpuPerWorker', type=int, default='1', help='input the path of initial weight file')
    args, _ = parser.parse_known_args()
    num_worker = args.numWorker * args.gpuPerWorker
    print ('args.numWorker: ', args.numWorker , 'args.gpuPerWorker: ', args.gpuPerWorker)
    return num_worker

BATCH_SIZE_PER_DEVICE = 64
NUM_EPOCHS = 3
MAX_NUM_WORKERS = get_max_worker()
START_LEARNING_RATE = 0.4
LR_STEP_SIZE = 30
LR_GAMMA = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

## Define dataset location 
DATA_DIR = os.getenv("DATA_DIR")
if DATA_DIR is None:
    DATA_DIR = '/tmp'
print("DATA_DIR: " + DATA_DIR)
TRAIN_DATA = DATA_DIR + "/cifar10"
TEST_DATA = DATA_DIR + "/cifar10"


## <Xue Yin>  Documentation of Callback function
class LRScheduleCallback(Callback):
    def __init__(self, step_size, gamma):
        super(LRScheduleCallback, self).__init__()
        self.step_size = step_size
        self.gamma = gamma

    def on_epoch_begin(self, epoch):
        if (epoch != 0) and (epoch % self.step_size == 0):
            for param_group in self.params['optimizer'].param_groups:
                param_group['lr'] *= self.gamma

        print("LRScheduleCallback epoch={}, learning_rate={}".format(epoch,
              self.params['optimizer'].param_groups[0]['lr']))

## Data loading function for EDT
def getDatasets():
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return (torchvision.datasets.CIFAR10(root=TRAIN_DATA, train=True, download=True, transform=transform_train),
            torchvision.datasets.CIFAR10(root=TEST_DATA, train=False, download=True, transform=transform_test))

def custom_train(model, data, eva, train_loader, fn_args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    opt = model.get_optimizer()
    opt.zero_grad()
    outputs = model(inputs)
    cri = model.get_loss_function()
    loss = cri(outputs, labels)
    loss.backward()
    acc = eva(outputs, labels)
    return acc, loss

def custom_test(model, test_iter, fn_args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cri = model.get_loss_function()
    valid_loss = 0.0
    counter = 0
    for(inputs, labels) in test_iter:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = cri(output, labels)
        valid_loss += loss.item()
        counter += 1
    valid_loss /= counter
    return valid_loss

def main(model_type):
    print('==> Building model..' + str(model_type))
    model = models.__dict__[model_type]()
    optimizer = optim.SGD(model.parameters(), lr=START_LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_function = F.cross_entropy
    
    edt_m = FabricModel(model, getDatasets, loss_function, optimizer, enable_onnx=True, fn_step_train=custom_train, fn_test=custom_test, user_callback=[LRScheduleCallback(LR_STEP_SIZE, LR_GAMMA)],  driver_logger=EDTLoggerCallback())
    print('==> epochs:' + str(NUM_EPOCHS) + ', batchsize:' + str(BATCH_SIZE_PER_DEVICE) + ', engines_number:' + str(MAX_NUM_WORKERS))
    edt_m.train(NUM_EPOCHS, BATCH_SIZE_PER_DEVICE, MAX_NUM_WORKERS, num_dataloader_threads=4, validation_freq=10, checkpoint_freq=0)

if __name__ == '__main__':
    main("resnet50")
