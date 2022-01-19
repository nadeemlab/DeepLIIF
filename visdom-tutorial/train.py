from __future__ import division

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

import numpy as np

import utils


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainEpoch(train_loader, model, criterion, optimizer, epoch):

    # object to store & plot the losses
    losses = utils.AverageMeter()

    # switch to train mode
    model.train()

    # Train in mini-batches
    for batch_idx, data in enumerate(train_loader):

        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.data.cpu().numpy(), labels.size(0))
        loss.backward()
        optimizer.step()

        # Print info
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), loss=losses))

    # Plot loss after all mini-batches have finished
    plotter.plot('loss', 'train', 'Class Loss', epoch, losses.avg)


def valEpoch(val_loader, model, criterion, epoch):

    losses = utils.AverageMeter()

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():

        # Mini-batches
        for batch_idx, data in enumerate(val_loader):

            # get the inputs
            inputs, labels = data

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.update(loss.data.cpu().numpy(), labels.size(0))
            _, predicted = torch.max(outputs, 1)


            # Save predicted to compute accuracy
            if batch_idx==0:
                out = predicted.data.cpu().numpy()
                label = labels.cpu().numpy()
            else:
                out = np.concatenate((out,predicted.data.cpu().numpy()),axis=0)
                label = np.concatenate((label, labels.cpu().numpy()),axis=0)

        # Accuracy
        acc = np.sum(out == label)/len(out)

        # Print validation info
        print('Validation set: Average loss: {:.4f}\t'
              'Accuracy {acc}'.format(losses.avg, acc=acc))

        # Plot validation results
        plotter.plot('loss', 'val', 'Class Loss', epoch, losses.avg)
        plotter.plot('acc', 'val', 'Class Accuracy', epoch, acc)

        # Return acc as the validation outcome
        return acc


def trainProcess():

    # Load model
    model = Net()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_val = float(0)

    # CIFAR-10 Data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=2)

    # Now, let's start the training process!
    print('Training...')
    for epoch in range(100):

        # Compute a training epoch
        trainEpoch(trainloader, model, criterion, optimizer, epoch)

        # Compute a validation epoch
        lossval = valEpoch(testloader, model, criterion, epoch)

        # Print validation accuracy and best validation accuracy
        best_val = max(lossval, best_val)
        print('** Validation: %f (best) - %f (current)' % (best_val, lossval))





if __name__ == "__main__":

    # Plots
    global plotter
    plotter = utils.VisdomLinePlotter(env_name='Tutorial Plots')

    # Training process
    trainProcess()