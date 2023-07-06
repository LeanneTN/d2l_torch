# build up a multilayer perception to extract features from FashionMnist from scratch with PyTorch
import torch
from d2l import torch as d2l
from torch.utils import data
import torchvision
from torch import nn
import numpy as np
import pandas as pd

num_epochs, lr, batch_size = 10, 0.5, 256


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def accuracy_cal(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))


if __name__ == '__main__':
    # data loader
    # construct Dataset class objects
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=torchvision.transforms.ToTensor(), download=False
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=torchvision.transforms.ToTensor(), download=False
    )
    # construct dataloaders
    train_loader = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

    # construct MLP
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 144),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(144, 10)
    )

    model.apply(init_weights)
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(model.parameters(), lr)

    train_loss, train_acc = 0., 0.

    for epoch in range(num_epochs):
        if isinstance(model, nn.Module):
            model.train()
        accuracy, loss_value = 0, 0
        metric = d2l.Accumulator(3)
        for X, y in test_loader:
            y_hat = model(X)
            l = loss(y_hat, y)
            if isinstance(trainer, torch.optim.Optimizer):
                trainer.zero_grad()
                l.mean().backward()
                trainer.step()
            else:
                l.sum().backward()
                trainer(X.shape[0])
            metric.add(float(l.sum()), accuracy_cal(y_hat, y), y.numel())
        loss_value = metric[0] / metric[2]
        accuracy = metric[1] / metric[2]
        print(f'loss: {loss_value}, acc: {accuracy}')
