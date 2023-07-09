import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

batch_size, lr, num_epochs = 256, 0.01, 5
train_losses = []
train_counter = []


def train(model, train_loader, num_epochs, optimizer):
    model.train()
    for epoch in range(1, num_epochs + 1):
        for batch, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, label)
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, batch * len(data), len(train_loader.dataset), 100. * batch / len(train_loader), loss.item()
                ))
                train_losses.append(loss.item())
                train_counter.append((batch * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    train_loader = DataLoader(dataset=torchvision.datasets.MNIST(
        '../data', train=True, download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    ), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=torchvision.datasets.MNIST(
        '../data', train=False, download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    ), batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 1*28*28 -> 16*28*28
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 16*28*28 -> 16*14*14
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 32*14*14 -> 32*7*7
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # train
    train(model, train_loader, num_epochs, optimizer)
