import torch
import torchvision
import tqdm
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from cosine_margin_loss import CosineMarginLoss
from model import Model
from torch.utils.data import random_split


def fit(epoch, model, optimizer, criterion, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    for data, label in tqdm(data_loader):
        if phase == 'training':
            optimizer.zero_grad()
            output = model(data)
        else:
            with torch.no_grad():
                output = model(data)

        # compute loss
        loss = criterion(output, label)
        running_loss += loss.item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader)
    print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))

    return epoch_loss


def train(batch_size, model, num_epochs, lr):
    print('loading data ...........')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_set, val_set = random_split(dataset, [54000, 6000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 1)
    criterion = CosineMarginLoss(embed_dim=3, num_classes=10)

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        scheduler.step(epoch)

        train_epoch_loss = fit(epoch, model, optimizer, criterion, train_loader, phase='training')
        val_epoch_loss = fit(epoch, model, optimizer, criterion, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_loss < np.min(val_losses):
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        write_figures(train_losses, val_losses)


def write_figures(train_losses, val_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig('output/loss.png')
    plt.close('all')


if __name__ == "__main__":
    model = Model()
    num_epochs = 201
    learning_rate = 0.1
    batch_size = 64
    train(batch_size, model, num_epochs, learning_rate)
