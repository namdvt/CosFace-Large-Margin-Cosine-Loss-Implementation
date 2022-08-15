import torch
import torch.nn as nn
import tqdm
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from cosine_margin_loss import CosineMarginLoss
from model import Model
from torch.utils.data import random_split

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def fit(epoch, model, optimizer, criterion, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_margin_loss = 0.0
    running_ce_loss = 0.0

    for data, label in tqdm(data_loader):
        data = data.to(device)
        label = label.to(device)
        if phase == 'training':
            optimizer.zero_grad()
            output = model(data)
        else:
            with torch.no_grad():
                output = model(data)

        # compute loss
        loss_margin = criterion[0](output[0], label)
        loss_ce = criterion[1](output[1], label)
        loss = loss_margin + loss_ce

        running_margin_loss += loss_margin.item()
        running_ce_loss += loss_ce.item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    print(f'[{epoch}][{phase}]loss_margin: {running_margin_loss/len(data_loader)} loss_ce: {running_ce_loss/len(data_loader)}')

    return (running_margin_loss + running_ce_loss)/len(data_loader)


def train(batch_size, model, num_epochs, lr):
    print('loading data ...........')
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_set, val_set = random_split(dataset, [54000, 6000])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    criterion = (CosineMarginLoss(embed_dim=3, num_classes=10).to(device), nn.CrossEntropyLoss())

    for epoch in range(num_epochs):
        fit(epoch, model, optimizer, criterion, train_loader, phase='training')
        fit(epoch, model, optimizer, criterion, val_loader, phase='validation')
        print('-----------------------------------------')
        torch.save(model.state_dict(), f'ckpts/ckpt_epoch_{epoch}.pth')

        scheduler.step()


def write_figures(train_losses, val_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig('output/loss.png')
    plt.close('all')


if __name__ == "__main__":
    model = Model().to(device)
    num_epochs = 30
    learning_rate = 0.01
    batch_size = 32
    train(batch_size, model, num_epochs, learning_rate)
