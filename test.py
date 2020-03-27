import torch
import torchvision
import tqdm
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Model
import torch.nn.functional as F
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D


def plot(embeds, labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)

    embeds = np.stack(embeds)
    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.tight_layout()
    plt.savefig('output/result.png')


def test(test_loader, model):
    all_embeds = []
    all_labels = []
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            embed = model(data)
            embed = F.normalize(embed, p=2, dim=1).squeeze().cpu().numpy()
            all_embeds.append(embed)
            all_labels.append(label)

    all_embeds = np.array(all_embeds)
    all_labels = np.array(all_labels)

    plot(all_embeds, all_labels)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    model = Model()
    model.eval()
    model.load_state_dict(torch.load('output/weight.pth'))

    test(test_loader, model)