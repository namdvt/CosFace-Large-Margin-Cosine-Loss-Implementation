from tkinter import font
import torch
import torchvision
import tqdm
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Model
import torch.nn.functional as F
from glob import glob
import imageio


def plot(embeds, labels, out_name):
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

    colormap = plt.cm.gist_ncar
    # colorst = [colormap(i) for i in range(10)] 



    colorst =  plt.cm.rainbow(np.linspace(0, 1, 10))

    color = list()
    for l in labels.T[0]:
        color.append(colorst[l])

    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=np.stack(color), s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.text2D(0.05, 0.95, f'{out_name}', transform=ax.transAxes, fontsize=40)
    plt.tight_layout()
    plt.savefig(f'output/{out_name}.png')


def test(test_loader, model, out_name):
    all_embeds = []
    all_labels = []
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            embed = model(data)[0]
            embed = F.normalize(embed, p=2, dim=1).squeeze().cpu().numpy()
            all_embeds.append(embed)
            all_labels.append(label.cpu().numpy())

    all_embeds = np.array(all_embeds)
    all_labels = np.array(all_labels)

    plot(all_embeds, all_labels, out_name)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    model = Model()
    model.eval()

    # test all ckpts
    test(test_loader, model, 'init')
    for ckpt in glob('ckpts/*.pth'):
        name = ckpt.split('/')[-1].split('.')[0].replace('ckpt_', '')
        model.load_state_dict(torch.load(ckpt))

        test(test_loader, model, name)

    # create gif
    images = []
    images.append(imageio.imread('output/_init.png'))
    for i in range(30):
        images.append(imageio.imread(f'output/epoch_{i}.png'))
    imageio.mimsave('output/result.gif', images, duration=10)