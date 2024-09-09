import torch
from torch import nn

import numpy as np
import os
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh, Sigmoid, BCELoss 
import torch.optim as optim
import torchvision.utils as vutils

from PIL import Image
from discriminator import Discriminator
from generator import Generator
from create_dataset import SpaceshipDataset

def plot_images(imgs, grid_size = 5):
    """
    imgs: vector containing all the numpy images
    grid_size: 2x2 or 5x5 grid containing images
    """
     
    fig = plt.figure(figsize = (8, 8))
    columns = rows = grid_size
    plt.title("Training Images")
 
    for i in range(1, columns*rows +1):
        plt.axis("off")
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i])
    plt.show()

def init_weights(m):
    if type(m) == ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    device = torch.device("cpu")

torch.manual_seed(111)

torch.autograd.set_detect_anomaly(True)

imgs = np.load('spaceships.npz')
plot_images(imgs['arr_0'], 3)
transpose_imgs = np.transpose(np.float32(imgs['arr_0']), (0, 3, 1, 2))

batch_size = 64
shuffle = True
dset = SpaceshipDataset(transpose_imgs)

dataloader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=shuffle)

discriminator = Discriminator().to(device=device)
generator = Generator().to(device=device)

discriminator.apply(init_weights)
generator.apply(init_weights)

optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = 0.0001, betas=(0.5, 0.999))
optimizer_generator = optim.Adam(generator.parameters(), lr = 0.0001, betas=(0.5, 0.999))

num_epochs = 50000
loss = nn.BCELoss()

os.makedirs('output_images', exist_ok=True)

for e in range(num_epochs):
    for i, b in enumerate(dataloader):
        real_images = b.to(device)
        
        optimizer_discriminator.zero_grad()
        
        yhat_real = discriminator(real_images).view(-1)
        target_real = torch.ones(len(b), dtype=torch.float, device=device)
        loss_real = loss(yhat_real, target_real)
        
        loss_real.backward()
        noise = torch.randn(len(b), 100, 1, 1, device=device)
        fake_img = generator(noise)
        
        yhat_fake = discriminator(fake_img.detach()).view(-1)
        target_fake = torch.zeros(len(b), dtype=torch.float, device=device)
        loss_fake = loss(yhat_fake, target_fake)
        
        loss_fake.backward()
        optimizer_discriminator.step()
        loss_disc = loss_real + loss_fake
        
        optimizer_generator.zero_grad()
        
        yhat_fake_for_gen = discriminator(fake_img).view(-1)
        target_gen = torch.ones(len(b), dtype=torch.float, device=device)
        loss_gen = loss(yhat_fake_for_gen, target_gen)
        
        loss_gen.backward()
        
        optimizer_generator.step()

    if e % 1000 == 0:
        fake_img_rescaled = (fake_img + 1) / 2  # Now in range [0, 1]

        save_path = f'output_images/fake_image_epoch_{e}_batch_{i}.png'


        vutils.save_image(fake_img_rescaled.detach(), save_path, normalize=True)

        print("******************")
        print(f"Epoch {e} and iteration {i}")


