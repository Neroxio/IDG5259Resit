import torch
from torch import nn
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh, Sigmoid, BCELoss 


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.gen = Sequential(
            ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            BatchNorm2d(num_features=512),
            ReLU(),

            ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=256),
            ReLU(),

            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=128),
            ReLU(),

            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=64),
            ReLU(),

            ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            Tanh()
        )

    def forward(self, input):
        return self.gen(input)

