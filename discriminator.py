import torch
from torch import nn
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh, Sigmoid, BCELoss 

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.dis = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            LeakyReLU(0.2),

            Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2),

            Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2),

            Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.2),

            Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
           Sigmoid()
        ) 


    def forward(self, input):
        return self.dis(input)