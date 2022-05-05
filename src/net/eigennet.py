from turtle import forward
from .minkunet_small import *
from .minkunet import *
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as F


class preconditionUnet(MinkUNetBaseSmall):
    BLOCK = Bottleneck
    LAYERS = (1, 1, 1, 1, 1)
    PLANES = (24, 32, 32, 32, 24)

class Convnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3)
    def forward(self, x):
        return self.conv(x)

class defaultUnet(MinkUNetBaseSmall):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1)
    PLANES = (24, 32, 64, 128, 64, 32, 24)

class initUnetLarge(MinkUNetBaseSmall):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1)
    PLANES = (48, 48, 64, 128, 64, 48, 48)

class initsmall(MinkUNetBaseSmall):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1)
    PLANES = (8, 16, 32, 64, 32, 16, 8)


class initbottle(MinkUNetBaseSmall):
    BLOCK = Bottleneck
    LAYERS = (1, 1, 1, 1, 1, 1, 1)
    PLANES = (24, 32, 64, 128, 64, 32, 24)

'''
https://github.com/NVIDIA/MinkowskiEngine/blob/8a9dae528f47e33d6f48820cbdfe6f8e7fab12ef/MinkowskiEngine/utils/coords.py
'''