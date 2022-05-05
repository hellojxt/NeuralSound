# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import torch
import torch.nn as nn
import MinkowskiEngine as ME
# from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from .utils import MPrint

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1, linear = False, norm = True):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.norm = norm
        if linear:
            self.relu = nn.Identity()
            MPrint('linear')
        else:
            self.relu = ME.MinkowskiReLU(inplace=True)
            MPrint('non linear')
        self.downsample = downsample

    def forward(self, x):
        residual = x

        # print(x.F.shape)
        out = self.conv1(x)
        if self.norm:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm:
            out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1, linear = False, norm = True):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(
            planes * self.expansion, momentum=bn_momentum)

        if linear:
            self.relu = nn.Identity()
            MPrint('linear')
        else:
            self.relu = ME.MinkowskiReLU(inplace=True)
            MPrint('non linear')
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3, linear = False):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D, linear)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D, linear):

        self.inplanes = self.INIT_DIM
        if linear:
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels, self.inplanes, kernel_size=1, dimension=D
                ),
                ME.MinkowskiBatchNorm(self.inplanes),
            )
            MPrint('linear')
        else:
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels, self.inplanes, kernel_size=1, dimension=D
                ),
                ME.MinkowskiBatchNorm(self.inplanes),
                ME.MinkowskiReLU(inplace=True),
            )
            MPrint('non linear')

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2, linear = linear
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2, linear = linear
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2, linear = linear
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2, linear = linear
        )

        if linear:
            self.conv5 = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
                ),
                ME.MinkowskiBatchNorm(self.inplanes),
            )     
        else:
            self.conv5 = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
                ),
                ME.MinkowskiBatchNorm(self.inplanes),
                ME.MinkowskiReLU(inplace=True),
            )

        self.glob_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1, linear = False, norm = True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if norm:
                downsample = nn.Sequential(
                    ME.MinkowskiConvolution(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        dimension=self.D,
                    ),
                    ME.MinkowskiBatchNorm(planes * block.expansion),
                )
            else:
                downsample = ME.MinkowskiConvolution(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        dimension=self.D,
                    )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
                linear = linear,
                norm = norm
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D, linear = linear, norm = norm
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x)


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)

