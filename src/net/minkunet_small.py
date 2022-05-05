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
import torch.nn as nn
import MinkowskiEngine as ME
from .resnet_instance import ResNetBase, BasicBlock, Bottleneck
from .utils import MPrint

class MinkUNetBaseSmall(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 128, 64, 32)
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3, linear = False, final_conv = True):
        self.final_conv = final_conv
        if not linear:
            print('non linear network')
        else:
            print('linear network')
        ResNetBase.__init__(self, in_channels, out_channels, D, linear)

    def network_initialization(self, in_channels, out_channels, D, linear):
        # Output of the first conv concated to conv6
        self.inplanes = in_channels
        self.block0 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], linear = linear)
        self.down0 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiInstanceNorm(self.inplanes),
        )
        self.block1 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], linear = linear)
        self.down1 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiInstanceNorm(self.inplanes),
        )

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], linear = linear)


        self.down2 = nn.Sequential(
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiInstanceNorm(self.inplanes),
        )

        self.block3 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], linear = linear)

        self.up2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiInstanceNorm(self.PLANES[4])
        )
        self.inplanes =  self.PLANES[4]*self.BLOCK.expansion
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4], linear = linear)

        self.up1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiInstanceNorm(self.PLANES[5])
        )
        self.inplanes =  self.PLANES[5]*self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5], linear = linear)

        self.up0 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiInstanceNorm(self.PLANES[6])
        )
        self.inplanes =  self.PLANES[6]*self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6], linear = linear)

        self.final = ME.MinkowskiConvolution(
            self.PLANES[6] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)

        if linear:
            self.relu = nn.Identity()
            MPrint('linear')
        else:
            self.relu = ME.MinkowskiReLU(inplace=True)
            MPrint('non linear')



    def forward(self, x):
        print(x.F.shape)
        out0 = self.block0(x)

        out = self.down0(out0)
        out1 =  self.block1(out)

        out = self.down1(out1)
        out2 = self.block2(out)

        out = self.down2(out2)
        out = self.block3(out)
        out = self.up2(out) + out2

        out = self.block4(out)
        out = self.up1(out) + out1

        out = self.block5(out)
        out = self.up0(out) + out0

        out = self.block6(out)
        return self.final(out)
 



