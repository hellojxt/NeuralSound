import torch
import torch.nn as nn
from .module import *

class BasicBlock(nn.Module):
    def __init__(self, channels_in, channels_out, non_linear = True):
        super().__init__()
        self.mlp1 = nn.Linear(8*channels_in, 8*channels_out)
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.relu = Activate(non_linear)
        self.mlp2 = nn.Linear(8*channels_out, 8*channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        if channels_in != channels_out:
            self.skip = nn.Conv1d(channels_in, channels_out, 1)
        else:
            self.skip = None

    
    def forward(self, data):
        out = hyper_graph_conv(data, self.mlp1)
        out.x = self.relu(self.bn1(out.x))
        out = hyper_graph_conv(out, self.mlp2)
        out.x = self.bn2(out.x)
        if self.skip is not None:
            out.x += self.skip(data.x)
        else:
            out.x += data.x
        out.x = self.relu(out.x)
        return out

class MConv1d(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = nn.Conv1d(channels_in, channels_out, 1, bias=False)
        
    def forward(self, data):
        data.x = self.conv(data.x)
        return data


class GraphUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [4, 16, 64, 128]
        self.deep = 1
        self.conv1 = MConv1d(3, self.channels[0])
        for i in range(len(self.channels)-1):
            setattr(self, f'encode{i}', self._make_layers(self.channels[i], self.deep))
            setattr(self, f'down{i}', DownSample(self.channels[i], self.channels[i+1]))
        
        for i in range(len(self.channels)-1, 0, -1):
            setattr(self, f'decode{i}', self._make_layers(self.channels[i], self.deep))
            setattr(self, f'up{i}', UpSample(self.channels[i], self.channels[i-1]))
        self.conv2 = MConv1d(self.channels[0], 3)

    def _make_layers(self, channels, block_num, block = BasicBlock):
        return nn.Sequential(*[block(channels, channels) for _ in range(block_num)])

    
    def forward(self, data):
        x_in = data.x
        out = self.conv1(data)
        x = {}
        for i in range(len(self.channels)-1):
            out = getattr(self, f'encode{i}')(out)
            x[self.channels[i]] = out.x
            out = getattr(self, f'down{i}')(out)
        
        for i in range(len(self.channels)-1, 0, -1):
            out = getattr(self, f'decode{i}')(out)
            out = getattr(self, f'up{i}')(out)
            out.x = out.x + x[self.channels[i-1]]
            
        out = self.conv2(out)
        return out.x + x_in






        