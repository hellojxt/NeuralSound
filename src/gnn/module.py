import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class Grid():
    '''
        voxel:      [batch_size, 1, res, res, res] or [res,res,res] for ndarray
        edge:       [edge_num, 8]
        x:          [node_num, feature_num, vector_num]
        pool_edge:  [edge_num], the value represents new edge index 
        finer_grid: point to finer grid
    '''
    def __init__(self, voxel = None, edge = None, pool_edge = None, finer_grid = None, device = torch.device('cuda:0')):
        if not isinstance(voxel, torch.Tensor):
            voxel = torch.tensor(voxel).unsqueeze(0).unsqueeze(0).to(device)
        self.voxel = voxel
        self.edge_ = edge
        self.pool_edge = pool_edge
        self.finer_grid = finer_grid

    @property
    def edge(self):
        if self.edge_ == None:
            self.edge_ = self.voxel2edge()
        return self.edge_
    @edge.setter
    def edge(self, e):
        self.edge_ = e

    
    def voxel2edge(self):
        '''
        input:
            voxel: [batch_size, 1, res, res, res]
        return:
            hyper_edge: [edge_num, 8]
        '''
        voxel = self.voxel.float()
        grid = F.conv_transpose3d(voxel, torch.ones(1,1,2,2,2,device=voxel.device).float())
        mask = (grid > 0)
        grid[mask] = torch.arange(mask.sum(), device=voxel.device).float()
        kernel = torch.tensor([
            1,0,0,0,0,0,0,0,
            0,0,0,0,1,0,0,0,
            0,0,0,0,0,0,1,0,
            0,0,1,0,0,0,0,0,
            0,1,0,0,0,0,0,0,
            0,0,0,0,0,1,0,0,
            0,0,0,0,0,0,0,1,
            0,0,0,1,0,0,0,0,
        ]).reshape(8,1,2,2,2).float().to(voxel.device)
        hyper_edge = F.conv3d(grid, kernel).permute(0,2,3,4,1)
        return hyper_edge[voxel.squeeze(1)>0].long()

class GraphData():
    def __init__(self, grid, x):
        '''
        input grid or voxel
        '''
        if not isinstance(grid, Grid):
            grid = Grid(voxel=grid)
        self.grid = grid
        self.x = x


def hyper_graph_conv(data, mlp, reduce = 'sum'):
    '''
    input: 
        edge:   [edge_num, 8]
        x:      [node_num, feature_num, vector_num]
    return:
        x_:     [node_num, feature_num, vector_num]
    '''
    grid, x = data.grid, data.x
    edge = grid.edge
    edge_num = edge.shape[0]
    node_num, feature_num, vector_num = x.shape
    x_ij = x[edge].permute(3,0,1,2).reshape(vector_num, edge_num, 8*feature_num)
    x_ij = mlp(x_ij).reshape(vector_num, 8*edge_num, -1)
    x_ = scatter(x_ij, edge.reshape(-1), dim=1, reduce=reduce).permute(1,2,0)
    return GraphData(data.grid, x_)


class UpSample(nn.Module):
    def __init__(self, channels_in = None, channels_out = None):
        super().__init__()
        if channels_in is not None:
            self.mlp = nn.Conv1d(channels_in, channels_out, 1)
        else:
            self.mlp = None
    
    
    def forward(self,data):
        '''
        input: 
            voxel:      [batch_size, 1, res, res, res]
            edge:       [edge_num, 8]
            x:          [node_num, feature_num, vector_num]
            pool_edge:  [finer_edge_num], the value represents new edge index 
            finer_grid: point to last fine grid
        return:
            x_:         [finer_node_num, feature_num, vector_num]
        '''
        grid, x = data.grid, data.x
        x_ij = x[grid.edge]
        edge_ = grid.finer_grid.edge.reshape(-1)
        node_num, feature_num, vector_num = x.shape
        x_ij_ = x_ij[grid.pool_edge].reshape(edge_.shape[0], feature_num, vector_num)
        x_ = scatter(x_ij_, edge_, dim=0, reduce='mean')
        if self.mlp is not None:
            x_ = self.mlp(x_)
        return GraphData(grid.finer_grid, x_)

class DownSampleTranspose(nn.Module):
    def __init__(self, channels_in = None, channels_out = None):
        super().__init__()
        if channels_in is not None:
            self.mlp = nn.Conv1d(channels_in, channels_out, 1)
        else:
            self.mlp = None
    
    def forward(self,data):
        '''
        input: 
            voxel:      [batch_size, 1, res, res, res]
            edge:       [edge_num, 8]
            x:          [node_num, feature_num, vector_num]
        return:
            voxel_:     [batch_size, 1, res/2, res/2, res/2]
            edge_:      [new_edge_num, 8]
            pool_edge:  [edge_num], the value represents new edge index 
            finer_grid: point to last fine grid
            x_:         [new_node_num, feature_num, vector_num]
        '''
        if self.mlp is not None:
            data.x = self.mlp(data.x)
        grid, x = data.grid, data.x
        voxel, edge = grid.voxel, grid.edge
        voxel = voxel.float()
        voxel_ = F.max_pool3d(voxel, kernel_size=2, stride=2)
        grid_ = Grid(voxel_)

        mask = (voxel_ > 0)
        voxel__ = voxel_.clone()
        voxel__[mask] = torch.arange(mask.sum(), device=voxel.device).float()
        pool_edge = F.interpolate(voxel__, scale_factor=2)[voxel > 0].long()
        grid_.pool_edge = pool_edge
        grid_.finer_grid = grid
    

        edge_num = edge.shape[0]
        node_num, feature_num, vector_num = x.shape
        degree = scatter(torch.ones(8*edge_num, 1, 1, device=x.device), edge.reshape(-1), dim=0, reduce='sum')
        x_ij = (x / degree)[edge].permute(3,0,1,2).reshape(vector_num, edge_num, 8*feature_num)
        x_ij_ = scatter(x_ij, pool_edge, dim=1, reduce='mean')
        edge_num = x_ij_.shape[1]

        x_ = scatter(x_ij_.reshape(vector_num, 8*edge_num, feature_num), 
                    grid_.edge.reshape(-1), 
                    dim=1, reduce='mean').permute(1,2,0)

        return GraphData(grid_, x_)

class DownSample(nn.Module):
    def __init__(self, channels_in = None, channels_out = None):
        super().__init__()
        if channels_in is not None:
            self.mlp = nn.Conv1d(channels_in, channels_out, 1)
        else:
            self.mlp = None
    
    def forward(self,data):
        '''
        input: 
            voxel:      [batch_size, 1, res, res, res]
            edge:       [edge_num, 8]
            x:          [node_num, feature_num, vector_num]
        return:
            voxel_:     [batch_size, 1, res/2, res/2, res/2]
            edge_:      [new_edge_num, 8]
            pool_edge:  [edge_num], the value represents new edge index 
            finer_grid: point to last fine grid
            x_:         [new_node_num, feature_num, vector_num]
        '''
        if self.mlp is not None:
            data.x = self.mlp(data.x)
        grid, x = data.grid, data.x
        voxel, edge = grid.voxel, grid.edge
        voxel = voxel.float()
        voxel_ = F.max_pool3d(voxel, kernel_size=2, stride=2)
        grid_ = Grid(voxel_)

        mask = (voxel_ > 0)
        voxel__ = voxel_.clone()
        voxel__[mask] = torch.arange(mask.sum(), device=voxel.device).float()
        pool_edge = F.interpolate(voxel__, scale_factor=2)[voxel > 0].long()
        grid_.pool_edge = pool_edge
        grid_.finer_grid = grid
    

        edge_num = edge.shape[0]
        node_num, feature_num, vector_num = x.shape
        x_ij = x[edge].permute(3,0,1,2).reshape(vector_num, edge_num, 8*feature_num)
        x_ij_ = scatter(x_ij, pool_edge, dim=1, reduce='mean')
        edge_num = x_ij_.shape[1]

        x_ = scatter(x_ij_.reshape(vector_num, 8*edge_num, feature_num), 
                    grid_.edge.reshape(-1), 
                    dim=1, reduce='mean').permute(1,2,0)

        return GraphData(grid_, x_)


