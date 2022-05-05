import torch
import torch.nn.functional as F
from torch_scatter import scatter

def voxel_to_edge(voxel, device = 'cuda'):
    '''
    input:
        voxel: [batch_size, 1, res, res, res]
    return:
        hyper_edge: [edge_num, 8]
    '''
    device = torch.device(device)
    voxel = torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0).to(device).float()
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
    return hyper_edge[voxel.squeeze(1)>0].long().cpu().numpy()

def spmm_conv(x, edge, kernel, reduce = 'sum'):
    '''
    input:
        x:      [vert_num, 3*feature_num]
        edge:   [voxel_num, 8]
    return:
        x_:     [vert_num, 3*feature_num]
    '''
    voxel_num = edge.shape[0]
    feature_num = x.shape[-1]//3
    x = x[edge].reshape(voxel_num, 24, feature_num).permute(0, 2, 1) # [voxel_num, feature_num, 24]
    x = (x @ kernel).permute(0, 2, 1).reshape(voxel_num*8, 3*feature_num) #[voxel_num, 24, feature_num]
    return scatter(x, edge.reshape(-1), dim=0, reduce=reduce)

def vert2vox(x, edge):
    '''
    input:
        x:      [vert_num, feature_num]
        edge:   [voxel_num, 8]
    return:
        x_:     [voxel_num, 8*feature_num]
    '''
    voxel_num = edge.shape[0]
    feature_num = x.shape[-1]
    return x[edge].reshape(voxel_num, 8*feature_num)

def vox2vert(x, edge, reduce = 'sum'):
    '''
    input:
        x:      [voxel_num, 8*feature_num]
        edge:   [voxel_num, 8]
    return:
        x_:     [vert_num, feature_num]
    '''
    voxel_num = x.shape[0]
    feature_num = x.shape[-1]//8
    return scatter(
            x.reshape(voxel_num*8, feature_num), 
            edge.reshape(-1), 
            dim=0, 
            reduce=reduce
        )

def diag_matrix(edge):
    x = torch.ones(edge.shape[0]*8, 1, device=edge.device)
    return scatter(
            x,
            edge.reshape(-1), 
            dim=0, 
            reduce='sum',
        )

