from numba import njit
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix

def vertex_num_of_voxel(voxel, device = 'cpu'):
    device = torch.device(device)
    voxel = torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0).float()    
    vertex_idx = F.conv_transpose3d(voxel, torch.ones(1,1,2,2,2,device=device).float())
    vertex_num = (vertex_idx > 0).sum()
    return vertex_num

@njit()
def _v2v(v_map, v_idx):
    rows = []
    cols = []
    values = []
    for i in range(len(v_map)):
        slice = v_map[i]
        for j in range(len(slice)):
            row = slice[j]
            for k in range(len(row)):
                maps = row[k]
                num = 0
                for l in range(len(maps)):
                    if maps[l] >= 0 and v_idx[i,j,k] >= 0:
                        num += 1
                for l in range(len(maps)):
                    if maps[l] >= 0 and v_idx[i,j,k] >= 0:
                        rows.append(int(v_idx[i,j,k]))
                        cols.append(int(maps[l]))
                        values.append(1/num)
    return np.array(rows), np.array(cols), np.array(values)


def vertex_to_voxel_matrix(voxel, device = 'cpu'):
    device = torch.device(device)
    voxel = torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0).float()    
    vertex_idx = F.conv_transpose3d(voxel, torch.ones(1,1,2,2,2,device=device).float())
    vertex_num = (vertex_idx > 0).sum()
    vertex_idx[vertex_idx == 0] = -1
    vertex_idx[vertex_idx > 0] = torch.arange(vertex_num, device=device).float()
    voxel_map = F.conv3d(vertex_idx, torch.eye(8, device=device).reshape(8, 1, 2, 2, 2))
    voxel_num = (voxel > 0).sum()
    voxel[voxel == 0] = -1 
    voxel[voxel > 0] = torch.arange(voxel_num, device=device).float()
    voxel_idx = voxel.cpu().numpy()[0,0]
    voxel_map = voxel_map.permute(0,2,3,4,1).cpu().numpy()[0]
    rows, cols, values = _v2v(voxel_map, voxel_idx)
    return coo_matrix((values, (rows, cols)), shape=(voxel_num, vertex_num))

def voxel_to_vertex_matrix(voxel, device = 'cpu'):
    device = torch.device(device)
    voxel = torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0).float()    
    vertex_idx = F.conv_transpose3d(voxel, torch.ones(1,1,2,2,2,device=device).float())
    vertex_num = (vertex_idx > 0).sum()
    vertex_idx[vertex_idx == 0] = -1
    vertex_idx[vertex_idx > 0] = torch.arange(vertex_num, device=device).float()
    voxel_num = (voxel > 0).sum()
    voxel[voxel > 0] = torch.arange(voxel_num, device=device).float() + 1

    vertex_map =  F.conv_transpose3d(voxel, torch.eye(8, device=device).reshape(1, 8, 2, 2, 2)) - 1
    vertex_map = vertex_map.permute(0,2,3,4,1).cpu().numpy()[0]
    vertex_idx = vertex_idx.cpu().numpy()[0,0]
    rows, cols, values = _v2v(vertex_map, vertex_idx)
    return coo_matrix((values, (rows, cols)), shape=(vertex_num, voxel_num))


@njit()
def to_sparse_coords(voxel):
    coords = []
    for i in range(len(voxel)):
        slice = voxel[i]
        for j in range(len(slice)):
            row = slice[j]
            for k in range(len(row)):
                val = row[k]
                if val != 0:
                    coords.append([i, j, k])
    return np.array(coords)

from scipy.sparse import coo_matrix

def scipy2torch(M, device = 'cpu'):
    device = torch.device(device)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long().to(device)
    values = torch.from_numpy(M.data).to(device)
    shape = torch.Size(M.shape)
    M_torch = torch.sparse_coo_tensor(indices, values, shape, device=device)
    return M_torch.coalesce()
