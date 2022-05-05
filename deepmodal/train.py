import sys
import argparse
import MinkowskiEngine as ME
from numpy.linalg import eig
import torch
import torch.nn as nn
from torch import optim
sys.path.append('..')
from dataset import DeepModalDataset, deepmodal_collation_fn
from src.net.trainer_template import *
import src.net.eigennet as eigennet
from src.classic.fem.femModel import Hexahedron_model, Material
from src.classic.lobpcg import relative_error, random_init, matrix_norm, variance, vec_norm
from src.classic.fem.project_util import spmm_conv, vox2vert, vert2vox, diag_matrix
from src.classic.lobpcg import lobpcg

# torch.set_grad_enabled(False)
def net(coords, x, edge):
    vert_num = x.shape[0]//3
    x = x.reshape(vert_num, 3*n)
    feats_in = vert2vox(x, edge)  # [voxel_num, 8*3*n]
    voxel_num = feats_in.shape[0]
    feats_in = feats_in.reshape(voxel_num, 8*3*n)
    bcoords = ME.utils.batched_coordinates(coords).to(device)
    sparse_tensor = ME.SparseTensor(feats_in, bcoords)
    output = Config.net(sparse_tensor).F
    output = vox2vert(output, edge, 'mean').reshape(vert_num*3, n)
    return output


def forward_fun(items):
    coords, edge_shifted, ind_lst, vecs, mask, filename = items
    vert_num = ind_lst[-1]
    x0 = torch.randn(vert_num*3, n).to(device)
    x1 = net(coords, x0, edge_shifted)
    vecs = vecs[0].to(device)
    mask = mask[0].to(device)
    err_vecs = torch.norm(vecs - x1[:,:n//2]) / torch.norm(vecs)
    err_mask = torch.norm(mask - x1[:,n//2:]) / torch.norm(mask)
    return err_vecs + err_mask

    
def loss_fun(err):
    # print(err.item())
    return err

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='../dataset/deepmodal')
    parser.add_argument('--net', type=str, default='defaultUnet')
    parser.add_argument('--nonlinear', dest='nonlinear', action='store_true')
    parser.set_defaults(nonlinear=True)
    parser.add_argument('--diag', dest='diag', action='store_true')
    parser.set_defaults(diag=False)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    Config.dataset_root_dir = args.dataset
    Config.tag = args.tag
    device = torch.device(f'cuda:{args.cuda}')
    Config.device = device
    Config.CustomDataset = DeepModalDataset
    Config.custom_collation_fn = deepmodal_collation_fn
    Config.forward_fun = forward_fun
    Config.loss_fun = loss_fun

    n = 32*2
    if args.diag:
        in_feat_num = 32
    else:
        in_feat_num = 24*n
    out_k = 2
    out_feat_num = 24*n

    Config.net = getattr(eigennet, args.net)(in_feat_num, out_feat_num, linear = (not args.nonlinear))

    # Config.net = ConvNet(24, 24)
    Config.optimizer = optim.Adam(Config.net.parameters(), lr=1e-4)
    Config.scheduler = optim.lr_scheduler.StepLR(Config.optimizer, step_size=5, gamma=0.8)
    Config.BATCH_SIZE = 1
    Config.dataset_worker_num = 8

    start_train(100)