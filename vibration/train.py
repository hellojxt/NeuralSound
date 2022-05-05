import sys
import argparse
import MinkowskiEngine as ME
import torch
import torch.nn as nn
from torch import optim
sys.path.append('..')
from dataset import LobpcgDataset, lobpcg_collation_fn
from src.net.trainer_template import *
import src.net.eigennet as eigennet
from src.classic.fem.femModel import Hexahedron_model, Material
from src.classic.lobpcg import relative_error, random_init, matrix_norm, variance, vec_norm
from src.classic.fem.project_util import spmm_conv, vox2vert, vert2vox, diag_matrix
from src.classic.lobpcg import lobpcg

# torch.set_grad_enabled(False)

def get_norms(X, edge):
    X_norm = float(torch.norm(X))
    iX_norm = X_norm ** -1
    A_norm = float(torch.norm(A(X, edge))) * iX_norm
    B_norm = float(torch.norm(B(X, edge))) * iX_norm
    return A_norm, B_norm

def rerr(U, E, edge, A_norm, B_norm):
    # check_shape(X, E)
    R = A(U, edge) - B(U, edge) * E
    rerr = torch.norm(R, 2, (0, )) * (torch.norm(U, 2, (0, )) * (A_norm + E * B_norm)) ** -1
    return rerr

def get_svqb(U, edge):
    tau = 1.2e-07
    UBU = U.T @ B(U, edge)
    d = UBU.diagonal(0, -2, -1)
    nz = torch.where(abs(d) != 0.0)
    assert len(nz) == 1, nz
    if len(nz[0]) < len(d):
        U = U[:, nz[0]]
        UBU = U.T @ B(U, edge)
        d = UBU.diagonal(0, -2, -1)
        nz = torch.where(abs(d) != 0.0)
        assert len(nz[0]) == len(d)
    d_col = (d ** -0.5).reshape(d.shape[0], 1)
    DUBUD = (UBU * d_col) * d_col.T
    E, Z = torch.linalg.eigh(DUBUD, UPLO='U')
    t = tau * abs(E).max()
    keep = torch.where(E > t)
    assert len(keep) == 1, keep
    E = E[keep[0]]
    Z = Z[:, keep[0]]
    # check_shape(U, d_col, Z, E, (U * d_col.T), (Z * E ** -0.5))
    U = (U * d_col.T) @ (Z * E ** -0.5)
    UAU = U.T @ A(U, edge)
    E, Z = torch.linalg.eigh(UAU, UPLO='U')
    return U @ Z[:, :n ], E[:n]

def A(x, edge):
    vert_num = x.shape[0]//3
    x = spmm_conv(x.reshape(vert_num, -1), edge, stiff_e)
    return x.reshape(vert_num*3, -1)

def B(x, edge):
    vert_num = x.shape[0]//3
    x = spmm_conv(x.reshape(vert_num, -1), edge, mass_e)
    return x.reshape(vert_num*3, -1)

def net(coords, x, edge):
    # if args.diag:
    #     D = diag_matrix(edge)
    #     x = torch.cat((x, D), 1)
    vert_num = x.shape[0]//3
    x = x.reshape(vert_num, 3*n)
    feats_in = vert2vox(x, edge)  # [voxel_num, 8*3*n]
    voxel_num = feats_in.shape[0]
    feats_in = feats_in.reshape(voxel_num, 8*3, n).permute(2,0,1).reshape(n*voxel_num, 8*3)
    bcoords = ME.utils.batched_coordinates(coords*n).to(device)
    sparse_tensor = ME.SparseTensor(feats_in, bcoords)
    output = Config.net(sparse_tensor).F
    output = output.reshape(n, voxel_num, 8*3*out_k).permute(1, 2, 0).reshape(voxel_num, 8*3*out_k*n)
    output = vox2vert(output, edge, 'mean').reshape(vert_num*3, out_k*n)
    return output



def forward_fun(items):
    coords, edge, ind_lst, filename = items
    vert_num = ind_lst[-1]
    x0 = torch.randn(vert_num*3, n).to(device)
    A_norm, B_norm = get_norms(x0, edge)
    x, E = get_svqb(torch.cat((x0, net(coords, x0, edge)), dim=1), edge)
    err = rerr(x, E, edge, A_norm, B_norm)
    err = err[:20].mean()
    E = E[:20].mean()
    return err, E

    
def loss_fun(err, E):
    return {
        'err': err,
        'E': E,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='../dataset/lobpcg')
    parser.add_argument('--net', type=str, default='defaultUnet')
    parser.add_argument('--nonlinear', dest='nonlinear', action='store_true')
    parser.set_defaults(nonlinear=False)
    parser.add_argument('--diag', dest='diag', action='store_true')
    parser.set_defaults(diag=False)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    Config.dataset_root_dir = args.dataset
    Config.tag = args.tag
    device = torch.device(f'cuda:{args.cuda}')
    Config.device = device
    Config.CustomDataset = LobpcgDataset
    Config.custom_collation_fn = lobpcg_collation_fn
    Config.forward_fun = forward_fun
    Config.loss_fun = loss_fun

    if args.diag:
        in_feat_num = 32
    else:
        in_feat_num = 24
    out_k = 2
    out_feat_num = 24*out_k
    Config.net = getattr(eigennet, args.net)(in_feat_num, out_feat_num, linear = (not args.nonlinear))

    # Config.net = ConvNet(24, 24)
    Config.optimizer = optim.Adam(Config.net.parameters(), lr=1e-4)
    Config.scheduler = optim.lr_scheduler.StepLR(Config.optimizer, step_size=5, gamma=0.8)
    Config.BATCH_SIZE = 1
    Config.dataset_worker_num = 8

    hex_model = Hexahedron_model(mat = Material.Ceramic)
    stiff_e = torch.tensor(hex_model._element_stiff_matrix).to(device).float()
    stiff_e = stiff_e / vec_norm(stiff_e) / 2
    mass_e = torch.tensor(hex_model._element_mass_matrix).to(device).float()
    mass_e = mass_e / vec_norm(mass_e) / 2
    n = 20
    # print(stiff_e, mass_e)
    start_train(200)



'''
conda deactivate; conda remove -n modal_analysis --all -y
conda create -n modal_analysis python=3.7 pip -y
conda activate modal_analysis
conda install -c conda-forge bempp-cl -y
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy matplotlib numba tensorboard tqdm
pip install cupy
export CUDA_HOME=/usr/local/cuda-11.1/
cd /data1/MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
'''



