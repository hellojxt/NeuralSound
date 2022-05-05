import sys
sys.path.append('..')
from acoustic.dataset import AcousticDatasetFar, acoustic_collation_fn, AcousticDataset 
from src.net.trainer_template import *
from src.net.acousticnet import AcousticNet
from torch.nn.functional import mse_loss
import argparse
import os
import torch
from PIL import Image
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from time import time
def to_img(data1, data2, filename):
    data1 = data1.T / data1.max()
    data2 = data2.T / data2.max()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(211)
    ax.matshow(data1.detach().cpu().numpy())
    ax = fig.add_subplot(212)
    ax.matshow(data2.detach().cpu().numpy())
    plt.savefig(filename)
    plt.clf()
    plt.close()

def forward_fun(items):
    bcoords, feats_in, feats_out, feats_out_norm, filename = items
    t = time()
    ffat_map, ffat_norm = Config.net(bcoords, feats_in)
    # check_shape(bcoords, feats_in, freq_norm, feats_out, feats_out_norm,ffat_map, ffat_norm)
    print('time cost:', time() - t)
    save_interval = 100
    if Config.idx_in_epoch % save_interval == 0:
        with torch.set_grad_enabled(False):
            log_dir = f'./images/{Config.tag}/{Config.epoch_idx}/{Config.phase}'
            os.makedirs(log_dir, exist_ok=True)
            gt, predict = feats_out[0,0], ffat_map[0,0]
            # check_shape(gt, predict)
            to_img(gt, predict, f'{log_dir}/{Config.idx_in_epoch // save_interval}.png')
            
    return ffat_map, ffat_norm, feats_out, feats_out_norm

def loss_fun(ffat_map, ffat_norm, feats_out, feats_out_norm):
    loss1 = mse_loss(ffat_map, feats_out)
    loss2 = mse_loss(ffat_norm, feats_out_norm)
    # print(loss1.item(), loss2.item())
    return {
        'ffatmap': loss1,
        'norm': loss2,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--dataset', type=str, default='../dataset/acousticMapData')
    parser.add_argument('--linear', dest='linear', action='store_true')
    parser.set_defaults(linear=False)
    parser.add_argument('--far', dest='far', action='store_true')
    parser.set_defaults(far=False)
    parser.add_argument('--wdir', type=str, default='./weights')
    args = parser.parse_args()
    Config.dataset_root_dir = args.dataset
    Config.tag = args.tag
    if args.far:
        Config.CustomDataset = AcousticDatasetFar
    else:
        Config.CustomDataset = AcousticDataset 
    Config.custom_collation_fn = acoustic_collation_fn
    Config.forward_fun = forward_fun
    Config.loss_fun = loss_fun

    Config.net = AcousticNet(23, linear = args.linear)
    Config.optimizer = optim.Adam(Config.net.parameters(), lr=1e-3)
    Config.scheduler = optim.lr_scheduler.StepLR(Config.optimizer, step_size=8, gamma=0.8)
    Config.BATCH_SIZE = 40
    Config.dataset_worker_num = 8
    Config.weights_dir = args.wdir
    Config.debug = True
    Config.batch_num_limit = 1
    start_train(2)

'''

conda create -n mink python=3.7 pip -y
conda activate mink
conda install openblas-devel -c anaconda -y
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
export CUDA_HOME=/usr/local/cuda-10.1
export CXX=g++-7
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

'''