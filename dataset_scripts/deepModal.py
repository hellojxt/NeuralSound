import sys
sys.path.append('..')
from src.classic.tools import *
from src.classic.fem.util import to_sparse_coords
from src.classic.fem.project_util import voxel_to_edge
from glob import glob
from tqdm import tqdm
import os
import torch

def process_single(vecs, vals, voxel, out_file_name):
    dir(out_file_name)
    band_vecs = np.zeros((MelConfig.mel_res, vecs.shape[0]))
    mask = np.zeros(MelConfig.mel_res)
    smooth_vecs  = np.zeros((MelConfig.mel_res, vecs.shape[0]))

    mode_num = len(vals)
    for i in range(mode_num):
        mel_i = val2mel(vals[i])
        mel_idx = mel_index(mel_i)
        band_vecs[mel_idx] += vecs[:, i]
        mask[mel_idx] = 1

    # print((band_vecs**2).sum(axis=1))
    for i in range(MelConfig.mel_res):
        min_dist = np.inf
        for j in range(MelConfig.mel_res):
            if mask[j] == 1 and abs(i-j) < min_dist:
                min_dist = abs(i-j)
                min_idx = j
        smooth_vecs[i] = band_vecs[min_idx]
    # print((smooth_vecs**2).sum(axis=1))

    # normalize smooth_vecs
    smooth_vecs = smooth_vecs / (smooth_vecs**2).mean()**0.5
    # expand mask to (MelConfig.mel_res, vecs.shape[0])
    mask = np.expand_dims(mask, axis=1)
    mask = np.repeat(mask, vecs.shape[0], axis=1)

    torch.save({
        'coords': torch.from_numpy(to_sparse_coords(voxel)),
        'edge': torch.from_numpy(voxel_to_edge(voxel)),
        'vecs': torch.from_numpy(smooth_vecs.T),
        'mask': torch.from_numpy(mask.T),
    }, out_file_name)


def dir(file_name):
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)

if __name__ == '__main__':
    file_list = glob(sys.argv[1])
    out_dir = sys.argv[2]
    for filename in tqdm(file_list):
        data = np.load(filename)
        vecs, vals, voxel = data['vecs'], data['vals'], data['voxel']
        out_file_name = os.path.join(out_dir, os.path.basename(filename).replace('.npz', '.pt'))
        process_single(vecs, vals, voxel, out_file_name)
        



