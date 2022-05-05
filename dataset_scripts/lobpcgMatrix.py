import sys
sys.path.append('..')
from src.classic.fem.util import scipy2torch, to_sparse_coords, vertex_to_voxel_matrix, voxel_to_vertex_matrix
from src.classic.fem.project_util import voxel_to_edge
from src.classic.fem.femModel import Hexahedron_model, Material
import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

def process_single(filename, output_name):
    dir_name = os.path.dirname(output_name)
    os.makedirs(dir_name, exist_ok=True)
    voxel = np.load(filename)
    torch.save({
        'coords': torch.from_numpy(to_sparse_coords(voxel)),
        'edge': torch.from_numpy(voxel_to_edge(voxel))
    }, output_name)

if __name__ == '__main__':
    file_list = glob(sys.argv[1])
    out_dir = sys.argv[2]
    for filename in tqdm(file_list):
        out_name = os.path.join(out_dir, os.path.basename(filename).replace('.npy', '.pt'))
        process_single(filename, out_name)
        