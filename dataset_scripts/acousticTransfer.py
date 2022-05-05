
import sys
sys.path.append('..')
from src.classic.fem.util import to_sparse_coords, vertex_to_voxel_matrix
from src.classic.bem.ffat import ffat_to_imgs, vibration_to_ffat
from src.classic.bem.util import boundary_encode, boundary_voxel
from src.classic.tools import val2freq
from src.classic.fem.femModel import Material, random_material
from tqdm import tqdm
import numpy as np
import os
from glob import glob
import bempp.api
import warnings
warnings.filterwarnings('ignore')

def process_single_model(filename, output_name):
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    data = np.load(filename)
    vecs, vals, voxel = data['vecs'], data['vals'], data['voxel']
    freqs = val2freq(vals)
    coords = to_sparse_coords(voxel)
    coords_surface, feats_index = map(np.asarray,boundary_voxel(coords))
    surface_code = np.asarray(boundary_encode(coords_surface))

    # Skip objects with too many suface voxels for feasible time cost.
    if len(coords_surface) > 5500:
        return

    # Random material and object size for different frequency
    vertex_num = vecs.shape[0] // 3
    mode_num = len(freqs)
    idx_lst = np.arange(mode_num)
    np.random.shuffle(idx_lst)
    freqs_lst = []
    vecs_lst = []
    for mode_idx in idx_lst:
        vec = vertex_to_voxel_matrix(voxel) @ vecs[:, mode_idx].reshape(vertex_num, -1)
        length = np.random.rand() + 0.05
        freq = freqs[mode_idx]*Material.omega_rate(Material.Ceramic, random_material(), length / 0.15)
        if freq < 20:
            freq = 20
        if freq > 20000:
            freq = 20000
        freqs_lst.append(freq)
        vecs_lst.append(vec)
    vecs = np.stack(vecs_lst, axis=2)
    freqs = np.array(freqs_lst)
    
    # caculate ffat map
    ffat_map, ffat_map_far = vibration_to_ffat(coords, vecs, freqs)

    # save ffat map to image for visualization
    # ffat_to_imgs(ffat_map, './', tag='ffat')
    # ffat_to_imgs(ffat_map_far, './', tag='ffat_far')

    np.savez_compressed(output_name,
            coords=coords_surface,
            feats_in = vecs[feats_index],
            feats_out = ffat_map,
            feats_out_far = ffat_map_far,
            surface = surface_code,
            freqs = freqs,
    )


if __name__ == '__main__':

    # commit these two lines if GPU is not available
    bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
    bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"

    file_list = glob(sys.argv[1])
    out_dir = sys.argv[2]
    for filename in tqdm(file_list):
        out_name = os.path.join(out_dir, os.path.basename(filename))
        process_single_model(filename, out_name)
       