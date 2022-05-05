import numpy as np
from ..classic.fem.util import to_sparse_coords, vertex_to_voxel_matrix
from ..classic.bem.ffat import ffat_to_imgs, vibration_to_ffat
from ..classic.bem.util import boundary_encode, boundary_voxel
from ..classic.tools import freq2mel, mel2freq, index2mel, MelConfig
from ..classic.fem.femModel import Material, random_material

def bempp_solve(voxel, vecs, freqs):
    coords = to_sparse_coords(voxel)
    ffat_map, ffat_map_far = vibration_to_ffat(coords, vecs, freqs)
    return ffat_map, ffat_map_far