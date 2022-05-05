import numpy as np
import os
from .fem.femModel import Hexahedron_model, Material
from .fem.solver import LOBPCG_solver, Lanczos_Solver
from .fem.util import to_sparse_coords
# from .bem.ffat import vibration_to_ffat
from .mesh.loader import ObjLoader

class MelConfig():
    mel_min, mel_max = 1000, 3500
    mel_res = 10
    mel_spacing = (mel_max - mel_min) / mel_res

def mel2norm(mel):
    return (mel - MelConfig.mel_min) / (MelConfig.mel_max - MelConfig.mel_min)

def norm2mel(n):
    return n*(MelConfig.mel_max - MelConfig.mel_min) + MelConfig.mel_min

def mel_index(mel):
    idx = int((mel- MelConfig.mel_min)/MelConfig.mel_spacing)
    if idx >= MelConfig.mel_res:
        idx = MelConfig.mel_res - 1
    if idx <= 0:
        idx = 0
    return idx

def index2mel(idx):
    return MelConfig.mel_min + (idx + 0.5)*MelConfig.mel_spacing

def mel2freq(mel):
    return 700*(10**(mel/2595) - 1)

def freq2mel(f):
    return np.log10(f/700 + 1)*2595

def freq2val(f):
    return (f*(2*np.pi))**2

def val2freq(v):
    return v**0.5/(2*np.pi)

def val2mel(v):
    return freq2mel(val2freq(v))

def mel2val(mel):
    return freq2val(mel2freq(mel))
    
def voxelize_mesh(mesh_name, res = 32):
    from .voxelize.hexahedral import Hexa_model
    mesh = ObjLoader(mesh_name)
    mesh.normalize()
    hexa = Hexa_model(mesh.vertices, mesh.faces, res=32)
    hexa.fill_shell()
    return hexa.voxel_grid

def modal_analysis(voxel, mat = Material.Ceramic, k = 20):
    model = Hexahedron_model(voxel,mat=mat)
    
    model.modal_analysis(LOBPCG_solver(k=k))
    return model.vecs, model.vals




