import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import os
from numba import njit
gcc_version = int(os.popen('gcc --version').read().split('\n')[0].split()[-1][0])
if gcc_version < 7:
    mod = SourceModule(open(os.path.dirname(__file__) + '/hexahedral.cu').read(), options=["-ccbin","gcc-7"])
else:
    mod = SourceModule(open(os.path.dirname(__file__) + '/hexahedral.cu').read())
cuda_voxelizer = mod.get_function("voxelize")

class Hexa_model():
    def __init__(self, vertices, triangles, res = 32):
        dest = np.zeros(res*res*res,dtype= np.int32)
        cuda_voxelizer(
            drv.Out(dest), np.int32(res), drv.In(vertices.astype(np.float32)),
            drv.In(triangles.astype(np.int32)), np.int32(len(triangles)),
            block=(res,res,1), grid=(res,1)
            )
        self.voxel_grid = dest.reshape(res,res,res)
        self.res = res

    @njit()
    def fill_shell_(voxel, res):
        points = [np.array([0,0,0])]
        ds = np.array([
            [1,0,0],[-1,0,0],
            [0,1,0],[0,-1,0],
            [0,0,1],[0,0,-1]
        ])
        while len(points) > 0:
            p = points.pop()
            for i in range(6):
                d = ds[i]
                p_ = p + d
                if (p_ >= 0).all() and (p_ < res).all() and voxel[p_[0],p_[1],p_[2]] != 1:
                    if voxel[p_[0],p_[1],p_[2]] == 0:
                        points.append(p_)
                    voxel[p_[0],p_[1],p_[2]] = 1
        voxel = voxel[1:res-1, 1:res-1, 1:res-1]
        return voxel
    
    def fill_shell(self):
        voxel = np.pad(self.voxel_grid, 1)*2
        res = self.res + 2
        voxel = Hexa_model.fill_shell_(voxel, res)
        self.voxel_grid[voxel == 0] = 1



