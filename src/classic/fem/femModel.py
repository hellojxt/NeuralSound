
import numpy as np
# import torch

class Material(object):
    # ρ E ν α β
    # 0 1 2 3 4
    Ceramic = 2700,7.2E10,0.19,6,1E-7
    Glass = 2600,6.2E10,0.20,1,1E-7
    Wood = 750,1.1E10,0.25,60,2E-6
    Plastic = 1070,1.4E9,0.35,30,1E-6
    Iron = 8000,2.1E11,0.28,5,1E-7
    Polycarbonate = 1190,2.4E9,0.37,0.5,4E-7
    Steel = 7850,2.0E11,0.29, 5 ,3E-8
    Tin = 7265, 5e10, 0.325, 2 ,3E-8

    def omega_rate(src, target, rescale):
        k1 = target[1] / src[1]
        k2 = target[0] / src[0]
        return k1**0.5*k2**(-0.5)/rescale

    def vibration_rate(src, target, rescale):
        k1 = target[1] / src[1]
        k2 = target[0] / src[0]
        return k2**(-0.5)*rescale**(-3/2)

    def ffat_map_rate(rescale):
        return rescale**3

def random_material():
    mat_lst = [
        Material.Ceramic, Material.Glass, Material.Wood, Material.Plastic,
        Material.Iron, Material.Polycarbonate, Material.Steel, Material.Tin
    ]
    return mat_lst[np.random.randint(len(mat_lst))]
    
class Hexahedron_model(object):
    def __init__(self, voxel = None, mat = Material.Plastic, length = 0.15, res = 32):
        self.density, self.youngs, self.poison, self.alpha, self.beta = mat
        self.length = length
        if voxel is not None:
            self.voxel = np.asarray(voxel)
            self.res = voxel.shape[-1]
        else:
            self.voxel = None
            self.res = res
        self.mass_matrix_ = None
        self.stiff_matrix_ = None
        self.vertices_ = None

    @property
    def lumped_mass_matrix(self):
        M = self.mass_matrix.sum(0)
        return np.asarray(M).reshape(-1)
        
    @property
    def mass_matrix(self):
        from . import assembler
        if self.mass_matrix_ is None:
            self.mass_matrix_ = assembler.mass_matrix(self.voxel, self.density, self.length / self.res)
        return self.mass_matrix_
    
    @property
    def stiff_matrix(self):
        from . import assembler
        if self.stiff_matrix_ is None:
            self.stiff_matrix_ = assembler.stiff_matrix(self.voxel, self.youngs, self.poison, self.length / self.res)
        return self.stiff_matrix_

    @property
    def _element_mass_matrix(self):
        from . import assembler
        return assembler._mass_element_matrix(self.density, self.length / self.res)

    @property
    def _element_stiff_matrix(self):
        from . import assembler
        return assembler._stiff_element_matrix(self.youngs, self.poison, self.length / self.res)

    @property
    def vertices(self):
        if self.vertices_ is None:
            from . import assembler
            self.vertices_ = assembler.vertices(self.voxel, self.length)
        return self.vertices_

    # save obj file from vertices
    def save_obj(self, filename):
        vertices = self.vertices
        with open(filename, 'w') as f:
            f.write('# OBJ file\n')
            for v in vertices:
                f.write('v {} {} {}\n'.format(*v))

    def modal_analysis(self, solver):
        self.vals, self.vecs = solver(self.stiff_matrix, self.mass_matrix)

    @property
    def damps(self):
        return 0.5*(self.alpha + self.beta*self.vals)

    @property
    def freqs(self):
        d = self.damps
        mid_term = self.vals - d**2
        mid_term[mid_term < 0] = 0
        freqs = mid_term**0.5/(2*np.pi)
        return freqs

    def mask_hearing_range(self, min_freq = 20, max_freq = 20000):
        freqs = np.nan_to_num(self.freqs)
        mask = ((freqs >= min_freq) & (freqs <= max_freq))
        self.vals = self.vals[mask]
        self.vecs = self.vecs[:,mask]
    
    def amplitudes_after_click(self, idx, direction):
        print(idx, direction)
        vertex_num = self.vecs.shape[0]//3
        mode_num = self.vecs.shape[-1]
        vecs = self.vecs.reshape(vertex_num, 3, mode_num)
        vec_ = vecs[idx].T
        g = vec_ @ direction
        amp =  g / (self.freqs * 2 * np.pi)
        return amp

    def amplitudes_after_multiple_clicks(self, idxs, directions):
        vertex_num = self.vecs.shape[0]//3
        mode_num = self.vecs.shape[-1]
        vecs = self.vecs.reshape(vertex_num, 3, mode_num)
        vec_ = np.transpose(vecs[idxs], (0, 2, 1))
        g = np.matmul(vec_, directions.reshape(-1, 3, 1)).reshape(-1, mode_num)
        amps =  g / (self.freqs.reshape(1, mode_num) * 2 * np.pi)
        return amps

    def M_norm(self, V):
        if len(V.shape) == 2:
            return V / (V.T @ self.mass_matrix @ V).diagonal()**0.5
        else:
            return V / (V.T @ self.mass_matrix @ V)**0.5

    def get_M_orth(self, U, V):
        return U - V @ (V.T @ self.mass_matrix @ U)

    @property
    def zero_eigenvalue_vector(self):
        V = np.zeros((self.mass_matrix.shape[0], 6)).reshape(-1, 3, 6)
        vertices = self.vertices # [vetice_num, 3]
        V[:,0,0] = 1
        V[:,1,1] = 1
        V[:,2,2] = 1
        V[:,0,3] = -vertices[:,1]
        V[:,1,3] = vertices[:,0]
        V[:,0,4] = -vertices[:,2]
        V[:,2,4] = vertices[:,0]        
        V[:,1,5] = -vertices[:,2]
        V[:,2,5] = vertices[:,1]
        V = V.reshape(-1, 6)
        V[:,:3] = self.M_norm(V[:,:3])
        for i in range(3,6):
            V[:,i] = self.M_norm(self.get_M_orth(V[:,i], V[:,:i]))
        return V

    def eigenvector_guess(self, channel):
        x = np.random.rand(self.vecs.shape[0], channel)
        x0 = self.zero_eigenvalue_vector
        x = x - ((x.T @ self.mass_matrix @ x0) * x0.reshape(-1,1,6)).sum(-1)
        return x
    
    def eigenvector_random_guess(self, channel):
        x = np.random.rand(self.vecs.shape[0], channel)
        return x

    def eigenvector_random_guess(self, channel):
        x = np.random.rand(self.vecs.shape[0], channel)
        return x

    # def zero_eigenvalue_vector_cuda(self, device = torch.device('cuda:0')):
    #     def M_norm(V):
    #         if len(V.shape) == 2:
    #             return V / (V.T @ (mass_matrix @ V)).diagonal()**0.5
    #         else:
    #             return V / (V.T @ (mass_matrix @ V))**0.5

    #     def get_M_orth(U, V):
    #         return U - V @ (V.T @ (mass_matrix @ U))
    #     from .util import scipy2torch
    #     mass_matrix = scipy2torch(self.mass_matrix, device)
    #     V = torch.zeros((mass_matrix.shape[0], 6)).reshape(-1, 3, 6).to(device).double()
    #     vertices = torch.from_numpy(self.vertices).to(device)
    #     V[:,0,0] = 1
    #     V[:,1,1] = 1
    #     V[:,2,2] = 1
    #     V[:,0,3] = -vertices[:,1]
    #     V[:,1,3] = vertices[:,0]
    #     V[:,0,4] = -vertices[:,2]
    #     V[:,2,4] = vertices[:,0]        
    #     V[:,1,5] = -vertices[:,2]
    #     V[:,2,5] = vertices[:,1]
    #     V = V.reshape(-1, 6)
    #     V[:,:3] = M_norm(V[:,:3])
    #     for i in range(3,6):
    #         V[:,i] = M_norm(get_M_orth(V[:,i], V[:,:i]))
    #     return V

    # def eigenvector_guess_cuda(self, channel, device = torch.device('cuda:0')):
    #     x = torch.rand(self.vecs.shape[0], channel).to(device).double()
    #     x0 = self.zero_eigenvalue_vector_cuda(device).double()
    #     from .util import scipy2torch
    #     mass_matrix = scipy2torch(self.mass_matrix, device)
    #     x = x - ((x.T @ (mass_matrix @ x0)) * x0.reshape(-1,1,6)).sum(-1)
    #     return x
    