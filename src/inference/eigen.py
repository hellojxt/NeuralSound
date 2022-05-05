import imp
from ..classic.lobpcg.utils import vec_norm
from .utils import load_weights
from ..classic.lobpcg import lobpcg
from ..classic.fem.util import scipy2torch, to_sparse_coords
from ..classic.fem.project_util import vert2vox, vox2vert, voxel_to_edge
from ..classic.tools import freq2mel, freq2val, val2freq
from ..net.eigennet import preconditionUnet, initUnet
from ..classic.fem.femModel import Material, Hexahedron_model
import torch
import numpy as np
from time import time
import MinkowskiEngine as ME
from os.path import join
torch.set_grad_enabled(False)

def get_norms(X, A, B):
    X_norm = float(torch.norm(X))
    iX_norm = X_norm ** -1
    A_norm = float(torch.norm(A @ X)) * iX_norm
    B_norm = float(torch.norm(B @ X)) * iX_norm
    return A_norm, B_norm

def rerr_fun(U, E, A, B):
    # check_shape(X, E)
    R = A @ U - B @ U * E
    A_norm, B_norm = get_norms(torch.rand_like(U), A, B)
    rerr = torch.norm(R, 2, (0, )) * (torch.norm(U, 2, (0, )) * (A_norm + E * B_norm)) ** -1
    return rerr.mean()

def get_svqb(U, A, B, n):
    tau = 1.2e-07
    UBU = U.T @ (B @ U)
    d = UBU.diagonal(0, -2, -1)
    nz = torch.where(abs(d) != 0.0)
    assert len(nz) == 1, nz
    if len(nz[0]) < len(d):
        U = U[:, nz[0]]
        UBU = U.T @ B(B @ U)
        d = UBU.diagonal(0, -2, -1)
        nz = torch.where(abs(d) != 0.0)
        assert len(nz[0]) == len(d)
    d_col = (d ** -0.5).reshape(d.shape[0], 1)
    DUBUD = (UBU * d_col) * d_col.T
    E, Z = torch.linalg.eigh(DUBUD, UPLO='U')
    t = tau * abs(E).max()
    # print(abs(E))
    keep = torch.where(E > t)
    assert len(keep) == 1, keep
    E = E[keep[0]]
    Z = Z[:, keep[0]]
    # check_shape(U, d_col, Z, E, (U * d_col.T), (Z * E ** -0.5))
    U = (U * d_col.T) @ (Z * E ** -0.5)
    UAU = U.T @ (A @ U)
    # print(UAU.shape)
    E, Z = torch.linalg.eigh(UAU, UPLO='U')
    return U @ Z[:, :n ], E[:n]

m_min = freq2mel(20)
m_max = freq2mel(20000)
def freqError(freq, freq_):
    m, m_ = freq2mel(freq), freq2mel(freq_)
    err = (m - m_)**2 / (m_max - m_min)**2
    return err.mean()
    
class Solver():
    def __init__(self):
        self.timecost = []
        self.rerr = []
        self.f_rerr = []
        self.name = ''

    def log_out(self):
        print(self.name)
        print(f'time: {np.mean(self.timecost[2:]):.3f}, \
                err: {np.mean(self.rerr[2:]):.5f}, \
                f_err: {np.mean(self.f_rerr[2:]):.4f}')


class Net():
    NETWORK = None
    WEIGHTS_DIR = None
    def __init__(self, tag, device = torch.device('cuda')):
        if tag is None:
            PATH = None
        else:
            PATH = join(self.WEIGHTS_DIR, f'{tag}.pt')
        self.net = load_weights(
            self.NETWORK(24, 24, linear = True).to(device),
            PATH,
        )
        print(f'{self.NETWORK.__name__} network initialized')
        self.device = device
    
    def update_voxel(self, voxel):
        self.coords = to_sparse_coords(voxel)
        self.edge = torch.from_numpy(voxel_to_edge(voxel)).to(self.device)

    def NetX(self, X):
        vert_num = X.shape[0]//3
        n = X.shape[1]
        t = time()
        feats_in = vert2vox(X.reshape(vert_num, 3*n), self.edge)
        voxel_num = feats_in.shape[0]
        feats_in = feats_in.reshape(voxel_num, 24, n).permute(2, 0, 1).reshape(n*voxel_num, 24)
        bcoords = ME.utils.batched_coordinates([self.coords]*n).to(X.device)
        sparse_tensor = ME.SparseTensor(feats_in, bcoords)
        output = self.net(sparse_tensor)
        # print(time() - t)
        output = output.F.reshape(n, voxel_num, 24).permute(1, 2, 0).reshape(voxel_num, 24*n)
        ret = vox2vert(output, self.edge, 'mean').reshape(vert_num*3, n)
        return ret

import cupy as cp
import cupyx
import cupyx.scipy.sparse.linalg
from cupyx.scipy.sparse.linalg import cg, gmres, spilu, splu
from cupyx.scipy.sparse import coo_matrix

class RRInit():
    def __init__(self, K, J) -> None:
        self.torch = False
        self.K = K
        self.J = J

    def update_AB(self, A, B):
        self.A = coo_matrix(A)
        self.B = coo_matrix(B)
        self.A_torch = scipy2torch(A, 'cuda').double()
        self.B_torch = scipy2torch(B, 'cuda').double()

    def __call__(self, X_):
        X_gpu = cp.asarray(X_[:, :self.K])
        A_inv = splu(self.A)
        xs = torch.zeros(X_gpu.shape[0], 0, device=X_.device)       
        x = X_gpu
        for i in range(self.J):
            x = A_inv.solve(self.B @ x)
            xs = torch.cat((xs, torch.as_tensor(x, device='cuda')), dim=1)
            # print(xs.shape)
        # print(xs.shape)
        X, E = get_svqb(xs, self.A_torch, self.B_torch, X_.shape[-1])
        # print(X.shape)
        if X.shape[-1] < X_.shape[-1]:
            r = torch.randn((X_.shape[0], X_.shape[-1] - X.shape[-1]), dtype=X_.dtype, device=X_.device)
            X = torch.cat((X, r), dim=1)
        return X.double()

class NetInit(Net):
    NETWORK = initUnet
    WEIGHTS_DIR  = '/data1/deepEigen/eigen_init/weights'
    def __init__(self, tag, device = torch.device('cuda')):
        self.torch = True
        super().__init__(tag, device=device)

    def update_voxel_AB(self, voxel, A, B):
        self.update_voxel(voxel)
        self.A, self.B = A, B

    def __call__(self, X):
        X1 = self.NetX(X.float())
        X, E = get_svqb(torch.cat((X, X1), dim=1), self.A, self.B, X.shape[-1])
        # rerr(X, E, self.A, self.B)
        # print(E**0.5 / (2*np.pi))
        return X.double()


class NetPrecondition(Net):
    NETWORK = preconditionUnet
    WEIGHTS_DIR  = '/data1/deepEigen/eigen_precondition/weights'
    def __init__(self, tag, device=torch.device('cuda')):
        super().__init__(tag, device=device)

    def __call__(self, X):
        return self.NetX(X)


class LanczosFast(Solver):
    def __init__(self, inner = 'ILU'):
        super().__init__()
        self.inner = inner
        self.name = 'Lanczos'

    def solve(self, stiff_matrix, mass_matrix, k, voxel, reference, tol, niter):
        from scipy.sparse.linalg import eigsh, ArpackNoConvergence
        from scipy.sparse.linalg import LinearOperator


        A = stiff_matrix.astype(np.float64)
        B = mass_matrix.astype(np.float64)
        m = A.shape[-1]

        start_time = time()
        sigma = 0
        A_gpu = coo_matrix(A)
        if self.inner == 'ILU':
            A_inv = spilu(A_gpu, drop_tol=tol, fill_factor=niter)
        end_time = time()
        start_time2 = time()
        # print('LU time:', end_time - start_time)
        def OPinv_fun(x_cpu):
            x_gpu = cp.asarray(x_cpu)
            if self.inner == 'ILU':
                x_gpu = A_inv.solve(x_gpu)
            else:
                x_gpu,_ = gmres(A_gpu, x_gpu, maxiter=niter)
            return x_gpu.get()

        OPinv = LinearOperator(A.shape, matvec=OPinv_fun)
        try:
            vals_, vecs_ =  eigsh(A = A, M = B, which='LM', sigma = sigma, k=k, 
                                maxiter=1, 
                                OPinv = OPinv, 
                                tol=1e-5)
        except ArpackNoConvergence as e:
            vals_ = e.eigenvalues
            vecs_ = e.eigenvectors

        end_time = time()
        # print('solve time:', end_time - start_time2)

        # print(vals_.shape, vecs_.shape)

        vals = reference
        freqs, freqs_ = val2freq(vals.cpu().numpy()), val2freq(vals_)
        freqs = np.nan_to_num(freqs)
        freqs_ = np.nan_to_num(freqs_)
        if len(freqs_) < len(freqs):
            freqs = freqs[:len(freqs_)]
        
        rerr = rerr_fun(torch.from_numpy(vecs_), 
                        torch.from_numpy(vals_),
                        scipy2torch(A),
                        scipy2torch(B))
        
        # print(freqs_, freqs)
        # print(freqError(freqs, freqs_))
        # print(rerr)

        self.f_rerr.append(freqError(freqs, freqs_))
        self.vals = vals_
        self.vecs = vecs_
        self.timecost.append(end_time - start_time)
        self.rerr.append(rerr.mean().item())

    def log_out(self):
        print(self.name)
        print(f'time: {np.mean(self.timecost[2:]):.3f}, \
                err: {np.mean(self.rerr[2:])}, \
                f_err: {np.mean(self.f_rerr[2:])}')


class ModifiedLOBPCG(Solver):
    def __init__(self, init = None, precondition = None, name = ''):
        super().__init__()
        self.init = init
        self.precondition = precondition
        self.name = name

    def solve(self, stiff_matrix, mass_matrix, k, voxel, reference = None, **kwargs):
        A = scipy2torch(stiff_matrix, 'cuda').double()
        B = scipy2torch(mass_matrix, 'cuda').double()
        m = A.shape[-1]
        if self.precondition is not None:
            self.precondition.update_voxel(voxel)
        if self.init is not None:
            if self.init.torch:
                self.init.update_voxel_AB(voxel, A, B)
            else:
                self.init.update_AB(stiff_matrix, mass_matrix)

        start_time = time()
        X = torch.randn((m, k), dtype=A.dtype, device=A.device)
        if self.init is not None:
            X = self.init(X)
        vals_, vecs_ , rerr = lobpcg(A, k, B, X=X,
                            iK=self.precondition, 
                            largest=False, return_rerr=True, **kwargs)
        end_time = time()
        if reference is not None:
            vals = reference
            freqs, freqs_ = val2freq(vals.cpu().numpy()), val2freq(vals_.cpu().numpy())
            freqs = np.nan_to_num(freqs)
            freqs_ = np.nan_to_num(freqs_)
            self.f_rerr.append(freqError(freqs, freqs_))

        self.vals = vals_
        self.vecs = vecs_
        self.timecost.append(end_time - start_time)
        self.rerr.append(rerr.mean().item())


    
class ModifiedLOBPCGReference(ModifiedLOBPCG):
    def __init__(self, init=None, precondition=None):
        super().__init__(init, precondition)
        self.name = 'LOBPCG_REF'
    
    def solve(self, stiff_matrix, mass_matrix, k, voxel, reference=None, **kwargs):
        A = scipy2torch(stiff_matrix, 'cuda').double()
        B = scipy2torch(mass_matrix, 'cuda').double()
        m = A.shape[-1]
        start_time = time()
        X = torch.randn((m, k), dtype=A.dtype, device=A.device)
        vals_, vecs_ , rerr = lobpcg(A, k, B, X=X,
                            iK=self.precondition, 
                            largest=False, return_rerr=True, **kwargs)
        end_time = time()
        self.f_rerr.append(0)
        self.vals = vals_
        self.vecs = vecs_
        self.timecost.append(end_time - start_time)
        self.rerr.append(rerr.mean().item())
    
    def log_out(self):
        print(self.name)
        print(f'time: {np.mean(self.timecost[2:]):.3f}, \
                err: {np.mean(self.rerr[2:])}, \
                f_err: {np.mean(self.f_rerr[2:]):.4f}')

