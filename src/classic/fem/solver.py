

def Lanczos_Solver(k = 30, **kw):
    from scipy.sparse.linalg import eigsh
    def solver(stiff_matrix, mass_matrix):
        sigma = 0
        return eigsh(A = stiff_matrix, **kw, M = mass_matrix, which='LM', sigma = sigma, k=k)
    return solver

def LOBPCG_solver(k = 20):
    k = k + 6
    from torch import lobpcg
    from .util import scipy2torch
    def solver(stiff_matrix, mass_matrix):
        def tracker(e):
            print(e.R)
            print(e.ivars['istep'])
        A, B = scipy2torch(stiff_matrix, 'cuda').float(), scipy2torch(mass_matrix, 'cuda').float()
        vals, vecs = lobpcg(A, k, B, tracker=None, largest=False)
        return vals.cpu().numpy()[6:], vecs.cpu().numpy()[:,6:]
    return solver
