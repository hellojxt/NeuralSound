import torch
from . import _linalg_utils as _utils


def matrix_norm(A):
    X = torch.randn(A.shape[0], 1, device = A.device)
    X_norm = float(torch.norm(X))
    iX_norm = X_norm ** -1
    A_norm = float(torch.norm(_utils.matmul(A, X))) * iX_norm
    return A_norm

def relative_error(A, X, B):
    e_norm = torch.norm(B - A @ X)
    b_norm = torch.norm(B)
    return e_norm / b_norm

def random_init(A):
    return torch.randn(A.shape[0], 1, device = A.device)

def variance(X):
    return (X**2).mean()

def vec_norm(X):
    return float(torch.norm(X))

