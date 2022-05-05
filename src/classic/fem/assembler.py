
from numba import njit
import numpy as np
from scipy.sparse import coo_matrix

def vertices(voxel, length):
    return _vertices(voxel) * length

def stiff_matrix(voxel, youngs, possion, length):
    values, rows, cols, m, n = _assemble_matrix(
        voxel,
        _stiff_element_matrix(youngs,possion,length)
    )
    return coo_matrix((values, (rows, cols)), shape=(m, n))

def mass_matrix(voxel, density, length):
    values, rows, cols, m, n = _assemble_matrix(
        voxel,
        _mass_element_matrix(density,length)
    )
    return coo_matrix((values, (rows, cols)), shape=(m, n))

@njit(parallel=True)
def _vertices(voxel):
    voxel = voxel.copy()
    res = voxel.shape[-1]
    vertex = np.zeros((res+1,res+1,res+1))
    coordinates = np.array([
        [0,0,0],[1,0,0],
        [1,1,0],[0,1,0],
        [0,0,1],[1,0,1],
        [1,1,1],[0,1,1]
    ])
    hex_num = 0
    for i in range(res):
        for j in range(res):
            for k in range(res):
                if voxel[i,j,k] == 1:
                    for c in coordinates:
                        vertex[i+c[0],j+c[1],k+c[2]] = 1
                    hex_num += 1
                    voxel[i,j,k] = hex_num

    vertex_num = 0
    for i in range(res+1):
        for j in range(res+1):
            for k in range(res+1):
                if vertex[i,j,k] == 1:
                    vertex_num += 1
    vs = np.zeros((vertex_num, 3))
    vertex_num = 0
    for i in range(res+1):
        for j in range(res+1):
            for k in range(res+1):
                if vertex[i,j,k] == 1:
                    vs[vertex_num] = np.array([i,j,k])
                    vertex_num += 1
    return (vs - res / 2) / res
    
@njit(parallel=True)
def _assemble_matrix(voxel, element_matrix):
    voxel = voxel.copy()
    res = voxel.shape[-1]
    vertex = np.zeros((res+1,res+1,res+1))
    coordinates = np.array([
        [0,0,0],[1,0,0],
        [1,1,0],[0,1,0],
        [0,0,1],[1,0,1],
        [1,1,1],[0,1,1]
    ])
    hex_num = 0
    for i in range(res):
        for j in range(res):
            for k in range(res):
                if voxel[i,j,k] == 1:
                    for c_idx in range(8):
                        c = coordinates[c_idx]
                        vertex[i+c[0],j+c[1],k+c[2]] = 1
                    hex_num += 1
                    voxel[i,j,k] = hex_num

    vertex_num = 0
    for i in range(res+1):
        for j in range(res+1):
            for k in range(res+1):
                if vertex[i,j,k] == 1:
                    vertex[i,j,k] = vertex_num
                    vertex_num += 1
    m = n = 3*vertex_num
    single_element_num = 0
    for i in range(24):
        for j in range(24):
            if element_matrix[i,j] != 0:
                single_element_num += 1

    element_num = hex_num*single_element_num
    values = np.zeros(element_num,dtype = np.float64)
    rows = np.zeros(element_num,dtype = np.int32)
    cols = np.zeros(element_num,dtype = np.int32)

    for i in range(res):
        for j in range(res):
            for k in range(res):
                if voxel[i,j,k] > 0:
                    idx = 0
                    offset = int(voxel[i,j,k] - 1)*single_element_num
                    for i_ in range(24):
                        for j_ in range(24):
                            if element_matrix[i_,j_] != 0:
                                row_i = i_ // 3
                                row_j = i_ % 3
                                c = coordinates[row_i]
                                v_id = vertex[i+c[0],j+c[1],k+c[2]]
                                rows[offset + idx] = v_id*3+row_j
                                col_i = j_ // 3
                                col_j = j_ % 3
                                c = coordinates[col_i]
                                v_id = vertex[i+c[0],j+c[1],k+c[2]]
                                cols[offset + idx] = v_id*3+col_j
                                values[offset + idx] = element_matrix[i_,j_]
                                idx += 1
    return values, rows, cols, m, n

def _mass_element_matrix(density, length):
    M = density*length*length*length/216*np.array([
        [4,4,2,4,4,2,1,2],
        [0,4,4,2,2,4,2,1],
        [0,0,4,4,1,2,4,2],
        [0,0,0,4,2,1,2,4],
        [0,0,0,0,4,4,2,4],
        [0,0,0,0,0,4,4,2],
        [0,0,0,0,0,0,4,4],
        [0,0,0,0,0,0,0,4]
    ])
    M = M + M.T
    M = M[...,np.newaxis,np.newaxis]*np.eye(3)
    M = np.transpose(M, [0,2,1,3])
    M = M.reshape(24,24)
    return M

def _stiff_element_matrix(youngs, possion, length):
    E = youngs
    nu = possion
    C = E / ((1+nu)*(1-2*nu))*np.array([
            [1-nu,nu,nu,0,0,0],[nu,1-nu,nu,0,0,0],
            [nu,nu,1-nu,0,0,0],[0,0,0,(1-2*nu)/2,0,0],
            [0,0,0,0,(1-2*nu)/2,0],[0,0,0,0,0,(1-2*nu)/2]
    ])
    GaussPoint = [-1/3**0.5, 1/3**0.5]
    coordinates = np.array([
        [-length/2,-length/2,-length/2],
        [length/2,-length/2,-length/2],
        [length/2,length/2,-length/2],
        [-length/2,length/2,-length/2],
        [-length/2,-length/2,length/2],
        [length/2,-length/2,length/2],
        [length/2,length/2,length/2],
        [-length/2,length/2,length/2]
    ])
    K = np.zeros((24,24))

    for xi1 in GaussPoint:
        for xi2 in GaussPoint:
            for xi3 in GaussPoint:
                dShape = (1/8)*np.array([
                            [-(1-xi2)*(1-xi3),(1-xi2)*(1-xi3),
                            (1+xi2)*(1-xi3),-(1+xi2)*(1-xi3),
                            -(1-xi2)*(1+xi3),(1-xi2)*(1+xi3),
                            (1+xi2)*(1+xi3),-(1+xi2)*(1+xi3)],
                            [-(1-xi1)*(1-xi3),-(1+xi1)*(1-xi3),
                            (1+xi1)*(1-xi3),(1-xi1)*(1-xi3),
                            -(1-xi1)*(1+xi3),-(1+xi1)*(1+xi3),
                            (1+xi1)*(1+xi3),(1-xi1)*(1+xi3)],
                            [-(1-xi1)*(1-xi2),-(1+xi1)*(1-xi2),
                            -(1+xi1)*(1+xi2),-(1-xi1)*(1+xi2),
                            (1-xi1)*(1-xi2),(1+xi1)*(1-xi2),
                            (1+xi1)*(1+xi2),(1-xi1)*(1+xi2)]
                ])
                JacobianMatrix = dShape @ coordinates
                auxiliar = np.linalg.inv(JacobianMatrix) @ dShape

                B = np.zeros((6,24))
                for i in range(3):
                    for j in range(8):
                        B[i,3*j + i] = auxiliar[i,j]

                for j in range(8):
                    B[3,3*j] = auxiliar[1,j]
                
                for j in range(8):
                    B[3,3*j+1] = auxiliar[0,j]
                
                for j in range(8):
                    B[4,3*j+2] = auxiliar[1,j]
                
                for j in range(8):
                    B[4,3*j+1] = auxiliar[2,j]
                
                for j in range(8):
                    B[5,3*j] = auxiliar[2,j]
                
                for j in range(8):
                    B[5,3*j+2] = auxiliar[0,j]
                
                K = K + B.T @ C @ B * np.linalg.det(JacobianMatrix)
    return K
