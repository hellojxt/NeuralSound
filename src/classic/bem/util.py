import numpy as np
from numba import njit

@njit()
def unit_cube_surface_points(res):
    face_datas = np.array([
        [ 0, 0,  1],[ 0, 1, 0],[ 1, 0, 0],
        [ 0, 0, -1],[ 0, 1, 0],[ 1, 0, 0],
        [ 0, 1,  0],[ 1, 0, 0],[ 0, 0, 1],
        [ 0,-1,  0],[ 1, 0, 0],[ 0, 0, 1],
        [ 1, 0,  0],[ 0, 1, 0],[ 0, 0, 1],
        [-1, 0,  0],[ 0, 1, 0],[ 0, 0, 1],
    ]).reshape(-1, 3, 3)
    points = np.zeros((6, res, res, 3))
    for face_idx, data in enumerate(face_datas):
        normal, w, h = data
        dw, dh = w/(res-1), h/(res-1)
        p0 = 0.5*normal - 0.5*w - 0.5*h
        for i in range(res):
            for j in range(res):
                points[face_idx, i, j] = p0 + i*dw + j*dh
    return points

@njit()
def unit_sphere_surface_points(res):
    # r = 0.5
    points = np.zeros((2*res, res, 3))
    phi_spacing = 2*np.pi / (res*2-1)
    theta_spacing = np.pi / (res-1)
    for i in range(2*res):
        for j in range(res):
            phi = phi_spacing * i
            theta = theta_spacing * j
            x = np.sin(theta)*np.cos(phi)
            y = np.sin(theta)*np.sin(phi)
            z = np.cos(theta)
            points[i,j] = [x,y,z]
    return points*0.5

def obj_to_grid(vertices, elements):
    import bempp.api
    vertices = np.asarray(vertices)
    elements = np.asarray(elements)
    return bempp.api.Grid(vertices.T.astype(np.float64), 
                        elements.T.astype(np.uint32))


cube_vertex = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])
            

cube_normal = np.array([
    [ 0, 0,  1],
    [ 0, 0, -1],
    [ 0, 1,  0],
    [ 0,-1,  0],
    [ 1, 0,  0],
    [-1, 0,  0],
])
cube_faces = np.array([
    [1, 7, 5], 
    [1, 3, 7], 
    [1, 4, 3], 
    [1, 2, 4], 
    [3, 8, 7], 
    [3, 4, 8], 
    [5, 7, 8], 
    [5, 8, 6], 
    [1, 5, 6], 
    [1, 6, 2], 
    [2, 6, 8], 
    [2, 8, 4], 
]) - 1

cube_faces_normal_index = np.array([2,2,6,6,3,3,5,5,4,4,1,1]) - 1

@njit()
def boundary_voxel(coords, resolution = 32):
    ds = np.array([
        [1,0,0],[-1,0,0],
        [0,1,0],[0,-1,0],
        [0,0,1],[0,0,-1]
    ])
    coords_surface = []
    feats_index = []
    coords = coords + 1
    res = resolution + 2
    voxel = np.zeros((res, res, res))
    voxel[0,0,0] = 1
    coords_n = len(coords)
    for c_idx in range(coords_n):
        c = coords[c_idx]
        voxel[c[0],c[1],c[2]] = 2
    points = [np.array([0,0,0])]
    while len(points) > 0:
        p = points.pop()
        for d_idx in range(6):
            d = ds[d_idx]
            p_ = p + d
            if (p_ >= 0).all() and (p_ < res).all() and voxel[p_[0],p_[1],p_[2]] != 1:
                if voxel[p_[0],p_[1],p_[2]] == 0:
                    points.append(p_)
                voxel[p_[0],p_[1],p_[2]] = 1
    for c_idx in range(coords_n):
        c = coords[c_idx]
        if voxel[c[0],c[1],c[2]] == 1:
            coords_surface.append(c - 1)
            feats_index.append(c_idx)
    return coords_surface, feats_index
    

@njit()
def voxel2boundary(coords, resolution = 32):
    ds = np.array([
        [1,0,0],[-1,0,0],
        [0,1,0],[0,-1,0],
        [0,0,1],[0,0,-1]
    ])
    coords = coords + 1
    res = resolution + 2
    voxel = np.zeros((res, res, res))
    voxel[0,0,0] = 1
    coords_n = len(coords)
    for c_idx in range(coords_n):
        c = coords[c_idx]
        voxel[c[0],c[1],c[2]] = 2
    points = [np.array([0,0,0])]
    while len(points) > 0:
        p = points.pop()
        for d_idx in range(6):
            d = ds[d_idx]
            p_ = p + d
            if (p_ >= 0).all() and (p_ < res).all() and voxel[p_[0],p_[1],p_[2]] != 1:
                if voxel[p_[0],p_[1],p_[2]] == 0:
                    points.append(p_)
                voxel[p_[0],p_[1],p_[2]] = 1
    for c_idx in range(coords_n):
        c = coords[c_idx]
        voxel[c[0],c[1],c[2]] = 2
    vertex_flag = np.zeros((res,res,res)) - 1

    vertices = []
    elements = []
    feats_index = []
    for i in range(len(coords)):
        for j in range(len(cube_faces)):
            c = coords[i]
            face = cube_faces[j]
            normal = cube_normal[cube_faces_normal_index[j]]
            c_ = normal + c
            if voxel[c_[0],c_[1],c_[2]] == 1:
                element = []
                for vertex_idx in face:
                    v =  cube_vertex[vertex_idx] + c
                    if vertex_flag[v[0],v[1],v[2]] == -1:
                        vertex_flag[v[0],v[1],v[2]] = len(vertices)
                        vertices.append(v - 1)
                    element.append(vertex_flag[v[0],v[1],v[2]])
                elements.append(element)
                feats_index.append(i)
    return vertices, elements, feats_index



@njit()
def boundary_encode(coords, resolution = 32):
    ds = np.array([
        [1,0,0],[-1,0,0],
        [0,1,0],[0,-1,0],
        [0,0,1],[0,0,-1]
    ])
    coords = coords + 1
    res = resolution + 2
    voxel = np.zeros((res, res, res))
    voxel[0,0,0] = 1
    coords_n = len(coords)
    for c_idx in range(coords_n):
        c = coords[c_idx]
        voxel[c[0],c[1],c[2]] = 2
    points = [np.array([0,0,0])]
    while len(points) > 0:
        p = points.pop()
        for d_idx in range(6):
            d = ds[d_idx]
            p_ = p + d
            if (p_ >= 0).all() and (p_ < res).all() and voxel[p_[0],p_[1],p_[2]] != 1:
                if voxel[p_[0],p_[1],p_[2]] == 0:
                    points.append(p_)
                voxel[p_[0],p_[1],p_[2]] = 1
    for c_idx in range(coords_n):
        c = coords[c_idx]
        voxel[c[0],c[1],c[2]] = 2
    h_vector = []
    for i in range(len(coords)):
        c = coords[i]
        code = np.zeros(6)
        for j in range(len(cube_normal)):
            normal = cube_normal[j]
            c_ = normal + c
            if voxel[c_[0],c_[1],c_[2]] == 1:
                code[j] = 1
        h_vector.append(code)
    return h_vector
