import numpy as np
# boundary_encode
cube_normal = np.array([
    [ 0, 0,  1],
    [ 0, 0, -1],
    [ 0, 1,  0],
    [ 0,-1,  0],
    [ 1, 0,  0],
    [-1, 0,  0],
])
def random_rotate_mirror(coords, feats_in, feats_out, surface_code):
    mirror_params = [[0,0],[1,0],[0,1],[1,1]][np.random.randint(0,4)]
    if mirror_params[0] == 1:
        # phi = 2pi - phi
        # print('phi,1 mirror')
        coords[:,1] = 31-coords[:,1]
        feats_in[:,1] = -feats_in[:,1]
        feats_out = np.flip(feats_out, 1)
        s2, s3 = surface_code[:, 2].copy(), surface_code[:, 3].copy()
        surface_code[:, 2], surface_code[:, 3] = s3, s2

    if mirror_params[1] == 1:
        # theta = pi - theta
        # print('theta,2 mirror')
        coords[:,2] = 31-coords[:,2]
        feats_in[:,2] = -feats_in[:,2]
        feats_out = np.flip(feats_out, 2)
        s0, s1 = surface_code[:, 0].copy(), surface_code[:, 1].copy()
        surface_code[:, 0], surface_code[:, 1] = s1, s0
    def t_c(c, k):
        k = np.rint(k)
        if k < 0:
            return 31 - c
        return c*k
    rotate_params = np.random.randint(0,4)
    s_idx_lst = np.array([
        [0, 1, 2, 3],
        [2, 3, 1, 0],
        [1, 0, 3, 2],
        [3, 2, 0, 1]
    ])
    s = [surface_code[:, 2].copy(), surface_code[:, 3].copy(), surface_code[:, 4].copy(), surface_code[:, 5].copy()]
    si = s_idx_lst[rotate_params]
    surface_code[:, 2], surface_code[:, 3], surface_code[:, 4], surface_code[:, 5] = s[si[0]], s[si[1]], s[si[2]], s[si[3]]

    cos_t,sin_t = np.cos(rotate_params*np.pi/2), np.sin(rotate_params*np.pi/2)
    c0, c1 = t_c(coords[:,0],cos_t) + t_c(coords[:,1],-sin_t), t_c(coords[:,1], cos_t) + t_c(coords[:,0],sin_t)
    coords[:,0], coords[:,1] = c0, c1
    f0, f1 = feats_in[:,0]*cos_t - feats_in[:,1]*sin_t, feats_in[:,1]*cos_t + feats_in[:,0]*sin_t
    feats_in[:,0], feats_in[:,1] = f0, f1
    size_phi = feats_out.shape[1]
    feats_out = np.roll(feats_out, size_phi // 4 * rotate_params, axis=1)

    return coords, feats_in, feats_out, surface_code