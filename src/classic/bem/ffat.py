from time import time
from .bemModel import boundary_mesh
from .util import voxel2boundary, unit_sphere_surface_points, obj_to_grid
import numpy as np
import os
import bempp.api as api
from tqdm import tqdm

SPEED_OF_SOUND = 343
AIR_DENSITY = 1.225

def potential_compute(boundary_model, points_list, image_size):
    data_list = [boundary_model.points_dirichlet(ps.reshape(-1,3)) for ps in points_list]
    data_sum = 0
    for data, points in zip(data_list, points_list):
        points = points.reshape(-1, 3)
        r = (points**2).sum(-1)**0.5
        data = np.abs(data)*r
        data_sum = data + data_sum
    data = data_sum / len(data_list)
    return data.reshape(2*image_size, image_size)

def vibration_to_ffat(voxel_coords, voxel_vib, freqs, length = 0.15, image_size = 32, showTimeStatistics = False):
    voxel_res = 32
    vertices, elements, feats_index = map(np.asarray, voxel2boundary(voxel_coords, voxel_res))
    vertices = (vertices / voxel_res - 0.5)*length
    grid = obj_to_grid(vertices, elements)
    boundary_model = boundary_mesh(grid)
    
    voxel_num = len(voxel_coords)
    map_num = len(freqs)
    ffat_map = np.zeros((map_num, 2*image_size, image_size))
    ffat_map_far = np.zeros((map_num, 2*image_size, image_size))
    feats_in_reshaped = voxel_vib.reshape(voxel_num, 3, -1)
    
    points_ = unit_sphere_surface_points(image_size)*length
    points_list = [points_*1.25, points_*1.25**2, points_*1.25**3]
    points_list_far = [points_*3, points_*3**2, points_*3**3]
    # print('Start BEM for ffat map')
    time_lst = []
    for i in tqdm(range(map_num)):
        start_time = time()
        feats_select = feats_in_reshaped[:,:,i]
        omega = 2*np.pi*freqs[i]
        wave_number = omega / SPEED_OF_SOUND
        dis = (grid.normals * feats_select[feats_index]).sum(-1)
        neumann_coeff = AIR_DENSITY*dis
        neumann_fun = api.GridFunction(boundary_model.dp0_space, coefficients=np.asarray(neumann_coeff))
        boundary_model.set_wave_number(wave_number)
        boundary_model.set_neumann_fun(neumann_fun)
        boundary_model.preprocess_layer()
        boundary_model.ext_neumann2dirichlet()
        ffat_map[i] = potential_compute(boundary_model, points_list, image_size)
        ffat_map_far[i] = potential_compute(boundary_model, points_list_far, image_size)
        time_lst.append(time() - start_time)
    if showTimeStatistics:
        print(time_lst)
    return ffat_map, ffat_map_far


def ffat_to_imgs(ffat_map, output_dir, tag = ''):
    '''
        ffat_map: [n, size1, size2]
    '''
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)
    for idx, data in enumerate(ffat_map):
        data = data/data.max() * 255
        Image.fromarray(np.uint8(data), 'L').save(f'{output_dir}/{tag}{idx}.jpg')