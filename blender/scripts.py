import bpy
from bpy import context
import sys
sys.path.append('..')
from src.classic.fem.femModel import *
from glob import glob
import numpy as np
from numba import njit
from utils import *
from src.classic.mesh.loader import ObjLoader

camera = bpy.data.objects['Camera']

@njit()
def _IIR(f, e, theta, r, wwd):
    signal = np.zeros_like(f)
    for idx in range(len(signal)):
        if idx < 3: 
            continue
        signal[idx] = 2*e*np.cos(theta)*signal[idx - 1] - \
                    e**2*signal[idx - 2] + \
                    2*f[idx-1]*(e*np.cos(theta+r)-e**2*np.cos(2*theta + r))/(3*wwd)
    return signal

def IIR(f, val, alpha, beta, h):
    d = 0.5*(alpha + beta*val)
    e = np.exp(-d*h)
    wd = (val - d**2)**0.5
    theta = wd*h
    w = (val**0.5)
    r =  np.arcsin(d / w)
    return _IIR(f, e, theta, r, w*wd)

@njit()
def ffat_value_lst(ffat_map, xs, ys, rs):
    frame_num = len(xs)
    ffat_vs = np.zeros(int(BITRATE*frame_num/fps))
    for i in range(len(ffat_vs)):
        idx_float = i / (BITRATE / fps)
        i1 = int(idx_float)
        i2 = i1 + 1
        if i2 == frame_num:
            i2 = frame_num - 1
        t = idx_float - i1
        r = rs[i1]*(1-t) + rs[i2]*t
        v1 = ffat_map[xs[i1], ys[i1]]
        v2 = ffat_map[xs[i2], ys[i2]]
        ffat_vs[i] = (v1*(1-t) + v2*t)/r
    return ffat_vs

# load object with position and rotation into blender
def load_object():
    print('load object')
    if 'mesh' in bpy.data.objects:
        return bpy.data.objects['mesh']

    bpy.ops.import_scene.obj(filepath=dirname + '/mesh.obj')
    obj = bpy.context.selected_objects[-1]
    data = np.load(dirname + '/motion.npz')
    pos, ori, physical_step = data['pos'], data['ori'], data['step']
    physical_fps = 1 / physical_step
    physical_frame_num = len(pos)
    time_all = physical_frame_num / physical_fps
    animation_frame_num = int(time_all * fps)

    for frame_idx in range(animation_frame_num):
        print(frame_idx)
        bpy.context.scene.frame_set(frame_idx+1)
        physical_frame_idx = int(frame_idx * physical_fps / fps)
        obj.location = pos[physical_frame_idx]
        # transform ori from x, y, z, w order to w, x, y, z order
        ori[physical_frame_idx] = ori[physical_frame_idx][[3, 0, 1, 2]]
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = ori[physical_frame_idx]
        # add a keyframe with locatopn and rotation
        obj.keyframe_insert(data_path='location', frame=frame_idx+1)
        obj.keyframe_insert(data_path='rotation_quaternion', frame=frame_idx+1)
    return obj

def save_camera_motion():
    print('save camera motion')
    xs = []
    ys = []
    rs = []
    res = 32
    phi_spacing = 2*np.pi / (res*2-1)
    theta_spacing = np.pi / (res-1)
    _, _, step = get_motion()
    contact_info = get_contact_info()
    max_frame = int(len(contact_info) * step) * fps
    for frame_idx in range(max_frame):
        bpy.context.scene.frame_set(frame_idx+1)
        pos = camera.matrix_world.translation
        camera_loc = np.array([pos.x, pos.y, pos.z, 1])
        matrix_world = np.array(obj.matrix_world)
        x, y, z, _ = np.linalg.inv(matrix_world) @ camera_loc
        r = (x**2 + y**2 + z**2)**0.5
        theta = np.arccos(z/r)
        phi = np.arctan2(y, x)
        # print(int(phi // phi_spacing), int(theta // theta_spacing))
        xs.append(int(phi // phi_spacing))
        ys.append(int(theta // theta_spacing))
        rs.append(r)
    np.savez_compressed(dirname + '/camera_motion', xs = xs, ys = ys ,rs = rs)

from scipy.io.wavfile import write

BITRATE = 44100
def synthesis_sound():
    print('synthesis sound')
    freqs = get_freqs()
    vals = get_vals() / 3
    _, ffat_map = get_ffat_map()
    xs, ys, rs = get_camera_motion()
    _, _, step = get_motion()
    contact_info = get_contact_info()
    print('contact info shape')
    print(contact_info.shape)
    print('freqs, ffat_map shape')
    print(freqs.shape, ffat_map.shape)

    mode_num = len(vals)
    frame_num = len(contact_info)
    total_time = frame_num * step
    
    signal = np.zeros(int(BITRATE*total_time))
    for mode_idx in range(mode_num):
        if mode_idx < 6:
            continue
        f = contact_info[:, mode_idx]
        mode_ffat_map = ffat_map[mode_idx]
        mode_ffat_mask = ffat_value_lst(mode_ffat_map, xs, ys, rs)
        alpha, beta = material[3], material[4]
        mode_sound = IIR(f, vals[mode_idx], alpha, beta, 1/BITRATE)
        # check shape of mode_ffat_mask and mode_sound 
        print(mode_ffat_mask.shape, mode_sound.shape)
        plot_and_save(f, f'f_{mode_idx}')
        plot_and_save(mode_ffat_mask, f'ffat_mask_{mode_idx}')
        plot_and_save(mode_sound, f'sound_{mode_idx}')
        signal_mode = mode_sound*mode_ffat_mask
        signal += signal_mode
        signal_mode = signal_mode / np.abs(signal_mode).max()
        # np.save(f'signal_{mode_idx}', signal_mode)
        write(dirname + f'/sound_{mode_idx}.wav', BITRATE, signal_mode.astype(np.float32))

    plot_and_save(signal, f'sound')
    signal = signal / np.abs(signal).max()
    
    write(dirname + '/sound.wav', BITRATE, signal.astype(np.float32))
    scene = context.scene 
    if not scene.sequence_editor:
        scene.sequence_editor_create()  
    # remove all existing audio sequences
    for seq in scene.sequence_editor.sequences_all:
        if seq.type == 'SOUND':
            scene.sequence_editor.sequences.remove(seq)

    scene.sequence_editor.sequences.new_sound("physically based sound synthesis", 
                                                         dirname + "/sound.wav", 3, 1)
# get bounding box of vertices
def get_bbox(vertices):
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])
    return x_min, x_max, y_min, y_max, z_min, z_max
    
def solve_contact_info():
    print('solve contact info')
    model = Hexahedron_model(get_voxel(), mat=material,
                             length=ObjLoader(get_mesh()).get_length())
    from scipy.spatial import KDTree
    kdtree = KDTree(model.vertices)
    pos, normal, force = get_contact()
    _, _, step = get_motion()
    print(get_bbox(model.vertices))
    print(get_bbox(pos[:,0,:]))
    
    vecs = get_vecs()
    frame_idx_ = 0
    bpy.context.scene.frame_set(frame_idx_+1)
    contact_info = np.zeros((len(pos), n))
    for i, p in enumerate(pos):
        f_mode = np.zeros(n)
        frame_idx = int(i * step * 24)
        if frame_idx > frame_idx_:
            frame_idx_ = frame_idx
            print(frame_idx)
            bpy.context.scene.frame_set(frame_idx_+1)
        for contact_idx in range(4):
            p_i = p[contact_idx]
            camera_loc = np.array([p_i[0], p_i[1], p_i[2], 1])
            matrix_world = np.array(obj.matrix_world)
            x, y, z, _ = np.linalg.inv(matrix_world) @ camera_loc
            idx = kdtree.query([[x,y,z]], k=1)[1][0]
            f = vecs.reshape(-1, 3, n)[idx]
            f_mode += f[0]*normal[i, contact_idx][0] * force[i, contact_idx]
            f_mode += f[1]*normal[i, contact_idx][1] * force[i, contact_idx]
            f_mode += f[2]*normal[i, contact_idx][2] * force[i, contact_idx]
        contact_info[i] = f_mode
    np.save(dirname + '/contact_info.npy', contact_info)

import os
if __name__ == '__main__':
    fps = 24
    obj = load_object()
    solve_contact_info()
    save_camera_motion()
    synthesis_sound()
    
    # video = glob(dirname + '/video/*.avi')
    # print(video)
    # os.system(f'ffmpeg -i {video} -i {dirname}/sound.wav -map 0:v -map 1:a -c:v copy -shortest media.mp4')

    