import sys
sys.path.append('..')
from src.classic.fem.femModel import *
import configparser
cf = configparser.ConfigParser()

cf.read("config.ini") 
dirname = cf.get('default', 'dir')
n = int(cf.get('default', 'n'))
material = getattr(Material, cf.get('default', 'material'))
solver_name = cf.get('default', 'solver')

# plot a 1D signal using pyplot and save as image
def plot_and_save(signal, filename):
    import matplotlib.pyplot as plt
    # clear all content in the figure
    plt.clf()
    plt.plot(signal)
    plt.savefig(dirname + f'/{filename}.png')
    
def get_mesh():
    return dirname + '/mesh.obj'

def get_contact_info():
    return np.load(dirname + '/contact_info.npy')

def get_vecs():
    filename = dirname + '/' + solver_name + '.npz'
    data = np.load(filename)
    return data['vecs']

def get_voxel():
    return np.load(dirname + '/voxel.npy')

def get_vals():
    return np.load(dirname + '/' + solver_name + '.npz')['vals']

def get_freqs():
    model = Hexahedron_model(get_voxel(), mat=material)
    model.vals = get_vals()
    return model.freqs

def get_contact():
    filename = dirname + '/contact.npz'
    data = np.load(filename)
    return data['pos'], data['normal'], data['force']

def get_motion():
    data = np.load(dirname + '/motion.npz')
    return data['pos'], data['ori'], data['step']

def get_ffat_map():
    data = np.load(dirname + '/ffat_map.npz')
    return data['near'], data['far']

def get_camera_motion():
    camera_data = np.load(dirname + '/camera_motion.npz')
    xs, ys, rs = camera_data['xs'], camera_data['ys'], camera_data['rs']
    return xs, ys, rs