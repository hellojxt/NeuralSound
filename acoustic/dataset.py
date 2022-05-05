import torch
from torch.utils.data import Dataset
import numpy as np
import MinkowskiEngine as ME
import os
import sys
sys.path.append('..')

from src.classic.bem.data_aug import random_rotate_mirror
def freq2mel(f):
    return np.log10(f/700 + 1)*2595

mel_max = freq2mel(20000)
mel_min = freq2mel(20)

def freq2wave_number(f):
    return 2*np.pi*f/343

wn_max = freq2wave_number(20000)
wn_min = freq2wave_number(20)

class AcousticDataset(Dataset):
    SUFFIX = ''
    def __init__(self, root_dir, phase):
        with open(f'{root_dir}/{phase}.txt', 'r') as f:
            self.file_list = f.readlines()
        self.root_dir = root_dir
        print(len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = os.path.join(self.root_dir, self.file_list[index].replace('\n',''))
        data = np.load(filename)
        coords, feats_in, feats_out, surface_code, freqs = data['coords'], data['feats_in'], data['feats_out' + self.SUFFIX], data['surface'], data['freqs']
        # print('feats_out' + self.SUFFIX)
        # Random selection
        random_idx = np.random.randint(len(freqs))
        freq = freqs[random_idx]
        feats_in = feats_in[:,:,random_idx]
        feats_out = feats_out[random_idx][np.newaxis, ...]

        coords = coords - 1
        # print(coords.max(0), coords.min(0))
        voxel_num = coords.shape[0]
        # print(coords.shape, feats_in.shape, feats_out.shape, surface_code.shape)
        
        '''(2308, 3) (2308, 3) (1, 64, 32) (2308, 6)'''
        coords, feats_in, feats_out, surface_code = random_rotate_mirror(coords, feats_in, feats_out, surface_code)

        # normalize
        coords_feats = (coords/16 - 1)/0.5
        feats_in_norm = (feats_in**2).mean()**0.5
        feats_in = feats_in / feats_in_norm
 
        wave_number = freq2wave_number(freq)
        freq_norm = torch.ones(voxel_num, 1) * (wave_number - wn_min) / (wn_max - wn_min)
        voxel_size = 0.15/32
        sin_cos_item = torch.tensor(
            [
                np.cos(wave_number * voxel_size / 4), np.sin(wave_number * voxel_size / 4),
                np.cos(wave_number * voxel_size / 2), np.sin(wave_number * voxel_size / 2),
                np.cos(wave_number * voxel_size / 1), np.sin(wave_number * voxel_size / 1),
                np.cos(wave_number * voxel_size * 2), np.sin(wave_number * voxel_size * 2),
                np.cos(wave_number * voxel_size * 4), np.sin(wave_number * voxel_size * 4),
                # np.cos(wave_number * voxel_size * 8), np.sin(wave_number * voxel_size * 8),
                # np.cos(wave_number * voxel_size * 16), np.sin(wave_number * voxel_size * 16),
            ]
        )
        sin_cos = torch.ones(voxel_num, sin_cos_item.shape[-1])*sin_cos_item

        feats_in = np.concatenate([feats_in, surface_code, freq_norm, sin_cos, coords_feats], axis=1)
        feats_out = feats_out / feats_in_norm
        feats_out_norm = (feats_out**2).mean()**0.5
        feats_out = feats_out / feats_out_norm
        feats_out_norm = (np.log(feats_out_norm) + 8)/3

        return coords, feats_in, feats_out, feats_out_norm, filename
        
class AcousticDatasetFar(AcousticDataset):
    SUFFIX = '_far'
    
def acoustic_collation_fn(datas):
    coords, feats_in, feats_out, feats_out_norm, filename = list(zip(*datas))
    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)
    # Concatenate all lists
    feats_in = torch.from_numpy(np.concatenate(feats_in, axis=0)).float()
    feats_out = torch.from_numpy(np.stack(feats_out, axis=0)).float()
    # freq_norm = torch.tensor(freq_norm).float().unsqueeze(-1)
    feats_out_norm = torch.tensor(feats_out_norm).float().unsqueeze(-1)
    return bcoords, feats_in, feats_out, feats_out_norm, filename