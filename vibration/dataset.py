import torch
from torch.utils.data import Dataset
import numpy as np
import MinkowskiEngine as ME
import os

class LobpcgDataset(Dataset):
    def __init__(self, root_dir, phase):
        with open(f'{root_dir}/{phase}.txt', 'r') as f:
            self.file_list = f.readlines()
        self.root_dir = root_dir
        print(len(self.file_list))
        self.skip = 20

    def __len__(self):
        return len(self.file_list) // self.skip

    def __getitem__(self, index):
        index = index * self.skip + torch.randint(0, self.skip, (1,)).item()
        # print(index)
        filename = os.path.join(self.root_dir, self.file_list[index].replace('\n',''))
        data = torch.load(filename)
        zero_vec_data = torch.from_numpy(
            np.load(filename.replace('lobpcgData', 'zeroEigenvectorData'))['vecs']
        )
        coords = data['coords']
        # A, B = data['A'], data['B']
        edge = data['edge']
        # vox2vert, vert2vox = data['vox2vert'], data['vert2vox']
        return coords, edge.float(), zero_vec_data.float(), filename
        
def lobpcg_collation_fn(datas):
    coords, edge, zero_vec_data, filename = list(zip(*datas))
    indice = 0
    ind_lst = [0]
    edge_shifted = []
    for e in edge:
        vert_num = e.max() + 1
        edge_shifted.append(e + indice)
        indice += vert_num
        ind_lst.append(int(indice))
    edge_shifted = torch.from_numpy(np.concatenate(edge_shifted)).long()
    return coords, edge_shifted, zero_vec_data, ind_lst, filename