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
        print(phase, 'set size: ', len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = os.path.join(self.root_dir, self.file_list[index].replace('\n',''))
        data = torch.load(filename)
        coords = data['coords']
        edge = data['edge']
        return coords, edge.float(), filename
        
def lobpcg_collation_fn(datas):
    coords, edge, filename = list(zip(*datas))
    indice = 0
    ind_lst = [0]
    edge_shifted = []
    for e in edge:
        vert_num = e.max() + 1
        edge_shifted.append(e + indice)
        indice += vert_num
        ind_lst.append(int(indice))
    edge_shifted = torch.from_numpy(np.concatenate(edge_shifted)).long()
    return coords, edge_shifted, ind_lst, filename