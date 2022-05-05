from glob import glob 
from tqdm import tqdm
import sys
import os
import random

if __name__ == '__main__':
    root_dir = sys.argv[1]
    os.chdir(root_dir)
    file_list = glob('*/*.npz')
    random.shuffle(file_list)
    length = len(file_list)
    idx = [0, length*0.2, length*0.2 + length*0.8*0.2, length]
    phase = ['test', 'valid', 'train']
    for i in range(3):
        with open(f'{phase[i]}.txt','w') as f:
            lst = file_list[int(idx[i]):int(idx[i+1])]
            for line in lst:
                f.write(line + '\n')

