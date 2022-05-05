from glob import glob 
from tqdm import tqdm
import sys
import os
import random

if __name__ == '__main__':
    file_list_str = sys.argv[1]
    file_sample = glob(file_list_str)[0]
    root_dir = os.path.dirname(file_sample)
    file_list = glob(file_list_str)
    random.shuffle(file_list)
    length = len(file_list)
    idx = [0, length*0.2, length*0.2 + length*0.8*0.2, length]
    phase = ['test', 'valid', 'train']
    for i in range(3):
        with open(os.path.join(root_dir, f'{phase[i]}.txt'),'w') as f:
            lst = file_list[int(idx[i]):int(idx[i+1])]
            for line in lst:
                f.write(os.path.basename(line) + '\n')

