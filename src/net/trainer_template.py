import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

class Config():
    CustomDataset = None
    custom_collation_fn = None
    net = None
    optimizer = None
    scheduler = None
    forward_fun = None
    loss_fun = None
    BATCH_SIZE = 16
    batch_num_limit = None
    dataset_root_dir = ''
    dataset_worker_num = 0
    tag = 'default'
    device = torch.device('cuda:0')
    epoch_idx = 0
    idx_in_epoch = 0
    phase = ''
    writer = None
    best_loss = float('inf')
    weights_dir = 'weights'
    load_weights = False
    only_test = False

def check_shape(*lst):
    for i in lst:
        print(i.shape, end=' ')
    print('\n')

def forward_and_backward(loader):
    total_loss = []
    loss_item = {}
    Config.idx_in_epoch = 0
    print(f'Phase: {Config.phase}')
    for items_ in tqdm(loader):
        items = []
        loss_sum = 0
        for item in items_:
            if isinstance(item, torch.Tensor):
                item = item.to(Config.device, non_blocking=True)
            items.append(item)
        out = Config.forward_fun(items)
        if isinstance(out, tuple):
            loss = Config.loss_fun(*out)
        else:
            loss = Config.loss_fun(out)
        if isinstance(loss, dict):
            if not loss_item:
                for k in loss.keys():
                    loss_item[k] = []
            for k in loss.keys():
                loss_item[k].append(loss[k].item())
                loss_sum += loss[k]
        else:
            loss_sum = loss
        if torch.is_grad_enabled():
            Config.optimizer.zero_grad()
            loss_sum.backward()
            Config.optimizer.step()
        total_loss.append(loss_sum.item())
        Config.idx_in_epoch += 1
        if Config.batch_num_limit is not None and Config.idx_in_epoch == Config.batch_num_limit:
            break
    
    print(f'total loss: {np.mean(total_loss)}')
    writer.add_scalar(f'total_{Config.phase}', np.mean(total_loss), Config.epoch_idx)
    if loss_item:
        for k in loss_item.keys():
            writer.add_scalar(f'{k}_{Config.phase}', np.mean(loss_item[k]), Config.epoch_idx)
            print(f'{k} loss: {np.mean(loss_item[k])}')

    if Config.phase == 'valid' and  np.mean(total_loss) < Config.best_loss:
        Config.best_loss = np.mean(total_loss)
        os.makedirs(Config.weights_dir, exist_ok=True)
        torch.save(Config.net.state_dict(), f'{Config.weights_dir}/{Config.tag}.pt')
        print('Saved weights update') 
    
def get_loader(phase):
    dataset = Config.CustomDataset(Config.dataset_root_dir, phase)
    if len(dataset) < Config.BATCH_SIZE:
        drop_last = False
    else:
        drop_last = True
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                        collate_fn=Config.custom_collation_fn, drop_last=drop_last, num_workers=Config.dataset_worker_num)
    return loader

def train():
    Config.net.train()
    Config.phase = 'train'
    forward_and_backward(loaders['train'])

def valid_and_test():
    with torch.set_grad_enabled(False):
        Config.net.eval()
        if not Config.only_test:
            Config.phase = 'valid'
            forward_and_backward(loaders['valid'])
        Config.phase = 'test'
        forward_and_backward(loaders['test'])

def start_train(max_epoch):
    phases = ['train', 'valid', 'test']
    global loaders, writer
    loaders = {phase:get_loader(phase) for phase in phases}
    Config.net.to(Config.device)
    os.system(f'rm runs/{Config.tag}/*')
    writer = SummaryWriter(f'runs/{Config.tag}')
    Config.writer = writer
    if Config.load_weights:
        PATH = f'{Config.weights_dir}/{Config.tag}.pt'
        Config.net.load_state_dict(torch.load(PATH))
        
    for epoch in range(max_epoch):
        Config.epoch_idx = epoch
        print(f'Epoch:{epoch}')
        if not Config.only_test:
            Config.phase = 'train'
            train()
        valid_and_test()
        Config.scheduler.step()
    
    writer.close()