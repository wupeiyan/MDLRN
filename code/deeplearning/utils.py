import os
import os.path as osp
import glob
import logging
import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
from model import D_BUS_Net
import random
from torch import nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def get_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log = logging.FileHandler(filename, 'a', encoding='utf-8')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s ')
    log.setFormatter(formatter)
    logger.addHandler(log)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    return logger


def get_dataloader(data_root:str = 'data/origin/PC', 
                   bus_dir:str = 'bus', 
                   cdfi_dir:str = 'cdfi', 
                   suffix:str = '.png',
                   transform=None,
                   batch_size:int = 32, num_workers:int = 4, shuffle:bool = True):
    
    bus = glob.glob(osp.join(data_root, bus_dir, '*' + suffix))
    cdfi = glob.glob(osp.join(data_root, cdfi_dir, '*' + suffix))
    label = [1 if 'po' in osp.basename(i) else 0 for i in bus]
       
    data = [{'bus': i[0], 'cdfi': i[1], 'label': i[2]} for i in zip(bus, cdfi, label)]
    dataset = Dataset(data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader



def train_loop(epoch, model, train_loader, optimizer, loss_func, logger=None, device='cpu'):
    model.train()
    pbar = tqdm(train_loader)
    pbar.set_description("[training {}]".format(epoch))
    post_trans = AsDiscrete(threshold=0.5)

    losses = 0
    out_acc = 0
    bus_acc = 0
    cdfi_acc = 0
    
    for batch in pbar:
        bus = batch['bus'].to(device)
        cdfi = batch['cdfi'].to(device)
        label = batch['label'].to(device).unsqueeze(1).float()
        batch_size = torch.numel(label)

        optimizer.zero_grad()
        pred_out, pred_bus, pred_cdfi = model(bus, cdfi)

        out_loss = loss_func(pred_out, label)
        bus_loss = loss_func(pred_bus, label)
        cdfi_loss = loss_func(pred_cdfi, label)
        loss = out_loss + bus_loss + cdfi_loss
        loss.backward()
        optimizer.step()

        losses += loss.item() / batch_size

        pred_out = post_trans(pred_out)
        pred_bus = post_trans(pred_bus) 
        pred_cdfi = post_trans(pred_cdfi) 

        out_acc += (pred_out == label.data).sum().item() / batch_size
        bus_acc += (pred_bus == label.data).sum().item() / batch_size
        cdfi_acc += (pred_cdfi == label.data).sum().item() / batch_size

    losses = losses / len(train_loader)
    out_acc = out_acc / len(train_loader)
    bus_acc = bus_acc / len(train_loader)
    cdfi_acc = cdfi_acc / len(train_loader)
    if logger:
        logger.info(
            '[train_epoch:{}] loss:{:.4f} out_acc:{:.4f} bus_acc:{:.4f} cdfi_acc:{:.4f}'.format(epoch, losses, out_acc, bus_acc, cdfi_acc))
    return losses, out_acc

def val_loop(epoch, model, val_loader, loss_func, logger=None, device='cpu'):
    model.eval()
    pbar = tqdm(val_loader)
    pbar.set_description("[start val]")
    post_trans = AsDiscrete(threshold=0.5)

    losses = 0
    out_acc = 0
    bus_acc = 0
    cdfi_acc = 0

    with torch.no_grad():
        for batch in pbar:
            bus = batch['bus'].to(device)
            cdfi = batch['cdfi'].to(device)
            label = batch['label'].to(device).unsqueeze(1).float()
            batch_size = torch.numel(label)

            pred_out, pred_bus, pred_cdfi = model(bus, cdfi)
            out_loss = loss_func(pred_out, label)
            bus_loss = loss_func(pred_bus, label)
            cdfi_loss = loss_func(pred_cdfi, label)
            loss = out_loss + bus_loss + cdfi_loss

            losses += loss.item() / batch_size

            pred_out = post_trans(pred_out)
            pred_bus = post_trans(pred_bus) 
            pred_cdfi = post_trans(pred_cdfi) 
            out_acc += (pred_out == label.data).sum().item() / batch_size
            bus_acc += (pred_bus == label.data).sum().item() / batch_size
            cdfi_acc += (pred_cdfi == label.data).sum().item() / batch_size

        losses = losses / len(val_loader)
        out_acc = out_acc / len(val_loader)
        bus_acc = bus_acc / len(val_loader)
        cdfi_acc = cdfi_acc / len(val_loader)

        if logger:
            logger.info(
                '[val_epoch:{}] loss:{:.4f} out_acc:{:.4f} bus_acc:{:.4f} cdfi_acc:{:.4f}'.format(epoch, losses, out_acc, bus_acc, cdfi_acc))

        return losses, out_acc


def deep_score(model, data_loader, save_path, device='cpu'):
    model.eval()
    pbar = tqdm(data_loader)
    pbar.set_description("[start feature extraction]")


    names = []
    deep_features = []

    with torch.no_grad():
        for batch in pbar:
            bus = batch['bus'].to(device)
            cdfi = batch['cdfi'].to(device)
            label = batch['label'].to(device).unsqueeze(1).float()
            batch_size = torch.numel(label)

            pred_out, pred_bus, pred_cdfi = model(bus, cdfi)

            names.extend(batch['bus_meta_dict']['filename_or_obj'])
            deep_features.extend(*pred_out.cpu().numpy().tolist())
    names = [osp.basename(i).split('.')[0] for i in names]
    deep_score_df = pd.DataFrame({'names': names, 'deep_score': deep_features})
    deep_score_df.to_csv(save_path, index=False)

def plot_log(filepath):
    history = pd.read_csv(filepath, sep=' ', header=None).iloc[:, 5:-1]
    history.columns = ['epoch', 'loss', 'out_acc', 'bus_acc', 'cdfi_acc']
    history['mode'] = history['epoch'].str.slice(1, 2)
    history['epoch'] = history['epoch'].str.extract('(\d+)')
    history['loss'] = history['loss'].str.extract('(\d+.\d+)').astype('float')
    history['out_acc'] = history['out_acc'].str.extract('(\d+.\d+)').astype('float')
    history['bus_acc'] = history['bus_acc'].str.extract('(\d+.\d+)').astype('float')
    history['cdfi_acc'] = history['cdfi_acc'].str.extract('(\d+.\d+)').astype('float')

    train = history[history['mode'] == 't']
    val = history[history['mode'] == 'v']
    plt.figure(figsize=(12, 12))
    train.plot(x='epoch', y='loss', ax=plt.subplot(2, 2, 1), title='train loss')
    train.plot(x='epoch', y=['out_acc', 'bus_acc', 'cdfi_acc'], ax=plt.subplot(2, 2, 2), title='train acc')
    val.plot(x='epoch', y='loss', ax=plt.subplot(2, 2, 3), title='val loss')
    val.plot(x='epoch', y=['out_acc', 'bus_acc', 'cdfi_acc'], ax=plt.subplot(2, 2, 4), title='val acc')
    plt.savefig(os.path.join(os.path.dirname(filepath), 'training_log.png'))
    print('-'*20+'train')
    print(train.sort_values(by='out_acc', ascending=False).iloc[0, :])
    print('-'*20+'val')
    print(val.sort_values(by='out_acc', ascending=False).iloc[0, :])


def extract_features(ckpt_path, device):
    assert osp.exists(ckpt_path), 'ckpt_path not exists'
    transform = Compose([
        LoadImaged(keys=['bus', 'cdfi'], ensure_channel_first=True, reader='PILReader', converter=lambda x: x.convert('RGB')),
        ScaleIntensityd(keys=['bus', 'cdfi']),
        Resized(keys=['bus', 'cdfi'], spatial_size=(224, 224), mode=('bilinear', 'bilinear'))
    ])

    batch_size = 1
    PC_data_root = 'data/origin/PC'
    VC_data_root = 'data/origin/VC'
    TC1_data_root = 'data/origin/TC1'
    TC2_data_root = 'data/origin/TC2'
    PC_dataloader = get_dataloader(data_root=PC_data_root, transform=transform, batch_size=batch_size, shuffle=False)
    VC_dataloader = get_dataloader(data_root=VC_data_root, transform=transform, batch_size=batch_size, shuffle=False)
    TC1_dataloader = get_dataloader(data_root=TC1_data_root, transform=transform, batch_size=batch_size, shuffle=False)
    TC2_dataloader = get_dataloader(data_root=TC2_data_root, transform=transform, batch_size=batch_size, shuffle=False)

    model = D_BUS_Net(fuse_att=True, cdfi_att=True, dropout=0.2)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.cls = nn.Sequential(*list(model.cls.children())[:-1])
    model.to(device)
    
    deep_score(model, PC_dataloader, 'code/form/deeplearning/pc.csv', device)
    deep_score(model, VC_dataloader, 'code/form/deeplearning/vc.csv', device)
    deep_score(model, TC1_dataloader, 'code/form/deeplearning/tc1.csv', device)
    deep_score(model, TC2_dataloader, 'code/form/deeplearning/tc2.csv', device)

if __name__ == '__main__':
    # plot_log('deeplearning/log/wi_both_att_drop0.2/20230608115005/training.log')
    ckpt_path = 'deeplearning/log/wi_both_att_drop0.2/20230608115005/checkpoint/best_model.pth'
    device = torch.device('cuda:0')
    extract_features(ckpt_path, device)