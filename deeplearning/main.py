from utils import get_dataloader, get_logger, train_loop, val_loop, setup_seed
from monai.transforms import *
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler
from monai.inferers import SimpleInferer
from monai.losses import DeepSupervisionLoss
import logging
from model import D_BUS_Net
import sys
import torch
import os
import time

def main(epoch, batch_size, log_path, seed=0, ckpt_path = None):

    setup_seed(seed)

    train_transform = Compose([
        LoadImaged(keys=['bus', 'cdfi'], ensure_channel_first=True, reader='PILReader', converter=lambda x: x.convert('RGB')),
        ScaleIntensityd(keys=['bus', 'cdfi']),
        Rand2DElasticd(keys=['bus', 'cdfi'], prob=0.5, spacing=(20, 20), magnitude_range=(1, 2), mode=('bilinear', 'bilinear'), padding_mode='zeros'),
        RandRotated(keys=['bus', 'cdfi'], prob=0.5, range_x=(0.4, 0.4), mode=('bilinear', 'bilinear')),
        RandAffined(keys=['bus', 'cdfi'], prob=0.5, shear_range=(0.5, 0.5), mode=('bilinear', 'bilinear'), padding_mode='zeros'),
        RandAxisFlipd(keys=['bus', 'cdfi'], prob=0.5),
        RandShiftIntensityd(keys=['bus', 'cdfi'], offsets=0.1, prob=0.5),
        Resized(keys=['bus', 'cdfi'], spatial_size=(224, 224), mode=('bilinear', 'bilinear'))
    ])
    val_transform = Compose([
        LoadImaged(keys=['bus', 'cdfi'], ensure_channel_first=True, reader='PILReader', converter=lambda x: x.convert('RGB')),
        ScaleIntensityd(keys=['bus', 'cdfi']),
        Resized(keys=['bus', 'cdfi'], spatial_size=(224, 224), mode=('bilinear', 'bilinear'))
    ])

    batch_size = batch_size
    PC_data_root = '../data/origin/PC'
    VC_data_root = '../data/origin/VC'
    train_dataloader = get_dataloader(data_root=PC_data_root, transform=train_transform, batch_size=batch_size)
    val_dataloader = get_dataloader(data_root=VC_data_root, transform=val_transform, batch_size=batch_size)

    device = torch.device('cuda:0')
    model = D_BUS_Net(fuse_att=True, cdfi_att=True, dropout=0.2).to("cuda:0")
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(device)
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = 10, min_lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-7)
    epoch = epoch

    log_path = os.path.join(log_path, time.strftime("%Y%m%d%H%M%S", time.localtime()))
    os.makedirs(os.path.join(log_path, 'checkpoint'), exist_ok=True)
    logger = get_logger(os.path.join(log_path, 'training.log'))

    best_acc = 0
    for e in range(epoch):
        train_loss, train_acc = train_loop(e, model, train_dataloader, optimizer, loss_func, logger, device)
        val_loss, val_acc  = val_loop(e, model, val_dataloader, loss_func, logger, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_path, 'checkpoint', 'best_model.pth'))
        scheduler.step(val_loss)
    torch.save(model.state_dict(), os.path.join(log_path, 'checkpoint', 'latest.pth'))

if __name__ == '__main__':
    log_path = './log/wi_both_att_drop0.2'
    ckpt_path = None
    epoch = 500
    batch_size = 32
    seed = 424
    main(epoch, batch_size, log_path, seed, ckpt_path)