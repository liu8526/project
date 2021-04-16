import torchcontrib
import os
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from glob import glob
from PIL import Image

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pytorch_toolbelt import losses as L
from utils.utils import AverageMeter, second2time, inial_logger
from albumentations.augmentations import functional as F
from .metric import IOUMetric
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp

Image.MAX_IMAGE_PIXELS = 1000000000000000

# 网络
class seg_model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.model_name = 'timm-tf_efficientnet_lite4'#timm-tf_efficientnet_lite4
        self.model_name = 'efficientnet-b7'
        # self.model_name = 'timm-efficientnet-b7'
        self.model = smp.UnetPlusPlus(          # UnetPlusPlus/DeepLabV3Plus
                encoder_name=self.model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=4,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=10,                # model output channels (number of classes in your dataset)
            )

    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def train_net(param, model, train_data, valid_data, device=[0]):
    # 初始化参数
    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    lr              = param['lr']
    gamma           = param['gamma']
    step_size       = param['step_size']
    momentum        = param['momentum']
    weight_decay    = param['weight_decay']

    disp_inter      = param['disp_inter']
    save_inter      = param['save_inter']
    min_inter       = param['min_inter']
    iter_inter      = param['iter_inter']

    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']
    
    if not os.path.exists(save_log_dir): os.makedirs(save_log_dir)
    if not os.path.exists(save_ckpt_dir): os.makedirs(save_ckpt_dir)

    # 网络参数
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()

    c, y, x = train_data.__getitem__(0)['image'].shape
    # print('c, y, x:', c, y, x)
    
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=lr ,weight_decay=weight_decay)
    SWA_opt = torchcontrib.optim.SWA(optimizer, swa_start= ((epochs-10) if (epochs-10) > 0 else 0), swa_freq=1, swa_lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(SWA_opt, T_0=10, T_mult=2, eta_min=1e-5, last_epoch=-1) #学习率自适应调整
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(SWA_opt, gamma=gamma)

    # criterion = L.JointLoss(L.SoftCrossEntropyLoss(smooth_factor=0.1), L.DiceLoss(mode='multiclass'), 0.5, 0.5)#, L.LovaszLoss(),L.FocalLoss(),  0.5, 0.5, 0.2, 0.1).cuda()
    criterion = L.JointLoss(L.SoftCrossEntropyLoss(smooth_factor=0.1), L.DiceLoss(mode='multiclass'), 0.5, 0.5).cuda()
    logger = inial_logger(os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) +'_'+model_name+ '.log'))

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_iou = 0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        
        # state_dict=ckpt['state_dict']
        # print(state_dict['module.model.segmentation_head.0.weight'],state_dict.keys())
        # state_dict['module.model.segmentation_head.0.weight'] = state_dict['module.model.segmentation_head.0.weight'].resize_(2,16,3,3)
        # state_dict['module.model.segmentation_head.0.bias'] = state_dict['module.model.segmentation_head.0.bias'].resize_(2)
            
        best_iou = ckpt['best_iou']
        epoch_start = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer'])

    logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size, valid_data_size))
    
    # Writer will output to ./runs/ directory by default
    summary_dir = "save/summary_dir"
    if not os.path.exists(summary_dir): os.makedirs(summary_dir)
    writer = SummaryWriter(summary_dir)

    scaler = GradScaler()
    for epoch in range(epoch_start, epochs):
        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to('cuda')), Variable(target.to('cuda'))
            SWA_opt.zero_grad()
            with autocast():
                pred = model(data)
                loss = criterion(pred, target)
            # Scales loss. 为了梯度放大.
            scaler.scale(loss).backward()
            # scaler.step() 首先把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            scaler.step(SWA_opt)
            # 准备着，看是否要增大scaler
            scaler.update()
            scheduler.step(epoch + batch_idx / train_loader_size) #学习率自适应调整
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            # 日志
            if batch_idx % iter_inter == 0:
                writer.add_scalar('Loss', train_iter_loss.avg, train_loader_size*epoch + batch_idx)
                writer.add_scalar('Lr', SWA_opt.param_groups[-1]['lr'], train_loader_size*epoch + batch_idx)
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                SWA_opt.param_groups[-1]['lr'],
                train_iter_loss.avg,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()
        # scheduler.step()

        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        iou=IOUMetric(10)
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target = batch_samples['image'], batch_samples['label']
                data, target = Variable(data.to('cuda')), Variable(target.to('cuda'))
                with autocast():
                    pred = model(data)
                    loss = criterion(pred, target)
                pred=pred.cpu().data.numpy()
                pred= np.argmax(pred,axis=1)
                iou.add_batch(pred,target.cpu().data.numpy())
                
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)
                # if batch_idx % iter_inter == 0:
                #     logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
                #         epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, valid_iter_loss.avg))
            val_loss=valid_iter_loss.avg
            acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
            writer.add_scalar('val/miou', mean_iu, epoch)
            writer.add_scalar('val/acc', acc, epoch)
            print('acc:', acc, 'acc_cls:', acc_cls, '\n\niu:', iu, '\n\nmean_iu:', mean_iu, '\nfwavacc:', fwavacc)
            logger.info('[val] epoch:{} miou:{:.4f} val_loss:{:.4f} '.format(epoch,mean_iu, val_loss))
        
        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(SWA_opt.param_groups[0]['lr'])
        # 保存模型
        if epoch % save_inter == 0 and epoch > min_inter:
            state = {'best_iou': best_iou, 'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': SWA_opt.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
            # 保存最优模型
        if mean_iu > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'best_iou': best_iou, 'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': SWA_opt.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = mean_iu
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
    

    SWA_opt.swap_swa_sgd()
    state = {'best_iou': best_iou, 'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': SWA_opt.state_dict()}
    filename = os.path.join(save_ckpt_dir, 'SWA_checkpoint-best.pth')
    torch.save(state, filename)
    best_mode = copy.deepcopy(model)
    logger.info('[save] SWA Model saved =============================')
    writer.close()

    return best_mode, model