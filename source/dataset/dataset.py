import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import cv2
import numpy as np
import glob2
import random
import albumentations as A
import albumentations.pytorch as Ap
from source.dataset.custom_dataset import CustomDataSet
from source.dataset.denoise_dataset import DenoiseDataSet
from source.dataset.sr_dataset import SrDataSet

def get_dataset(cfg):
    train_dataloaders = []
    valid_dataloaders = []

    
    if cfg.train_config.mode == 'denoise':
        noiseDataTrain_all = DenoiseDataSet(cfg.dataset.noised_img_path_all, cfg.dataset.gt_img_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
        patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram, use_mask_loss=cfg.train_config.use_masked_loss)
        noiseDataValid_all = DenoiseDataSet(cfg.dataset.noised_img_path_val_all, cfg.dataset.gt_img_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram, use_mask_loss=cfg.train_config.use_masked_loss)
        
        train_dataloaders += [{'name': 'all', 'dataloader': DataLoader(dataset=noiseDataTrain_all, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
        valid_dataloaders += [{'name': 'all', 'dataloader': DataLoader(dataset=noiseDataValid_all, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
        return train_dataloaders, valid_dataloaders
    elif cfg.train_config.mode == 'sr':
        noiseDataTrain_all = SrDataSet(cfg.dataset.lr_img_path_all, cfg.dataset.gt_img_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
        patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram, use_mask_loss=cfg.train_config.use_masked_loss)
        noiseDataValid_all = SrDataSet(cfg.dataset.lr_img_path_val_all, cfg.dataset.gt_img_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram, use_mask_loss=cfg.train_config.use_masked_loss)
        
        train_dataloaders += [{'name': 'all', 'dataloader': DataLoader(dataset=noiseDataTrain_all, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
        valid_dataloaders += [{'name': 'all', 'dataloader': DataLoader(dataset=noiseDataValid_all, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
        return train_dataloaders, valid_dataloaders
    
    if cfg.train_config.db_split:
        noiseDataTrain_pr1 = CustomDataSet(cfg.dataset.noised_img_path_pr1, cfg.dataset.gt_img_path_pr1, cfg.dataset.img_org_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
            patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataTrain_pr0_5 = CustomDataSet(cfg.dataset.noised_img_path_pr0_5, cfg.dataset.gt_img_path_pr0_5, cfg.dataset.img_org_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
            patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataTrain_col1 = CustomDataSet(cfg.dataset.noised_img_path_col1, cfg.dataset.gt_img_path_col1, cfg.dataset.img_org_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
            patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataTrain_col2 = CustomDataSet(cfg.dataset.noised_img_path_col2, cfg.dataset.gt_img_path_col2, cfg.dataset.img_org_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
            patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataTrain_cl2 = CustomDataSet(cfg.dataset.noised_img_path_cl2, cfg.dataset.gt_img_path_cl2, cfg.dataset.img_org_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
            patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataTrain_cl3 = CustomDataSet(cfg.dataset.noised_img_path_cl3, cfg.dataset.gt_img_path_cl3, cfg.dataset.img_org_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
            patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)

        noiseDataValid_pr1 = CustomDataSet(cfg.dataset.noised_img_path_val_pr1, cfg.dataset.gt_img_path_val_pr1, cfg.dataset.img_org_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataValid_pr0_5 = CustomDataSet(cfg.dataset.noised_img_path_val_pr0_5, cfg.dataset.gt_img_path_val_pr0_5, cfg.dataset.img_org_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataValid_col1 = CustomDataSet(cfg.dataset.noised_img_path_val_col1, cfg.dataset.gt_img_path_val_col1, cfg.dataset.img_org_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataValid_col2 = CustomDataSet(cfg.dataset.noised_img_path_val_col2, cfg.dataset.gt_img_path_val_col2, cfg.dataset.img_org_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataValid_cl2 = CustomDataSet(cfg.dataset.noised_img_path_val_cl2, cfg.dataset.gt_img_path_val_cl2, cfg.dataset.img_org_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
        noiseDataValid_cl3 = CustomDataSet(cfg.dataset.noised_img_path_val_cl3, cfg.dataset.gt_img_path_val_cl3, cfg.dataset.img_org_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)

        train_dataloaders += [{'name': 'pr1', 'dataloader': DataLoader(dataset=noiseDataTrain_pr1, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
        train_dataloaders += [{'name': 'pr0_5', 'dataloader': DataLoader(dataset=noiseDataTrain_pr0_5, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
        train_dataloaders += [{'name': 'col1', 'dataloader': DataLoader(dataset=noiseDataTrain_col1, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
        train_dataloaders += [{'name': 'col2', 'dataloader': DataLoader(dataset=noiseDataTrain_col2, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
        train_dataloaders += [{'name': 'cl2', 'dataloader': DataLoader(dataset=noiseDataTrain_cl2, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
        train_dataloaders += [{'name': 'cl3', 'dataloader': DataLoader(dataset=noiseDataTrain_cl3, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]

        valid_dataloaders += [{'name': 'pr0_5', 'dataloader': DataLoader(dataset=noiseDataValid_pr0_5, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
        valid_dataloaders += [{'name': 'pr1', 'dataloader': DataLoader(dataset=noiseDataValid_pr1, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
        valid_dataloaders += [{'name': 'cl2', 'dataloader': DataLoader(dataset=noiseDataValid_cl2, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
        valid_dataloaders += [{'name': 'cl3', 'dataloader': DataLoader(dataset=noiseDataValid_cl3, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
        valid_dataloaders += [{'name': 'col1', 'dataloader': DataLoader(dataset=noiseDataValid_col1, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
        valid_dataloaders += [{'name': 'col2', 'dataloader': DataLoader(dataset=noiseDataValid_col2, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
    else: 
        # noiseDataTrain_all = CustomDataSet(cfg.dataset.noised_img_path_all, cfg.dataset.gt_img_path_all, cfg.dataset.img_org_path_all, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors,
        # patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram, use_mask_loss=cfg.train_config.use_masked_loss)
        noiseDataValid_all = CustomDataSet(cfg.dataset.noised_img_path_val_all, cfg.dataset.gt_img_path_val_all, cfg.dataset.img_org_path_val_all, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
            repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram, use_mask_loss=cfg.train_config.use_masked_loss)
        
        # train_dataloaders += [{'name': 'all', 'dataloader': DataLoader(dataset=noiseDataTrain_all, num_workers=cfg.train_config.train_threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
        valid_dataloaders += [{'name': 'all', 'dataloader': DataLoader(dataset=noiseDataValid_all, num_workers=cfg.train_config.val_threads, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
    return train_dataloaders, valid_dataloaders