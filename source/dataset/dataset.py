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


    
def get_dataset(cfg):

    # div2k = DIV2K(
    #     cfg.dataset.div2k_hr_path, 
    #     cfg.dataset.div2k_lr_path, 
    #     train=True, 
    #     augment=cfg.train_config.data_augment, 
    #     scale=cfg.train_config.scale, 
    #     colors=cfg.train_config.colors, 
    #     patch_size=cfg.train_config.patch_size, 
    #     repeat=cfg.train_config.data_repeat, 
    #     store_in_ram=cfg.train_config.store_in_ram
    # )
    
    #noiseDataTrain_pr5 = CustomDataSet(cfg.dataset.noised_img_path_pr5, cfg.dataset.gt_img_path_pr5, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
    #    patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataTrain_pr1 = CustomDataSet(cfg.dataset.noised_img_path_pr1, cfg.dataset.gt_img_path_pr1, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
         patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataTrain_pr0_5 = CustomDataSet(cfg.dataset.noised_img_path_pr0_5, cfg.dataset.gt_img_path_pr0_5, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
         patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataTrain_col1 = CustomDataSet(cfg.dataset.noised_img_path_col1, cfg.dataset.gt_img_path_col1, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
         patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataTrain_col2 = CustomDataSet(cfg.dataset.noised_img_path_col2, cfg.dataset.gt_img_path_col2, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
         patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataTrain_cl2 = CustomDataSet(cfg.dataset.noised_img_path_cl2, cfg.dataset.gt_img_path_cl2, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
         patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataTrain_cl3 = CustomDataSet(cfg.dataset.noised_img_path_cl3, cfg.dataset.gt_img_path_cl3, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
         patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    #noiseDataTrain_cl5 = CustomDataSet(cfg.dataset.noised_img_path_cl5, cfg.dataset.gt_img_path_cl5, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
    #    patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    
    #noiseDataTrain_row1 = CustomDataSet(cfg.dataset.noised_img_path_row1, cfg.dataset.gt_img_path_row1, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
    #    patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    #noiseDataTrain_row2 = CustomDataSet(cfg.dataset.noised_img_path_row2, cfg.dataset.gt_img_path_row2, train=True, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, 
    #    patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    
        
    #noiseDataValid_pr5 = CustomDataSet(cfg.dataset.noised_img_path_val_pr5, cfg.dataset.gt_img_path_val_pr5, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
    #    repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataValid_pr1 = CustomDataSet(cfg.dataset.noised_img_path_val_pr1, cfg.dataset.gt_img_path_val_pr1, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
        repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataValid_pr0_5 = CustomDataSet(cfg.dataset.noised_img_path_val_pr0_5, cfg.dataset.gt_img_path_val_pr0_5, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
        repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataValid_col1 = CustomDataSet(cfg.dataset.noised_img_path_val_col1, cfg.dataset.gt_img_path_val_col1, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
        repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataValid_col2 = CustomDataSet(cfg.dataset.noised_img_path_val_col2, cfg.dataset.gt_img_path_val_col2, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
        repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataValid_cl2 = CustomDataSet(cfg.dataset.noised_img_path_val_cl2, cfg.dataset.gt_img_path_val_cl2, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
        repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    noiseDataValid_cl3 = CustomDataSet(cfg.dataset.noised_img_path_val_cl3, cfg.dataset.gt_img_path_val_cl3, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
        repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    #noiseDataValid_cl5 = CustomDataSet(cfg.dataset.noised_img_path_val_cl5, cfg.dataset.gt_img_path_val_cl5, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
    #    repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    #noiseDataValid_row1 = CustomDataSet(cfg.dataset.noised_img_path_val_row1, cfg.dataset.gt_img_path_val_row1, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
    #    repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    #noiseDataValid_row2 = CustomDataSet(cfg.dataset.noised_img_path_val_row2, cfg.dataset.gt_img_path_val_row2, train=False, augment=cfg.train_config.data_augment, scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, 
    #    repeat=cfg.train_config.data_repeat, store_in_ram=cfg.train_config.store_in_ram)
    # div2k_val = DIV2K(cfg.dataset.div2k_hr_path, cfg.dataset.div2k_lr_path, train=False, augment=cfg.train_config.data_augment, 
    #     scale=cfg.train_config.scale, colors=cfg.train_config.colors, patch_size=cfg.train_config.patch_size, repeat=cfg.train_config.data_repeat, 
    #     store_in_ram=cfg.train_config.store_in_ram
    # )

    # set5  = Benchmark(cfg.dataset.set5_hr_path, cfg.dataset.set5_lr_path, scale=cfg.train_config.scale, colors=cfg.train_config.colors, store_in_ram=cfg.train_config.store_in_ram)
    # set14 = Benchmark(cfg.dataset.set14_hr_path, cfg.dataset.set14_lr_path, scale=cfg.train_config.scale, colors=cfg.train_config.colors, store_in_ram=cfg.train_config.store_in_ram)
    # b100  = Benchmark(cfg.dataset.b100_hr_path, cfg.dataset.b100_lr_path, scale=cfg.train_config.scale, colors=cfg.train_config.colors, store_in_ram=cfg.train_config.store_in_ram)
    # u100  = Benchmark(cfg.dataset.u100_hr_path, cfg.dataset.u100_lr_path, scale=cfg.train_config.scale, colors=cfg.train_config.colors, store_in_ram=cfg.train_config.store_in_ram)
    train_dataloader = []
    #train_dataloader = DataLoader(dataset=noiseDataTrain_pr5, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    #train_dataloader += [{'name': 'pr5', 'dataloader': DataLoader(dataset=noiseDataTrain_pr5, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    train_dataloader += [{'name': 'pr1', 'dataloader': DataLoader(dataset=noiseDataTrain_pr1, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    train_dataloader += [{'name': 'pr0_5', 'dataloader': DataLoader(dataset=noiseDataTrain_pr0_5, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    train_dataloader += [{'name': 'col1', 'dataloader': DataLoader(dataset=noiseDataTrain_col1, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    train_dataloader += [{'name': 'col2', 'dataloader': DataLoader(dataset=noiseDataTrain_col2, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    train_dataloader += [{'name': 'cl2', 'dataloader': DataLoader(dataset=noiseDataTrain_cl2, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    train_dataloader += [{'name': 'cl3', 'dataloader': DataLoader(dataset=noiseDataTrain_cl3, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    #train_dataloader += [{'name': 'cl5', 'dataloader': DataLoader(dataset=noiseDataTrain_cl5, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    #train_dataloader += [{'name': 'row1', 'dataloader': DataLoader(dataset=noiseDataTrain_row1, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    #train_dataloader += [{'name': 'row2', 'dataloader': DataLoader(dataset=noiseDataTrain_row2, num_workers=cfg.train_config.threads, batch_size=cfg.train_config.batch_size, shuffle=True, pin_memory=True, drop_last=True)}]
    
    valid_dataloaders = []
    #valid_dataloaders += [{'name': 'pr5', 'dataloader': DataLoader(dataset=noiseDataValid_pr5, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'pr0_5', 'dataloader': DataLoader(dataset=noiseDataValid_pr0_5, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
    valid_dataloaders += [{'name': 'pr1', 'dataloader': DataLoader(dataset=noiseDataValid_pr1, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
    valid_dataloaders += [{'name': 'cl2', 'dataloader': DataLoader(dataset=noiseDataValid_cl2, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
    valid_dataloaders += [{'name': 'cl3', 'dataloader': DataLoader(dataset=noiseDataValid_cl3, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
    valid_dataloaders += [{'name': 'col1', 'dataloader': DataLoader(dataset=noiseDataValid_col1, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
    valid_dataloaders += [{'name': 'col2', 'dataloader': DataLoader(dataset=noiseDataValid_col2, batch_size=cfg.train_config.batch_size_val, shuffle=False)}]
    #valid_dataloaders += [{'name': 'cl5', 'dataloader': DataLoader(dataset=noiseDataValid_cl5, batch_size=1, shuffle=False)}]
    #valid_dataloaders += [{'name': 'row1', 'dataloader': DataLoader(dataset=noiseDataValid_row1, batch_size=1, shuffle=False)}]
    #valid_dataloaders += [{'name': 'row2', 'dataloader': DataLoader(dataset=noiseDataValid_row2, batch_size=1, shuffle=False)}]
    #valid_dataloaders = DataLoader(dataset=noiseDataValid, batch_size=1, shuffle=False)
    
    return train_dataloader, valid_dataloaders