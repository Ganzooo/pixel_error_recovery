import os
import glob
import random
import pickle

import numpy as np
import imageio
import torch
import torch.utils.data as data
import skimage.color as sc
import time
#from source.utils.utils_sr import ndarray2tensor

def _ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

class CustomDataSet(data.Dataset):
    def __init__(self, image_folder, gt_folder, image_org_folder, train=True, augment=True, scale=2, colors=1, patch_size=96, repeat=168, store_in_ram=True):
        super(CustomDataSet, self).__init__()
        self.image_folder = image_folder
        self.gt_folder = gt_folder
        self.image_org_folder = image_org_folder
        self.augment   = augment
        self.train     = train
        self.img_postfix = '.png'
        self.scale = scale
        self.colors = colors
        self.store_in_ram = store_in_ram
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0

        self.imageName = []
        self.gtName = []
        self.imageOrgName = []

        _imgPath = sorted(glob.glob(self.image_folder + '*.png'))
        _imgPathGT = sorted(glob.glob(self.gt_folder + '*.png'))
        assert len(_imgPath) == len(_imgPathGT)
        for idx, (_imgNameHR, _imgNameLR) in enumerate(zip(_imgPath, _imgPathGT)):
            self.imageName.append(_imgNameHR)
            self.gtName.append(_imgNameLR)
            _name = os.path.basename(_imgNameHR).split('_')
            if _name[1] == 'pr':
                base_name = _name[4] + '_' + _name[5] + '_' + _name[6] + '_' + _name[7]
            else: 
                base_name = _name[3] + '_' + _name[4] + '_' + _name[5] + '_' + _name[6]
            self.imageOrgName.append(os.path.join(self.image_org_folder,base_name))

        self.nums_trainset = len(_imgPath)

        if self.store_in_ram:
            self.images = []
            self.gts = []
            self.fname = []
            self.imagesOrg = []
            

            for i in range(len(_imgPath)):
                _image, _gt, _imageOrg = imageio.imread(self.imageName[i], pilmode="RGB"), imageio.imread(self.gtName[i], pilmode="RGB"), imageio.imread(self.imageOrgName[i], pilmode="RGB")
                if self.colors == 1:
                    #_image, _gt = sc.rgb2ycbcr(_image)[:, :, 0:1], sc.rgb2ycbcr(_gt)[:, :, 0:1]
                    _image, _gt = _image.convert('L'), _gt.convert('L')
                self.images.append(_image)
                self.gts.append(_gt) 
                self.fname.append(self.imageName[i])
                self.imagesOrg.append(_imageOrg)
                
    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return len(self.imageName)
        
    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        # get whole image
        if self.store_in_ram:
            _image, _gt, _fname, _imageOrg = self.images[idx], self.gts[idx], self.fname[idx], self.imagesOrg[idx]
        else:
            if self.colors != 1:
                _image, _gt, _fname, _imageOrg = imageio.imread(self.imageName[idx], pilmode="RGB"), imageio.imread(self.gtName[idx], pilmode="L"), self.imageName[idx], imageio.imread(self.imageOrgName[idx], pilmode="RGB")
            if self.colors == 1:
                _image, _gt, _fname, _imageOrg = imageio.imread(self.imageName[idx], pilmode="RGB"), imageio.imread(self.gtName[idx], pilmode="L"), self.imageName[idx], imageio.imread(self.imageOrgName[idx], pilmode="RGB")
                #_image, _gt = _image.rgb2ycbcr(_image)[:, :, 0:1], _gt.rgb2ycbcr(_gt)[:, :, 0:1]
                _image = _image[:, :, 0:1]
        if self.train:
            # crop patch randomly
            _image_h, _image_w, _ = _image.shape
            hp = self.patch_size
            lp = self.patch_size
            lx, ly = random.randrange(0, _image_w - lp + 1), random.randrange(0, _image_h - lp + 1)
            hx, hy = lx, ly
            if self.colors == 1:
                _image_patch, _gt_patch = _image[ly:ly+lp, lx:lx+lp,:], _gt[hy:hy+hp, hx:hx+hp]
            else: 
                _image_patch, _gt_patch = _image[ly:ly+lp, lx:lx+lp,:], _gt[hy:hy+hp, hx:hx+hp]
            # augment data
            if self.augment:
                #print("data augmentation!")
                hflip = random.random() > 0.5
                vflip = random.random() > 0.5
                rot90 = random.random() > 0.5
                if self.colors == 1:
                    if hflip: _image_patch, _gt_patch = _image_patch[:, ::-1,:], _gt_patch[:, ::-1]
                    if vflip: _image_patch, _gt_patch = _image_patch[::-1, :,:], _gt_patch[::-1, :]
                    #if rot90: _image_patch, _gt_patch = _image_patch.transpose(1,0,2), _gt_patch.transpose(1,0,2)
                else: 
                    if hflip: _image_patch, _gt_patch = _image_patch[:, ::-1, :], _gt_patch[:, ::-1]
                    if vflip: _image_patch, _gt_patch = _image_patch[::-1, :, :], _gt_patch[::-1, :]
                    #if rot90: _image_patch, _gt_patch = _image_patch.transpose(1,0,2), _gt_patch.transpose(1,0,2)
            # numpy to tensor
            _image_patch = _image_patch.transpose(2,0,1)
            
            _image_patch = _image_patch / 255.
            _image_patch = torch.from_numpy(_image_patch).float()
            
            #_gt_patch = _gt_patch.transpose(2,0,1)
            #_gt_patch = np.squeeze(_gt_patch, axis=0) 
            #_gt_patch = _gt_patch / 255.
            #_gt_patch[_gt_patch == 0] = 250 
            #_gt_patch[_gt_patch == 255] = 1
            gt_patch = torch.from_numpy(_gt_patch.copy()).long()
    
            return _image_patch, gt_patch, idx, _fname, _imageOrg
        else:
            _image = _image.transpose(2,0,1)
            _image = _image / 255.
            _image = torch.from_numpy(_image).float()
            
            #_gt = _gt.transpose(2,0,1)
            #_gt = np.squeeze(_gt, axis=0) 
            #_gt_patch = _gt_patch / 255.
            # _gt[_gt == 0] = 250 
            # _gt[_gt == 255] = 1
            _gt = torch.from_numpy(_gt.copy()).long()
            return _image, _gt, idx, _fname, _imageOrg

if __name__ == '__main__':
    noise = '/dataset2/CITYSCAPES_DATASET/DEFECTION_NOISE_PAPER/noise_paper_d2/pr_0_5/'
    noise_pixel = '/dataset2/CITYSCAPES_DATASET/DEFECTION_NOISE_PAPER/noise_paper_d2/pr_0_5/index/'
    
    #LR_folder = '/dataset/SR2/DIV2K/DIV2K_train_LR_bicubic'
    argment   = True
    div2k = CustomDataSet(noise, noise_pixel, train=True, augment=True, scale=3, colors=3, patch_size=96, repeat=1, store_in_ram=True)

    print("number of sample: {}".format(len(div2k)))
    start = time.time()
    for idx in range(10):
        lr, hr = div2k[idx]
        print(lr.shape, hr.shape)
    end = time.time()
    print(end - start)
