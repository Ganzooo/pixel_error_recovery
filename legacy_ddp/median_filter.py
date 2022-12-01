from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
import math
import random

from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from skimage.metrics import structural_similarity as ssim_calc
from statistics import mean, median

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')

# All val images
parser.add_argument('--src_dir', default='/dataset/Cityscapes/DEFECTION_NOISE_PAPER/noise_rgb_paper_val/all/', type=str, help='Directory of GT images')
parser.add_argument('--tar_gt', default='/dataset/Cityscapes/DEFECTION_NOISE_PAPER/gt_val',type=str, help='Directory of Converted GrayScale GT images')
parser.add_argument('--rec_dir', default='./outputs/',type=str, help='Directory of Noise generated images')

parser.add_argument('--num_cores', default=20, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.rec_dir
tar_gt = args.tar_gt
NUM_CORES = args.num_cores

__DEBUG__ = 0
# if os.path.exists(tar):
#     os.system("rm -r {}".format(tar))
#os.makedirs(tar)

os.makedirs(tar + '/filtered_output/')

#get sorted folders
imgDir = natsorted(glob(os.path.join(src, '*.png')))

img_files, label_files = [], []
for _idx, file_ in enumerate(imgDir):
    filename = os.path.split(file_)[-1]
    img_files.append(file_)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

psnr_all = []
ssim_all = []
def median_filter(iter):
    _imgSrc = cv2.imread(img_files[iter])
    
    
    
    _imgRec = cv2.medianBlur(_imgSrc, 5)
    
    filename = os.path.split(img_files[iter])[-1]

    cv2.imwrite(os.path.join(tar + '/filtered_output/',filename), _imgRec)
    
    _name = os.path.basename(filename).split('_')
    if _name[1] == 'pr':
        base_name = _name[4] + '_' + _name[5] + '_' + _name[6] + '_' + _name[7]
        csv_result_name = 'result_{}_{}_{}.csv'.format(_name[1],_name[2],_name[3])
    else: 
        base_name = _name[3] + '_' + _name[4] + '_' + _name[5] + '_' + _name[6]
        csv_result_name = 'result_{}_{}.csv'.format(_name[1],_name[2])
    
    _imgOrg= cv2.imread(os.path.join(tar_gt,base_name))
    
    psnr, ssim = 0, 0
    
    psnr = psnr_calc(_imgOrg,_imgRec)
                    
    if _imgOrg.shape[2] == 3:
        ssim_R = ssim_calc(_imgOrg[:,:,0],_imgRec[:,:,0], full=True)
        ssim_G = ssim_calc(_imgOrg[:,:,1],_imgRec[:,:,1], full=True)
        ssim_B = ssim_calc(_imgOrg[:,:,2],_imgRec[:,:,2], full=True)
        ssim = mean([ssim_R[0], ssim_G[0], ssim_B[0]])

    psnr_all.append(psnr)
    ssim_all.append(ssim)
    with open(os.path.join(tar,csv_result_name), "a") as file:
        file.write("{},{},{}\n".format(str(filename), str(psnr), str(ssim)))

#for i in tqdm(range(0,len(img_files))):
#   median_filter(i)
    
Parallel(n_jobs=NUM_CORES)(delayed(median_filter)(i) for i in tqdm(range(len(img_files))))

# print("PSNR average: {}".format(mean(psnr_all)))
# print("SSIM average: {}".format(mean(ssim_all)))