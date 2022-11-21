import numpy as np 
import math 
import os
import argparse
from natsort import natsorted
from glob import glob
from joblib import Parallel, delayed
import cv2
from skimage.metrics import structural_similarity as ssim
import csv
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Defected NOISE Recovery Algorithm')
parser.add_argument('--src_gt', default='/dataset2/CITYSCAPES_DATASET/DEFECTION_NOISE_PAPER/gt_d2/',type=str, help='Directory for image patches')
parser.add_argument('--src_noise', default='/dataset2/CITYSCAPES_DATASET/DEFECTION_NOISE_PAPER/noise_paper_d2/',type=str, help='Directory for image patches')
parser.add_argument('--tar', default='./result_d2/', type=str, help='Directory of Recoverd images')
parser.add_argument('--num_cores', default=1, type=int, help='Number of CPU Cores')
parser.add_argument('--recovery_type', default='DPD_D', type=str, help='recovery type: DPD_D, DPD_M')

args = parser.parse_args()
REC_TYPE = args.recovery_type 
args.tar = args.tar + REC_TYPE 

noiseDir = []
noiseDir.append(os.path.join(args.src_noise, 'pr_0_5'))
noiseDir.append(os.path.join(args.src_noise, 'pr_1_0'))
noiseDir.append(os.path.join(args.src_noise, 'col_1')) 
noiseDir.append(os.path.join(args.src_noise, 'col_2')) 
noiseDir.append(os.path.join(args.src_noise, 'cluster_2'))
noiseDir.append(os.path.join(args.src_noise, 'cluster_3'))

gtDir = args.src_gt

tarDir = []
tarDir.append(os.path.join(args.tar, 'pr_0_5'))
tarDir.append(os.path.join(args.tar, 'pr_1_0'))
tarDir.append(os.path.join(args.tar, 'col_1')) 
tarDir.append(os.path.join(args.tar, 'col_2')) 
tarDir.append(os.path.join(args.tar, 'cluster_2'))
tarDir.append(os.path.join(args.tar, 'cluster_3'))

resultTxt = []
resultTxt.append(os.path.join(args.tar, 'result_psnr_pr_0_5.csv'))
resultTxt.append(os.path.join(args.tar, 'result_psnr_pr_1_0.csv'))
resultTxt.append(os.path.join(args.tar, 'result_psnr_col_1.csv')) 
resultTxt.append(os.path.join(args.tar, 'result_psnr_col_2.csv')) 
resultTxt.append(os.path.join(args.tar, 'result_psnr_cluster_2.csv'))
resultTxt.append(os.path.join(args.tar, 'result_psnr_cluster_3.csv'))

def calc_psnr(y, y_target):
    h, w = y.shape
    y = np.clip(np.round(y), 0, 255).astype(np.float32)
    y_target = np.clip(np.round(y_target), 0, 255).astype(np.float32)
    
    mse = np.mean((y - y_target) ** 2)
    if mse == 0:
        return 100
    return 20. * math.log10(255. / math.sqrt(mse))

def calc_ssim(y, y_target):
    (score, diff) = ssim(y, y_target, full=True)
    #diff = (diff * 255).astype("uint8")

    return score

def imgPSNR(idx, filePath1_gt, filePath2_noise, filePath3_rec, txtPath):
    #print(idx)
    gtImg = cv2.imread(filePath1_gt[idx], cv2.IMREAD_GRAYSCALE)
    noiseImg= cv2.imread(filePath2_noise[idx], cv2.IMREAD_GRAYSCALE)
    recImg = cv2.imread(filePath3_rec[idx], cv2.IMREAD_GRAYSCALE)
    
    _psnr_gt = calc_psnr(gtImg, noiseImg)
    _psnr_rec = calc_psnr(gtImg, recImg)
    _ssim_gt = calc_ssim(gtImg, noiseImg)
    _ssim_rec = calc_ssim(gtImg, recImg)
    
    fname = os.path.split(filePath1_gt[idx])[1]
    #result = "name:{},noise:{},rec:{},".format(fname[:-4], _psnr_gt, _psnr_rec)
    with open(txtPath, 'a', newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([fname[:-4], _psnr_gt, _psnr_rec, _ssim_gt, _ssim_rec])
    #print("PSNR:",_psnr)
    
if __name__=='__main__':
    for _csv in resultTxt:
        if os.path.exists(_csv):
            os.remove(_csv)
        with open(_csv, 'a', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'PSNR_gt', 'PSNR_rec', 'SSIM_gt', 'SSIM_rec'])
    #     #print("The file has been deleted successfully")
    
    for _idx in tqdm(range(len(noiseDir)), desc="Recovery Noise type"):
        #get sorted folders
        _noiseImgDir = natsorted(glob(os.path.join(noiseDir[_idx], '*.png')))
        _gtImgDir = natsorted(glob(os.path.join(gtDir, '*.png')))
        _recImgDir = natsorted(glob(os.path.join(tarDir[_idx], '*.png')))
        _txt = resultTxt[_idx]
        assert len(_noiseImgDir) == len(_gtImgDir), "check folder name"
        
        Parallel(n_jobs=args.num_cores)(delayed(imgPSNR)(idx=_idx, filePath1_gt=_gtImgDir, filePath2_noise=_noiseImgDir, filePath3_rec=_recImgDir, txtPath=_txt) for _idx in tqdm(range(0,len(_gtImgDir)), desc="    Images"))
        
    for _csv in resultTxt:
        data = pd.read_csv(_csv)
        with open(_csv, 'a', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([':::AVERAGE ALL:::', np.average(data['PSNR_gt']), np.average(data['PSNR_rec']), np.average(data['SSIM_gt']), np.average(data['SSIM_rec'])])