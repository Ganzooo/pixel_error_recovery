from pickletools import uint8
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

from torch import int8
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
noiseDir.append(os.path.join(args.src_noise, 'pr_0_5', 'index'))
noiseDir.append(os.path.join(args.src_noise, 'pr_1_0', 'index'))
noiseDir.append(os.path.join(args.src_noise, 'col_1', 'index')) 
noiseDir.append(os.path.join(args.src_noise, 'col_2', 'index')) 
noiseDir.append(os.path.join(args.src_noise, 'cluster_2', 'index'))
noiseDir.append(os.path.join(args.src_noise, 'cluster_3', 'index'))

gtDir = args.src_gt

tarDir = []
tarDir.append(os.path.join(args.tar, 'pr_0_5', 'index'))
tarDir.append(os.path.join(args.tar, 'pr_1_0', 'index'))
tarDir.append(os.path.join(args.tar, 'col_1', 'index')) 
tarDir.append(os.path.join(args.tar, 'col_2', 'index')) 
tarDir.append(os.path.join(args.tar, 'cluster_2', 'index'))
tarDir.append(os.path.join(args.tar, 'cluster_3', 'index'))

resultCSV = []
resultCSV.append(os.path.join(args.tar, 'sensitivity_pr_0_5.csv'))
resultCSV.append(os.path.join(args.tar, 'sensitivity_pr_1_0.csv'))
resultCSV.append(os.path.join(args.tar, 'sensitivity_col_1.csv')) 
resultCSV.append(os.path.join(args.tar, 'sensitivity_col_2.csv')) 
resultCSV.append(os.path.join(args.tar, 'sensitivity_cluster_2.csv'))
resultCSV.append(os.path.join(args.tar, 'sensitivity_cluster_3.csv'))

def calc_psnr(y, y_target):
    h, w, c = y.shape
    y = np.clip(np.round(y), 0, 255).astype(np.float32)
    y_target = np.clip(np.round(y_target), 0, 255).astype(np.float32)
    
    mse = np.mean((y - y_target) ** 2)
    if mse == 0:
        return 100
    return 20. * math.log10(255. / math.sqrt(mse))

def calc_ssim(y, y_target):
    score = ssim(y, y_target)
    #diff = (diff * 255).astype("uint8")
    return score

from sklearn.metrics import confusion_matrix
def errorSensitivity(idx, filePath2_noise, filePath3_rec, csvPath):
    #print(idx)
    #noiseImg= np.fromfile(filePath2_noise[idx], dtype=int)
    #recImg = np.fromfile(filePath3_rec[idx], dtype=int)
    
    #print(filePath2_noise[idx])
    #print(filePath3_rec[idx])
    
    noiseData = cv2.imread(filePath2_noise[idx], cv2.IMREAD_GRAYSCALE)
    _noiseData = noiseData.flatten()
    
    recData = cv2.imread(filePath3_rec[idx], cv2.IMREAD_GRAYSCALE)
    _recData = recData.flatten()
    
    tn, fp, fn, tp = confusion_matrix(_noiseData, _recData, labels=[0,255]).ravel()
    _SENSIBILITY = np.float(tp / (tp+fn))
    _SPECIFICITY = np.float(tn / (tn+fp))
    _PPV = np.float(tp / (tp + fp))
    _NPV = np.float(tn / (tn + fn))
    
    # print('TN:',tn)
    # print('FP:',fp)
    # print('FN:',fn)
    # print('TP:',tp)
    # _psnr_gt = calc_psnr(gtImg, noiseImg)
    # _psnr_rec = calc_psnr(gtImg, recImg)
    
    fname = os.path.split(filePath2_noise[idx])[1]
    #result = "name:{},noise:{},rec:{},".format(fname[:-4], _psnr_gt, _psnr_rec)
    with open(csvPath, 'a', newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([fname[:-4], tn, fp, fn, tp, _SENSIBILITY, _SPECIFICITY, _PPV, _NPV])
    
if __name__=='__main__':
    for _csv in resultCSV:
        if os.path.exists(_csv):
            os.remove(_csv)
        with open(_csv, 'a', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(['File Name', 'TN', 'FP', 'FN', 'TP', 'SENSIBILITY', "SPECIFICITY", "PPV", "NPV"])
        #print("The file has been deleted successfully")
    
    for _idx in tqdm(range(len(noiseDir)), desc="Recovery Noise type"):
        #get sorted folders
        _noiseImgDir = natsorted(glob(os.path.join(noiseDir[_idx], '*.png')))
        #_gtImgDir = natsorted(glob(os.path.join(gtDir, '*.png')))
        _recImgDir = natsorted(glob(os.path.join(tarDir[_idx], '*.png')))
        _csv = resultCSV[_idx]
        assert len(_noiseImgDir) == len(_recImgDir), "check folder name"
        #print('\n\n Noise Type: {}'.format(os.path.split(_noiseDir)[-1]))
        
        Parallel(n_jobs=args.num_cores)(delayed(errorSensitivity)(idx=_idx, filePath2_noise=_noiseImgDir, filePath3_rec=_recImgDir, csvPath=_csv) for _idx in tqdm(range(0,len(_noiseImgDir)), desc="    Images"))
        
    for _csv in resultCSV:
        data = pd.read_csv(_csv)
        _sens_avg = np.average(data['SENSIBILITY'])
        _spe_avg = np.average(data['SPECIFICITY'])
        with open(_csv, 'a', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([':::AVERAGE ALL:::', ' ', ' ', ' ', ' ', np.average(data['SENSIBILITY']), np.average(data['SPECIFICITY']), np.average(data['PPV']), np.average(data['NPV'])])
        
        