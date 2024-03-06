from cmath import sqrt
import cv2
import math 
import os
import sys
import random
import numpy as np
import argparse
from natsort import natsorted
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Defected NOISE Recovery Algorithm')
#parser.add_argument('--src_gt', default='/dataset2/CITYSCAPES_DATASET/DEFECTION_NOISE_PAPER/gt_d2/',type=str, help='Directory for image patches')
#parser.add_argument('--src_noise', default='/dataset2/CITYSCAPES_DATASET/DEFECTION_NOISE_PAPER/noise_paper_d2/',type=str, help='Directory for image patches')

parser.add_argument('--src_gt', default='/dataset/Cityscapes/DEFECTION_NOISE_PAPER/gt_val/',type=str, help='Directory for image patches')
parser.add_argument('--src_noise', default='/dataset/Cityscapes/DEFECTION_NOISE_PAPER/noise_rgb_paper_val/',type=str, help='Directory for image patches')

parser.add_argument('--tar', default='./result_gray/', type=str, help='Directory of Recoverd images')
parser.add_argument('--num_cores', default=1, type=int, help='Number of CPU Cores')
parser.add_argument('--recovery_type', default='DPD_D', type=str, help='recovery type: DPD_D, DPD_M')

args = parser.parse_args()
REC_TYPE = args.recovery_type 
args.tar = args.tar + REC_TYPE 
if os.path.exists(args.tar):
    os.system("rm -r {}".format(args.tar))
os.makedirs(args.tar)
# os.makedirs(args.tar + '/pr_5_0/')
# os.makedirs(args.tar + '/pr_5_0/index/')
os.makedirs(args.tar + '/pr_0_5/')
os.makedirs(args.tar + '/pr_0_5/index/')
os.makedirs(args.tar + '/pr_1_0/')
os.makedirs(args.tar + '/pr_1_0/index/')
os.makedirs(args.tar + '/col_1/')
os.makedirs(args.tar + '/col_1/index/')
os.makedirs(args.tar + '/col_2/')
os.makedirs(args.tar + '/col_2/index/')
os.makedirs(args.tar + '/cluster_2/')
os.makedirs(args.tar + '/cluster_2/index/')
os.makedirs(args.tar + '/cluster_3/')
os.makedirs(args.tar + '/cluster_3/index/')

noiseDir = []
#noiseDir.append(os.path.join(args.src_noise, 'pr_5_0'))
noiseDir.append(os.path.join(args.src_noise, 'pr_0_5'))
noiseDir.append(os.path.join(args.src_noise, 'pr_1_0'))
noiseDir.append(os.path.join(args.src_noise, 'col_1')) 
noiseDir.append(os.path.join(args.src_noise, 'col_2')) 
noiseDir.append(os.path.join(args.src_noise, 'cluster_2'))
noiseDir.append(os.path.join(args.src_noise, 'cluster_3'))

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

N = 5

def euclidean_distances(N):
    _nStep = int(N / 2)
    d = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
           d[i,j] =  math.sqrt(math.pow(i-_nStep,2) + math.pow(j-_nStep,2))
    #print(d)
    return d

dxyGlobal = euclidean_distances(N)
print(dxyGlobal)

def add_column_row_noise(img):
    print(":::Noise Add PART-1:::")
    # col_rand = random.randrange(0, img.shape[0])
    # row_rand = random.randrange(0, img.shape[1])
    col_rand = 5
    row_rand = 10
    
    print('Col: {}, Row: {}'.format(row_rand,col_rand))
    img[col_rand,:] = [255]
    img[:,row_rand] = [255]
    return img
    
def Vest_func(crop_img, _dxy):
    _data = crop_img.ravel()
    _sum_a = 0
    _sum_b = 0
    
    for i in range(len(_data)):
        if _dxy[i] != 0:
            _sum_a += float(_data[i]) / _dxy[i]   
            _sum_b += 1.0 / _dxy[i]
    return float(_sum_a / _sum_b)

def Vavg_func(crop_img, P0):
    _data = crop_img.ravel()
    _sum_a = 0
    for i in range(len(_data)):
        _sum_a += float(_data[i] + P0)
    _sum_a = _sum_a / len(_data)
    return _sum_a / 2

def DPD_D_detect(crop_img, y, x, N):
    status = 'good'
    P0_rec = 0

    # If N = 3
    curr_pix_pos = int(int(N/2)*N+int(N/2))
    Pneighbor = np.delete(crop_img.ravel(),curr_pix_pos)
    dxy = dxyGlobal
    dxy = np.delete(dxy.ravel(),curr_pix_pos)
    Ph = Pneighbor.max()
    Pl = Pneighbor.min()
    P0 = crop_img[int(N / 2), int(N / 2)]
    
    if Pl < P0 and P0 < Ph: 
        return status, P0_rec

    Vest = Vest_func(Pneighbor, dxy)
    Vdiff = math.fabs(Vest-P0)

    Vavg = Vavg_func(Pneighbor, P0)
    
    ### Defective Pixel 0 or 255
    ### V0 = 0
    V0 = 0
    Vth = math.fabs(Vavg - V0)
    
    if Vdiff > Vth:
        status = 'defective'
        P0_rec = np.median(crop_img.ravel())
    else: 
        ### Defective Pixel 0 or 255
        ### V0 = 255
        V0 = 255
        Vth = math.fabs(Vavg - V0)
        if Vdiff > Vth:
            status = 'defective'
            P0_rec = np.median(crop_img.ravel())
        else:
            status = 'good'
    return status, P0_rec

def DPD_M_detect(crop_img, y, x, N):
    status = 'good'
    P0_rec = 0

    # If N = 3
    curr_pix_pos = int(int(N/2)*N+int(N/2))
    Pneighbor = np.delete(crop_img.ravel(),curr_pix_pos)
    dxy = dxyGlobal
    dxy = np.delete(dxy.ravel(),curr_pix_pos)
    Ph = Pneighbor.max()
    Pl = Pneighbor.min()
    P0 = crop_img[int(N / 2), int(N / 2)]
    
    if Pl < P0 and P0 < Ph: 
        return status, P0_rec

    #Vest = Vest_func(Pneighbor, dxy)
    #Pm = np.median(Pneighbor)
    Pm = sorted(Pneighbor)[int(len(Pneighbor)/2)]
    Vdiff = math.fabs(P0 - Pm)

    #Vavg = Vavg_func(Pneighbor, P0)
    
    ### Defective Pixel 0 or 255
    ### V0 = 0
    V0 = 0
    Vth = math.fabs((P0 + Pm)/2 - V0)
    
    if Vdiff > Vth:
        status = 'defective'
        P0_rec = np.median(crop_img.ravel())
    else: 
        ### Defective Pixel 0 or 255
        ### V0 = 255
        V0 = 255
        Vth = math.fabs((P0 + Pm)/2 - V0)
        if Vdiff > Vth:
            status = 'defective'
            P0_rec = np.median(crop_img.ravel())
        else:
            status = 'good'
    return status, P0_rec

def recover_defected_pixel_rgb__gray_scale(idx, dir):
    #print(":::Recovery PART 2:::")
    #cv2.imwrite('./original_image.png', img)
    fname = dir[idx]
    filename = os.path.split(fname)[-1]
    #print(filename)
    imgRGB = cv2.imread(fname)
    
    imgGray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
    

    recFname = os.path.join(args.tar, os.path.split(os.path.split(fname)[0])[-1], filename)
    recFnameIdx = os.path.join(args.tar, os.path.split(os.path.split(fname)[0])[-1],'index', filename)
        
    nH = imgGray.shape[0]
    nW = imgGray.shape[1]
    
    image_copy = np.zeros((nH+(int(N/2)*2),nW+(int(N/2)*2)))
    image_copy.fill(128)
    
    _nStep = int(N / 2)
    image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW] = imgGray[0:nH, 0:nW]
    pixel_error_idx = np.zeros((nH+(int(N/2)*2),nW+(int(N/2)*2)), dtype=np.uint8)
    
    
    for i in range(_nStep, nH+_nStep, 1):
        for j in range(_nStep, nW+_nStep, 1):
                crop_img = np.zeros((N, N))
                crop_img = image_copy[i-_nStep:(i-_nStep)+N, j-_nStep:(j-_nStep)+N]

                if REC_TYPE == "DPD_D":
                    _status, P_rec = DPD_D_detect(crop_img, i, j, N)
                else: 
                    _status, P_rec = DPD_M_detect(crop_img, i, j, N)

                if _status == 'good':
                    pass
                else: 
                    #print('Image recovered y={},x={}, org_val={}, rec_val={}'.format(y,x,image_copy[y,x],P_rec))
                    pixel_error_idx[i,j] = 255
                    image_copy[i, j] = P_rec

    imgRGB_rec = cv2.merge([image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW],image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW],image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW]])
    #recRGB.append(image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW])
    #recIdx.append(pixel_error_idx[_nStep:_nStep+nH, _nStep:_nStep+nW])

    #recRGB = np.array(recRGB).transpose(1,2,0)
    #recIdx = np.array(recIdx).transpose(1,2,0)
    cv2.imwrite(recFname, imgRGB_rec)
    cv2.imwrite(recFnameIdx, pixel_error_idx)

def recover_defected_pixel_rgb_scale(idx, dir):
    #print(":::Recovery PART 2:::")
    #cv2.imwrite('./original_image.png', img)
    fname = dir[idx]
    filename = os.path.split(fname)[-1]
    #print(filename)
    imgRGB = cv2.imread(fname)
    
    #recRGB = np.zeros_like(imgRGB)
    #recIdx = np.zeros_like(imgRGB)
    
    recRGB = []
    recIdx = []
    for i in range(3):
        img = imgRGB[:,:,i]
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #noise_img = add_column_row_noise(img)
        recFname = os.path.join(args.tar, os.path.split(os.path.split(fname)[0])[-1], filename)
        recFnameIdx = os.path.join(args.tar, os.path.split(os.path.split(fname)[0])[-1],'index', filename)
            
        nH = img.shape[0]
        nW = img.shape[1]
        
        image_copy = np.zeros((nH+(int(N/2)*2),nW+(int(N/2)*2)))
        image_copy.fill(128)
        
        _nStep = int(N / 2)
        image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW] = img[0:nH, 0:nW]
        pixel_error_idx = np.zeros((nH+(int(N/2)*2),nW+(int(N/2)*2)), dtype=np.uint8)
        
        
        for i in range(_nStep, nH+_nStep, 1):
            for j in range(_nStep, nW+_nStep, 1):
                    crop_img = np.zeros((N, N))
                    crop_img = image_copy[i-_nStep:(i-_nStep)+N, j-_nStep:(j-_nStep)+N]
                    
                    if REC_TYPE == "DPD_D":
                        _status, P_rec = DPD_D_detect(crop_img, i, j, N)
                    else: 
                        _status, P_rec = DPD_M_detect(crop_img, i, j, N)
                    
                    if _status == 'good':
                        pass
                    else: 
                        #print('Image recovered y={},x={}, org_val={}, rec_val={}'.format(y,x,image_copy[y,x],P_rec))
                        pixel_error_idx[i,j] = 255
                        image_copy[i, j] = P_rec
        
        recRGB.append(image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW])
        #recRGB[i] = recR[:,:]
        recIdx.append(pixel_error_idx[_nStep:_nStep+nH, _nStep:_nStep+nW])
                 
        #cv2.imwrite('./recovered_image_padded.png', image_copy)
        #fnameIdx = recFname[:-3] + 'csv'
        #np.save(fnameIdx, pixel_error_idx[_nStep:_nStep+nH, _nStep:_nStep+nW])
        #pixel_error_idx[_nStep:_nStep+nH, _nStep:_nStep+nW].tofile(fnameIdx, sep=',')
    
    recRGB = np.array(recRGB).transpose(1,2,0)
    recIdx = np.array(recIdx).transpose(1,2,0)
    cv2.imwrite(recFname, recRGB)
    cv2.imwrite(recFnameIdx, recIdx)
    
def recover_defected_pixel_gray_scale(idx, dir):
    #print(":::Recovery PART 2:::")
    #cv2.imwrite('./original_image.png', img)
    fname = dir[idx]
    filename = os.path.split(fname)[-1]
    #print(filename)
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #noise_img = add_column_row_noise(img)
    recFname = os.path.join(args.tar, os.path.split(os.path.split(fname)[0])[-1], filename)
    recFnameIdx = os.path.join(args.tar, os.path.split(os.path.split(fname)[0])[-1],'index', filename)
        
    nH = img.shape[0]
    nW = img.shape[1]
    
    image_copy = np.zeros((nH+(int(N/2)*2),nW+(int(N/2)*2)))
    image_copy.fill(128)
    
    _nStep = int(N / 2)
    image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW] = img[0:nH, 0:nW]
    pixel_error_idx = np.zeros((nH+(int(N/2)*2),nW+(int(N/2)*2)), dtype=np.uint8)
    
    for i in range(_nStep, nH+_nStep, 1):
        for j in range(_nStep, nW+_nStep, 1):
                crop_img = np.zeros((N, N))
                crop_img = image_copy[i-_nStep:(i-_nStep)+N, j-_nStep:(j-_nStep)+N]
                
                if REC_TYPE == "DPD_D":
                    _status, P_rec = DPD_D_detect(crop_img, i, j, N)
                else: 
                    _status, P_rec = DPD_M_detect(crop_img, i, j, N)
                
                if _status == 'good':
                    pass
                else: 
                    #print('Image recovered y={},x={}, org_val={}, rec_val={}'.format(y,x,image_copy[y,x],P_rec))
                    pixel_error_idx[i,j] = 255
                    image_copy[i, j] = P_rec
                    
    #cv2.imwrite('./recovered_image_padded.png', image_copy)
    #fnameIdx = recFname[:-3] + 'csv'
    #np.save(fnameIdx, pixel_error_idx[_nStep:_nStep+nH, _nStep:_nStep+nW])
    #pixel_error_idx[_nStep:_nStep+nH, _nStep:_nStep+nW].tofile(fnameIdx, sep=',')
    cv2.imwrite(recFname, image_copy[_nStep:_nStep+nH, _nStep:_nStep+nW])
    cv2.imwrite(recFnameIdx, pixel_error_idx[_nStep:_nStep+nH, _nStep:_nStep+nW])
    
if __name__ == '__main__':
    gray_scale = 1
    
    for _noiseDir in tqdm(noiseDir, desc="Noise type"):
        #get sorted folders
        imgDir = natsorted(glob(os.path.join(_noiseDir, '*.png')))
        print('\n\n Noise Type: {}'.format(os.path.split(_noiseDir)[-1]))
        
        if gray_scale:        
            #Parallel(n_jobs=args.num_cores)(delayed(recover_defected_pixel_gray_scale)(idx=fname, dir=imgDir) for fname in tqdm(range(0,len(imgDir)), desc="    Images"))
            Parallel(n_jobs=args.num_cores)(delayed(recover_defected_pixel_rgb__gray_scale)(idx=fname, dir=imgDir) for fname in tqdm(range(0,len(imgDir)), desc="    Images"))
            
            # for file_ in imgDir:
            #     filename = os.path.split(file_)[-1]
            #     print(filename)
            #     img = cv2.imread(file_, )
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            #     #noise_img = add_column_row_noise(img)
            #     recFname = os.path.join(args.tar, os.path.split(_noiseDir)[-1], filename)
            #     recover_defected_pixel_gray_scale(img, recFname)
        else: 
            Parallel(n_jobs=args.num_cores)(delayed(recover_defected_pixel_rgb_scale)(idx=fname, dir=imgDir) for fname in tqdm(range(0,len(imgDir)), desc="    Images"))