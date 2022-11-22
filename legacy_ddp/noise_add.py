from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
#import imutils
import math
import random

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--tar_gt', default='/dataset/Cityscapes/DEFECTION_NOISE_PAPER/gt_final',type=str, help='Directory of Converted GrayScale GT images')
#parser.add_argument('--src_dir', default='/dataset2/Cityscapes/leftImg8bit/train_all/', type=str, help='Directory of GT images')
#parser.add_argument('--tar_dir', default='/dataset2/Cityscapes/DEFECTION_NOISE_PAPER/noise_paper_final_ch1',type=str, help='Directory of Noise generated images')
parser.add_argument('--src_dir', default='/dataset/Cityscapes/leftImg8bit/val_100/', type=str, help='Directory of GT images')
parser.add_argument('--tar_dir', default='/dataset/Cityscapes/DEFECTION_NOISE_PAPER/noise_paper_val_final_CH1',type=str, help='Directory of Noise generated images')
parser.add_argument('--num_cores', default=20, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
tar_gt = args.tar_gt

COLOR_NOISE = False
NUM_CORES = args.num_cores
COLOR_RANDOM = (0,0,0)
COLOR_DEAD_PIXEL_BLACK = 0
COLOR_DEAD_PIXEL_WHITE = 255
RESIZE_IMG = True
GRAY_IMG = True

__DEBUG__ = 0
if os.path.exists(tar):
    os.system("rm -r {}".format(tar))
os.makedirs(tar)

os.makedirs(tar + '/pr_0_5/')
os.makedirs(tar + '/pr_0_5/index/')
os.makedirs(tar + '/pr_0_5/index_color/')

os.makedirs(tar + '/pr_1_0/')
os.makedirs(tar + '/pr_1_0/index/')
os.makedirs(tar + '/pr_1_0/index_color/')

# os.makedirs(tar + '/pr_5_0/')
# os.makedirs(tar + '/pr_5_0/index/')
# os.makedirs(tar + '/pr_5_0/index_color/')

# os.makedirs(tar + '/pr_10_0/')
# os.makedirs(tar + '/pr_10_0/index/')
# os.makedirs(tar + '/pr_10_0/index_color/')

# os.makedirs(tar + '/pr_30_0/')
# os.makedirs(tar + '/pr_30_0/index/')
# os.makedirs(tar + '/pr_30_0/index_color/')

os.makedirs(tar + '/col_1/')
os.makedirs(tar + '/col_1/index/')
os.makedirs(tar + '/col_1/index_color/')

os.makedirs(tar + '/col_2/')
os.makedirs(tar + '/col_2/index/')
os.makedirs(tar + '/col_2/index_color/')

os.makedirs(tar + '/cluster_2/')
os.makedirs(tar + '/cluster_2/index/')
os.makedirs(tar + '/cluster_2/index_color/')

os.makedirs(tar + '/cluster_3/')
os.makedirs(tar + '/cluster_3/index')
os.makedirs(tar + '/cluster_3/index_color')

#get sorted folders
imgDir = natsorted(glob(os.path.join(src, '*.png')))

img_files, label_files = [], []
for _idx, file_ in enumerate(imgDir):
    #if _idx < 1000:
    filename = os.path.split(file_)[-1]
    img_files.append(file_)

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def noise_add(iter):
    imgName = os.path.join(tar,os.path.basename(img_files[iter]))
    imgSrc = cv2.imread(img_files[iter])
    nH ,nW, _ = imgSrc.shape

    random_col_row_sp = random.randint(0 , 3)
    #0 - column noise
    #1 - row noise
    #2 - salt paper noise

    pixel_error_idx = np.zeros((nH,nW), dtype=np.uint8)
    if random_col_row_sp == 0:
        random_pixel = random.randint(0 , nW-1)
        COLOR_RANDOM = random.randint(1,254), random.randint(1,254),random.randint(1,254)
        if COLOR_NOISE:
            imgSrc[:,random_pixel,:] = COLOR_RANDOM
        else:
            imgSrc[:,random_pixel,:] = (0,0,0)
        ###Save index
        pixel_error_idx[:,random_pixel] = 255
        if __DEBUG__:
            cv2.imshow('Column noise',imgSrc)
            cv2.waitKey(0)
    elif random_col_row_sp == 1:
        random_pixel = random.randint(0 , nH-1)
        COLOR_RANDOM = random.randint(1,254), random.randint(1,254),random.randint(1,254)

        if COLOR_NOISE:
            imgSrc[random_pixel,:,:] = COLOR_RANDOM
        else:
            imgSrc[random_pixel,:,:] = (0,0,0)

        ###Save index
        pixel_error_idx[random_pixel,:] = 255
        if __DEBUG__:
            cv2.imshow('ROW noise',imgSrc)
            cv2.waitKey(0)
    else:
        number_of_pixels = random.randint(300 , min(30000,nH*nW))
        for i in range(number_of_pixels):
            # Pick a random x,y coordinate
            yCoord=random.randint(0, nW - 1)
            xCoord=random.randint(0, nH - 1)

            COLOR_RANDOM = random.randint(1,254), random.randint(1,254),random.randint(1,254)
            # Color that pixel to black

            if COLOR_NOISE:
                imgSrc[xCoord,yCoord,:] = COLOR_RANDOM
            else:
                imgSrc[xCoord,yCoord,:] = (0,0,0)

            ###Save index
            pixel_error_idx[xCoord,yCoord] = 255

        if __DEBUG__:
            cv2.imshow('SP noise',imgSrc)
            cv2.waitKey(0)
    cv2.imwrite(os.path.join(tar, 'noise_{}'.format(os.path.basename(img_files[iter]))), imgSrc)
    np.save(os.path.join(tar, 'noise_idx_{}'.format(os.path.basename(img_files[iter]))), pixel_error_idx)

def save_gt(iter, gray=False):
    _imgSrc = cv2.imread(img_files[iter])
    #resize
    if RESIZE_IMG:
        _imgSrc = cv2.resize(_imgSrc, dsize=(1024,512))
    
    fnanmeOrg = os.path.join(tar_gt,os.path.basename(img_files[iter]))
    
    if GRAY_IMG:
        imgSrc = cv2.cvtColor(_imgSrc, cv2.COLOR_BGR2GRAY) 
    else: 
        imgSrc = _imgSrc
    
    cv2.imwrite(fnanmeOrg,imgSrc)
    
def noise_add_percent(iter, per, gray=False):
    _imgSrc = cv2.imread(img_files[iter])
    
    #resize
    if RESIZE_IMG:
        _imgSrc = cv2.resize(_imgSrc, dsize=(1024,512))
    
    if GRAY_IMG:
        imgSrcGray = cv2.cvtColor(_imgSrc, cv2.COLOR_BGR2GRAY)
        imgSrc = imgSrcGray.copy()
        nH, nW = imgSrc.shape
    else: 
        imgSrc = _imgSrc.copy()
        nH, nW, _ = imgSrc.shape
        
    pixel_error_idx = np.zeros((nH,nW), dtype=np.uint8)
    pixel_error_idx_test = np.zeros((nH,nW), dtype=np.uint8)
    erroPixelNumber = int((nH * nW * per)/100)
    print(erroPixelNumber)
    
    for i in range(erroPixelNumber):
        # Pick a random x,y coordinate
        yCoord=random.randint(0, nW - 1)
        xCoord=random.randint(0, nH - 1)

        dead_pixel_type = random.randint(0,1) #0 - Black, 1 - White
        #dead_pixel_type = 0
        if GRAY_IMG:
            if dead_pixel_type == 0:
                imgSrc[xCoord,yCoord] = COLOR_DEAD_PIXEL_BLACK
            else: 
                imgSrc[xCoord,yCoord] = COLOR_DEAD_PIXEL_WHITE
        else: 
            if dead_pixel_type == 0:
                imgSrc[xCoord,yCoord] = (0,0,0)
                #imgSrc[xCoord,yCoord] = COLOR_DEAD_PIXEL_BLACK
            else: 
                #imgSrc[xCoord,yCoord] = COLOR_DEAD_PIXEL_WHITE
                imgSrc[xCoord,yCoord] = (255,255,255)

        ###Save index
        pixel_error_idx[xCoord,yCoord] = 1
        pixel_error_idx_test[xCoord,yCoord] = 200

        if __DEBUG__:
            cv2.imshow('SP noise',imgSrc)
            cv2.waitKey(0)
    
    if per == 10.0 or per == 30.0:
        fnameImg = os.path.join(tar,'pr_{}'.format(str(per)[0:2]+'_'+str(per)[3]), 'noise_pr_{}_{}'.format(str(per)[0:2]+'_'+str(per)[3],os.path.basename(img_files[iter])))
        fnameIdx = os.path.join(tar,'pr_{}'.format(str(per)[0:2]+'_'+str(per)[3]),'index', 'noise_pr_{}_{}.png'.format(str(per)[0:2]+'_'+str(per)[3],os.path.basename(img_files[iter])[:-4]))
        fnameIdx_text = os.path.join(tar,'pr_{}'.format(str(per)[0:2]+'_'+str(per)[3]),'index_color', 'noise_pr_{}_{}.png'.format(str(per)[0:2]+'_'+str(per)[3],os.path.basename(img_files[iter])[:-4]))
    else:
        fnameImg = os.path.join(tar,'pr_{}'.format(str(per)[0]+'_'+str(per)[2]), 'noise_pr_{}_{}'.format(str(per)[0]+'_'+str(per)[2],os.path.basename(img_files[iter])))
        fnameIdx = os.path.join(tar,'pr_{}'.format(str(per)[0]+'_'+str(per)[2]),'index', 'noise_pr_{}_{}.png'.format(str(per)[0]+'_'+str(per)[2],os.path.basename(img_files[iter])[:-4]))
        fnameIdx_text = os.path.join(tar,'pr_{}'.format(str(per)[0]+'_'+str(per)[2]),'index_color', 'noise_pr_{}_{}.png'.format(str(per)[0]+'_'+str(per)[2],os.path.basename(img_files[iter])[:-4]))
    
    cv2.imwrite(fnameImg, imgSrc)   
    cv2.imwrite(fnameIdx, pixel_error_idx)
    cv2.imwrite(fnameIdx_text, pixel_error_idx_test)
    
    #pixel_error_idx.tofile(fnameIdx, sep=',')
    #np.tofile(fnameIdx, pixel_error_idx)

def noise_add_cluster(iter, cluster):
    imgSrc = cv2.imread(img_files[iter])
    
    #resize
    if RESIZE_IMG:
        imgSrc = cv2.resize(imgSrc, dsize=(1024,512))
    
    if GRAY_IMG:
        imgSrc = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY) 
        nH, nW = imgSrc.shape
    else: 
        nH, nW, _ = imgSrc.shape
        
    erroPixelNumber = 1000 # Type 3: 100 sets of cluster [2x2], [3x3]
    
    
    pixel_error_idx = np.zeros((nH,nW), dtype=np.uint8)
    pixel_error_idx_test = np.zeros((nH,nW), dtype=np.uint8)

    for i in range(erroPixelNumber):
        # Pick a random x,y coordinate
        yCoord=random.randint(cluster, nW - 1 - cluster)
        xCoord=random.randint(cluster, nH - 1 - cluster)

        dead_pixel_type = random.randint(0,1)
        if GRAY_IMG:
            if dead_pixel_type == 0:
                dead_pixel = np.zeros([cluster,cluster])
                imgSrc[xCoord:xCoord+cluster,yCoord:yCoord+cluster] = dead_pixel
            else: 
                dead_pixel = np.zeros([cluster,cluster])
                dead_pixel.fill(255)
                imgSrc[xCoord:xCoord+cluster,yCoord:yCoord+cluster] = dead_pixel    
        else: 
            if dead_pixel_type == 0:
                dead_pixel = np.zeros([cluster,cluster, 3])
                imgSrc[xCoord:xCoord+cluster,yCoord:yCoord+cluster, :] = dead_pixel
            else: 
                dead_pixel = np.zeros([cluster,cluster,3])
                dead_pixel.fill(255)
                imgSrc[xCoord:xCoord+cluster,yCoord:yCoord+cluster,:] = dead_pixel

        ###Save index
        pixel_error_idx[xCoord:xCoord+cluster,yCoord:yCoord+cluster] = 1
        pixel_error_idx_test[xCoord:xCoord+cluster,yCoord:yCoord+cluster] = 200

        if __DEBUG__:
            cv2.imshow('SP noise',imgSrc)
            cv2.waitKey(0)
    fnameImg = os.path.join(tar,'cluster_{}'.format(str(cluster)), 'noise_cluster_{}_{}'.format(cluster,os.path.basename(img_files[iter])))
    #fnameIdx = os.path.join(tar,'cluster_{}'.format(str(cluster)), 'noise_cluster_{}_{}.npy'.format(cluster,os.path.basename(img_files[iter])[:-4]))
    cv2.imwrite(fnameImg, imgSrc)
    
    fnameIdx = os.path.join(tar,'cluster_{}'.format(str(cluster)),'index', 'noise_cluster_{}_{}.png'.format(cluster,os.path.basename(img_files[iter])[:-4]))
    fnameIdx_text = os.path.join(tar,'cluster_{}'.format(str(cluster)),'index_color', 'noise_cluster_{}_{}.png'.format(cluster,os.path.basename(img_files[iter])[:-4]))
    
    cv2.imwrite(fnameIdx, pixel_error_idx)
    cv2.imwrite(fnameIdx_text, pixel_error_idx_test)
    #np.save(fnameIdx, pixel_error_idx)
    
def noise_add_column(iter, col):
    imgSrc = cv2.imread(img_files[iter])
    
    #resize
    if RESIZE_IMG:
        imgSrc = cv2.resize(imgSrc, dsize=(1024,512))
    
    if GRAY_IMG:
        imgSrc = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY) 
        nH ,nW = imgSrc.shape
    else: 
        nH ,nW, _ = imgSrc.shape

    pixel_error_idx = np.zeros((nH,nW), dtype=np.uint8)
    pixel_error_idx_test = np.zeros((nH,nW), dtype=np.uint8)
                                    
    xCoord=random.randint(0, nH - 1 - col)
    
    dead_pixel_type = random.randint(0,1)
    
    if GRAY_IMG:
        if dead_pixel_type == 0:
            dead_pixel = np.zeros([col,nW])
            imgSrc[xCoord:xCoord+col,:] = dead_pixel
        else: 
            dead_pixel = np.zeros([col,nW])
            dead_pixel.fill(255)
            imgSrc[xCoord:xCoord+col,:] = dead_pixel    
    else: 
        if dead_pixel_type == 0:
            dead_pixel = np.zeros([col,nW, 3])
            imgSrc[xCoord:xCoord+col,:, :] = dead_pixel
        else: 
            dead_pixel = np.zeros([col,nW,3])
            dead_pixel.fill(255)
            imgSrc[xCoord:xCoord+col,:,:] = dead_pixel
                
    pixel_error_idx[xCoord:xCoord+col,:] = 1
    pixel_error_idx_test[xCoord:xCoord+col,:] = 200

    if __DEBUG__:
        cv2.imshow('SP noise',imgSrc)
        cv2.waitKey(0)
        
    fnameImg = os.path.join(tar,'col_{}'.format(str(col)), 'noise_col_{}_{}'.format(col,os.path.basename(img_files[iter])))
    #fnameIdx = os.path.join(tar,'col_{}'.format(str(col)), 'noise_col_{}_{}.csv'.format(col,os.path.basename(img_files[iter])[:-4]))
    cv2.imwrite(fnameImg, imgSrc)
    
    fnameIdx = os.path.join(tar,'col_{}'.format(str(col)),'index', 'noise_col_{}_{}.png'.format(col,os.path.basename(img_files[iter])[:-4]))
    fnameIdx_text = os.path.join(tar,'col_{}'.format(str(col)),'index_color', 'noise_col_{}_{}.png'.format(col,os.path.basename(img_files[iter])[:-4]))
    cv2.imwrite(fnameIdx, pixel_error_idx)
    cv2.imwrite(fnameIdx_text, pixel_error_idx_test)
    
def noise_add_row(iter, row):
    imgSrc = cv2.imread(img_files[iter])
    
    #resize
    if RESIZE_IMG:
        imgSrc = cv2.resize(imgSrc, dsize=(1024,512))
    
    if GRAY_IMG:
        imgSrc = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY) 
        nH ,nW = imgSrc.shape
    else: 
        nH ,nW, _ = imgSrc.shape

    pixel_error_idx = np.zeros((nH,nW), dtype=np.uint8)
    pixel_error_idx_test = np.zeros((nH,nW), dtype=np.uint8)
                                    
    xCoord=random.randint(0, nH - 1 - row)
    yCoord=random.randint(0, nW - 1 - row)
    
    dead_pixel_type = random.randint(0,1)
    if GRAY_IMG:
        if dead_pixel_type == 0:
            dead_pixel = np.zeros([nH,row])
            imgSrc[:,yCoord:yCoord+row] = dead_pixel
        else: 
            dead_pixel = np.zeros([nH,row])
            dead_pixel.fill(255)
            imgSrc[:,yCoord:yCoord+row] = dead_pixel    
    else: 
        if dead_pixel_type == 0:
            dead_pixel = np.zeros([nH,row, 3])
            imgSrc[:,yCoord:yCoord+row, :] = dead_pixel
        else: 
            dead_pixel = np.zeros([nH,row,3])
            dead_pixel.fill(255)
            imgSrc[:,yCoord:yCoord+row,:] = dead_pixel
                
    pixel_error_idx[:,yCoord:yCoord+row] = 1
    pixel_error_idx_test[:,yCoord:yCoord+row] = 200

    if __DEBUG__:
        cv2.imshow('SP noise',imgSrc)
        cv2.waitKey(0)
        
    fnameImg = os.path.join(tar,'row_{}'.format(str(row)), 'noise_row_{}_{}'.format(row,os.path.basename(img_files[iter])))
    #fnameIdx = os.path.join(tar,'col_{}'.format(str(col)), 'noise_col_{}_{}.csv'.format(col,os.path.basename(img_files[iter])[:-4]))
    cv2.imwrite(fnameImg, imgSrc)
    
    fnameIdx = os.path.join(tar,'row_{}'.format(str(row)),'index', 'noise_row_{}_{}.png'.format(row,os.path.basename(img_files[iter])[:-4]))
    fnameIdx_text = os.path.join(tar,'row_{}'.format(str(row)),'index_color', 'noise_row_{}_{}.png'.format(row,os.path.basename(img_files[iter])[:-4]))
    cv2.imwrite(fnameIdx, pixel_error_idx)
    cv2.imwrite(fnameIdx_text, pixel_error_idx_test)

    
#Save Gt:
for i in tqdm(range(0,len(img_files))):
    save_gt(i)

#for i in tqdm(range(0,len(img_files))):
#    noise_add_percent(i, per=5.0)
#Parallel(n_jobs=NUM_CORES)(delayed(noise_add_percent)(i, per=5.0) for i in tqdm(range(len(img_files))))
Parallel(n_jobs=NUM_CORES)(delayed(noise_add_percent)(i, per=1.0) for i in tqdm(range(len(img_files))))
Parallel(n_jobs=NUM_CORES)(delayed(noise_add_percent)(i, per=0.5) for i in tqdm(range(len(img_files))))

Parallel(n_jobs=NUM_CORES)(delayed(noise_add_cluster)(i, cluster=2) for i in tqdm(range(len(img_files))))
Parallel(n_jobs=NUM_CORES)(delayed(noise_add_cluster)(i, cluster=3) for i in tqdm(range(len(img_files))))
#Parallel(n_jobs=NUM_CORES)(delayed(noise_add_cluster)(i, cluster=5) for i in tqdm(range(len(img_files))))

Parallel(n_jobs=NUM_CORES)(delayed(noise_add_column)(i, col=1) for i in tqdm(range(len(img_files))))
Parallel(n_jobs=NUM_CORES)(delayed(noise_add_column)(i, col=2) for i in tqdm(range(len(img_files))))

#Parallel(n_jobs=NUM_CORES)(delayed(noise_add_row)(i, row=1) for i in tqdm(range(len(img_files))))
#Parallel(n_jobs=NUM_CORES)(delayed(noise_add_row)(i, row=2) for i in tqdm(range(len(img_files))))


# for i in tqdm(range(0,len(img_files))):
#     noise_add_percent(i, per=30.0)
    
#TYPE 1: 0.5 percent random error
# for i in tqdm(range(0,len(img_files))):
#     noise_add_percent(i, per=0.5)

#TYPE 2: 1.0 percent random error
# for i in tqdm(range(0,len(img_files))):
#     noise_add_percent(i, per=1.0) 
    
#TYPE 3: 100 sets of 2x2 random error
# for i in tqdm(range(0,len(img_files))):
#     noise_add_cluster(i, cluster=2)

#TYPE 4: 100 sets of 2x2 random error
# for i in tqdm(range(0,len(img_files))):
#     noise_add_cluster(i, cluster=3)

#TYPE 5: 1 column random error
# for i in tqdm(range(0,len(img_files))):
#     noise_add_column(i, col=1)

#TYPE 6: 2 column random error   
# for i in tqdm(range(0,len(img_files))):
#     noise_add_column(i, col=2)

# for i in tqdm(range(0,len(img_files))):
#     noise_add(i) 
    
#Parallel(n_jobs=NUM_CORES)(delayed(image_overlap)(i) for i in tqdm(range(len(img_files))))
#[save_files_stride(i) for i in tqdm(range(len(noisy_files)))]
#Parallel(n_jobs=NUM_CORES)(delayed(save_files_random)(i) for i in tqdm(range(len(noisy_files))))
#Parallel(n_jobs=NUM_CORES)(delayed(save_files_stride)(i) for i in tqdm(range(len(noisy_files))))
