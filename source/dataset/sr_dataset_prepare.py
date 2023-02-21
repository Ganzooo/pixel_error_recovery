import os
import numpy 
import glob

import os.path as osp
import numpy as np
import imageio


def add_noise(image, sigma=50):
    """
    image: input image, numpy array, dtype=uint8, range=[0, 255]
    sigma: default 50
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, sigma / 255, image.shape)
    gauss_noise = image + noise
    return gauss_noise * 255

def save_image(image, path):
    """
    image: saved image, numpy array, dtype=float
    path: saving path
    """
    # The type of the image is float, and range of the image might not be in [0, 255]
    # Thus, before saving the image, the image needs to be clipped.
    image = np.round(np.clip(image, 0, 255)).astype(np.uint8)
    imageio.imwrite(path, image)

# gt_image_path = '/dataset/SR/Denoise/DIV2K/DIV2K_train_HR/'
# noise_image_path = '/dataset/SR/Denoise/DIV2K/DIV2K_train_HR_noise/'
from PIL import Image

gt_image_path = '/dataset/SR/All/train_HR/'
noise_image_path = '/dataset/SR/All/train_LR/'

gt_images = sorted(glob.glob(gt_image_path + '*.png'))
for idx, image_name in enumerate(gt_images):
    #img = imageio.imread(image_name)
    img = Image.open(image_name)
    # This image is used as the noisy image for training.
    #img_noise = add_noise(img, sigma=50)
    w, h = img.size
    try:
        assert w % 3 == 0
        assert h % 3 == 0

        img = img.resize((int(w/3), int(h/3)), resample=Image.BICUBIC)
        fname = os.path.join(noise_image_path,os.path.basename(image_name))[:-4]
        img.save(fname + '.jpg', "JPEG", quality=90)
    except:
        pass

    
    
    # This function ensures that the image is properly clipped and rounded before saving.
    #save_image(img_noise, fname)