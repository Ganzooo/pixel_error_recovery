# Baseline Trainer Code for Pixel Error Detection + Pixel recovery
An older version implemented based on tradition method placed in legacy folder.

The following is based Deep learning method with Hybrid version implemented by us.
![alt text](https://github.com/Ganzooo/pixel_error_recovery/blob/main/data/lightweight_model_hybrid_architecture.png)

### Dependencies & Installation

Please refer to the following simple steps for installation.

```
git clone https://github.com/Ganzooo/pixel_error_recovery
cd pixel_error_recovery
pip install -r requirements.txt
```
### Dataset of Pixel Recovery & Prepare Noise dataset.

You can download Cityscape dataset from Web [Link](https://www.cityscapes-dataset.com/login/)

For prepare noise dataset:
 - need to set correct path.
```
cd ./legacy_ddp

## If you set all settings correct
python noise_add.py
```


Path of Dataset must be set in ./config/dataset/main.yaml


### Training & Testing
You could also try less/larger batch-size, if there are limited/enough hardware resources in your GPU-server.
We use Hydra packadge for various settings during training. 
You can set all Parameters in ***./config/config.yaml***
```
cd pixel_error_recovery

## If you set all settings correct
python train_hybrid.py

## If you want to change optimizer, batch size and epoch from argument
python train_hybrid.py optimizer=sgd train_config.batch_size=1024 train_config.epochs=1000

## For Test
python test.py
```

### Result
Validation result image, Best weight, Last weight and log files saved in this ***./output/{DATE_of_TODAY}/{Last_folder}*** folder.


- (a) GT error image index,           (b) Input image with error pixel          (c) Pred error image            (d) Recovered image
![alt text](https://github.com/Ganzooo/pixel_error_recovery/blob/main/data/result_img_gt_pred.jpg)