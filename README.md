# Baseline Trainer Code for Real-Time Super Resolution
An older version implemented based on [ECBSR](https://github.com/xindongzhang/ECBSR#readme) placed in legacy folder.

The following is more advanced version implemented by us.

### Dependencies & Installation

Please refer to the following simple steps for installation.

```
git clone https://github.com/mai2022SR/baseline_trainer_sr
cd baseline_trainer_sr
pip install -r requirements.txt
```
### Dataset of SR

You can download Div2k dataset from Web [Link](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)

You can download Benchmark dataset from Web [Link] (https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

Path of Dataset must be set in ./config/dataset/main.yaml

### Training & Testing
You could also try less/larger batch-size, if there are limited/enough hardware resources in your GPU-server.
We use Hydra packadge for various settings during training. 
You can set all Parameters in ***./config/config.yaml***
```
cd baseline_trainer_sr

## If you set all settings correct
python train.py

## If you want to change optimizer, batch size and epoch from argument
python train.py optimizer=sgd train_config.batch_size=1024 train_config.epochs=1000
```

### Check Result

Validation result image, Best weight, Last weight and log files saved in this ***./output/{DATE_of_TODAY}/{Last_folder}*** folder.

Wandb result [WANDB](https://wandb.ai/iilab/ECCV_MAI2020_SR)