python train_detect_pixel.py --multirun hydra.sweep.dir=trained/channel hydra.sweep.subdir=48 model.channel_nums=48 train_config.gpu_id=2
python train_detect_pixel.py --multirun hydra.sweep.dir=trained/channel hydra.sweep.subdir=64 model.channel_nums=64 train_config.gpu_id=2
python train_detect_pixel.py --multirun hydra.sweep.dir=trained/channel hydra.sweep.subdir=128 model.channel_nums=128 train_config.gpu_id=2
