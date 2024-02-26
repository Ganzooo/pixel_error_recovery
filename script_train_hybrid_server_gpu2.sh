python hydra.sweep.dir=trained/channel hydra.sweep.subdir=48 train_detect_pixel.py --multirun model.channel_nums=48 train_config.gpu_id=2
python hydra.sweep.dir=trained/channel hydra.sweep.subdir=64 train_detect_pixel.py --multirun model.channel_nums=64 train_config.gpu_id=2
python hydra.sweep.dir=trained/channel hydra.sweep.subdir=128 train_detect_pixel.py --multirun model.channel_nums=128 train_config.gpu_id=2
