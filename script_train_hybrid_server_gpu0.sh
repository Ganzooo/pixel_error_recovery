#python train_detect_pixel.py --multirun hydra.sweep.dir=trained/module hydra.sweep.subdir=6 model.num_module=6 train_config.gpu_id=0
#python train_detect_pixel.py --multirun hydra.sweep.dir=trained/module hydra.sweep.subdir=8  model.num_module=8 train_config.gpu_id=0
python train_detect_pixel.py --multirun hydra.sweep.dir=multirun/ps train_config.gpu_id=0
