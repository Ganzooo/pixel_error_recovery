python test.py --multirun hydra.sweep.dir=test/module hydra.sweep.subdir=6 model.num_module=6 train_config.testmodelpath=/workspace/pixel_error_recovery/multirun/module/6/last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/module hydra.sweep.subdir=8 model.num_module=8 train_config.testmodelpath=/workspace/pixel_error_recovery/multirun/module/8/last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/avtivation hydra.sweep.subdir='relu' model.activation='relu' train_config.testmodelpath=/workspace/pixel_error_recovery/multirun/activation/relu/last_epoch.pt # base model
python test.py --multirun hydra.sweep.dir=test/avtivation hydra.sweep.subdir='gelu' model.activation='gelu' train_config.testmodelpath=/workspace/pixel_error_recovery/multirun/activation/gelu/last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/channel hydra.sweep.subdir=48 model.channel_nums=48 train_config.testmodelpath=/workspace/pixel_error_recovery/multirun/channel/48/last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/channel hydra.sweep.subdir=64 model.channel_nums=64 train_config.testmodelpath=/workspace/pixel_error_recovery/multirun/channel/64/last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/channel hydra.sweep.subdir=128 model.channel_nums=128 train_config.testmodelpath=/workspace/pixel_error_recovery/multirun/channel/128/last_epoch.pt
