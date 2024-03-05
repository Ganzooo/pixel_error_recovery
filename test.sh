python test.py --multirun hydra.sweep.dir=test_4k/avtivation hydra.sweep.subdir='relu' model.activation='relu' train_config.testmodelpath=/workspace/pixel_error_recovery/models/relu_best_epoch.pt # base model
python test.py --multirun hydra.sweep.dir=test_4k/avtivation hydra.sweep.subdir='gelu' model.activation='gelu' train_config.testmodelpath=/workspace/pixel_error_recovery/models/gelu_best_epoch.pt
python test.py --multirun hydra.sweep.dir=test_4k/channel hydra.sweep.subdir=48 model.channel_nums=48 train_config.testmodelpath=/workspace/pixel_error_recovery/models/ch48_best_epoch.pt
python test.py --multirun hydra.sweep.dir=test_4k/channel hydra.sweep.subdir=64 model.channel_nums=64 train_config.testmodelpath=/workspace/pixel_error_recovery/models/ch64_best_epoch.pt
python test.py --multirun hydra.sweep.dir=test_4k/channel hydra.sweep.subdir=128 model.channel_nums=128 train_config.testmodelpath=/workspace/pixel_error_recovery/models/ch128_best_epoch.pt
python test.py --multirun hydra.sweep.dir=test_4k/module hydra.sweep.subdir=6 model.num_module=6 train_config.testmodelpath=/workspace/pixel_error_recovery/models/m6_best_epoch.pt
python test.py --multirun hydra.sweep.dir=test_4k/module hydra.sweep.subdir=8 model.num_module=8 train_config.testmodelpath=/workspace/pixel_error_recovery/models/m8_best_epoch.pt

#python test.py --multirun hydra.sweep.dir=test_4k/ps hydra.sweep.subdir=factor2 model.ps_scale=2 train_config.testmodelpath=/workspace/pixel_error_recovery/models/ps_f2_best_epoch.pt
#python test.py --multirun hydra.sweep.dir=test_4k/ps hydra.sweep.subdir=factor4 model.ps_scale=4 model.channel_nums=64 train_config.testmodelpath=/workspace/pixel_error_recovery/models/ps_f4_best_epoch.pt
