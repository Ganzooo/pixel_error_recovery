python test.py --multirun hydra.sweep.dir=test/module hydra.sweep.subdir=6 model.num_module=6 train_config.testmodelpath=/workspace/pixel_error_recovery/models/m6_last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/module hydra.sweep.subdir=8 model.num_module=8 train_config.testmodelpath=/workspace/pixel_error_recovery/models/m8_last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/avtivation hydra.sweep.subdir='relu' model.activation='relu' train_config.testmodelpath=/workspace/pixel_error_recovery/models/relu_last_epoch.pt # base model
python test.py --multirun hydra.sweep.dir=test/avtivation hydra.sweep.subdir='gelu' model.activation='gelu' train_config.testmodelpath=/workspace/pixel_error_recovery/models/gelu_last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/channel hydra.sweep.subdir=48 model.channel_nums=48 train_config.testmodelpath=/workspace/pixel_error_recovery/models/ch48_last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/channel hydra.sweep.subdir=64 model.channel_nums=64 train_config.testmodelpath=/workspace/pixel_error_recovery/models/ch64_last_epoch.pt
python test.py --multirun hydra.sweep.dir=test/channel hydra.sweep.subdir=128 model.channel_nums=128 train_config.testmodelpath=/workspace/pixel_error_recovery/models/ch128_last_epoch.pt
