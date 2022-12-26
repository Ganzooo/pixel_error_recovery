import torch
from .resnet50_unet import UNetWithResnet50Encoder, UNetWithResnet50Hybrid
from .hardnet import hardnet
from .DDRNet_23_slim import DualResNet_imagenet

def get_model(cfg, device):
    if cfg.model.name == 'unet_res50':
        #model = UNetWithResnet50Encoder(n_classes=cfg.train_config.nclass).to(device)
        raise NameError('Choose proper model name!!!')
    elif cfg.model.name == 'hardnet':
        model = hardnet(n_classes=cfg.train_config.nclass, in_channels = 3).to(device)
    elif cfg.model.name == 'DualResNet_imagenet':
        model = DualResNet_imagenet(pretrained=False, n_classes=cfg.train_config.nclass).to(device)
    elif cfg.model.name == 'UNetWithResnet50Encoder':
        model = UNetWithResnet50Encoder(in_channel=cfg.train_config.colors,n_classes=cfg.train_config.nclass).to(device)
    elif cfg.model.name == 'UNetWithResnet50Hybrid':
        model = UNetWithResnet50Hybrid(in_channel=cfg.train_config.colors,n_classes=cfg.train_config.nclass).to(device)
    else: 
        raise NameError('Choose proper model name!!!')
    
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
