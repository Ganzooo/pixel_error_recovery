import torch
from .resnet50_unet import UNetWithResnet50Encoder, UNetWithResnet50Hybrid, UNetWithResnet50HybridV2
from .hardnet import hardnet
from .DDRNet_23_slim import DualResNet_imagenet
from .plainNetwork import plainDP, plainRP, plainHYBRID
from .NAFNet import NAFNet

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
    elif cfg.model.name == 'UNetWithResnet50HybridV2':
        model = UNetWithResnet50HybridV2(in_channel=cfg.train_config.colors,n_classes=cfg.train_config.nclass).to(device)
    elif cfg.model.name == 'plainDP':
        model = plainDP(module_nums=cfg.model.num_module, channel_nums=cfg.model.channel_nums, num_class=cfg.train_config.nclass, act_type='relu', colors=cfg.train_config.colors, use_bn=cfg.model.use_bn)
    elif cfg.model.name == 'plainRP':
        model = plainRP(module_nums=cfg.model.num_module, channel_nums=cfg.model.channel_nums, act_type='relu', colors=cfg.train_config.colors, use_bn=cfg.model.use_bn)
    elif cfg.model.name == 'plainHYBRID':
        model = plainHYBRID(module_nums=cfg.model.num_module, channel_nums=cfg.model.channel_nums, act_type=cfg.model.activation, colors=cfg.train_config.colors, use_bn_det=cfg.model.use_bn_det, use_bn_rec=cfg.model.use_bn_rec, rec_mode=cfg.model.rec_mode)
    elif cfg.model.name == 'NAFNet':
        model = NAFNet(img_channel=cfg.model.in_channel, width=cfg.model.width, middle_blk_num=cfg.model.middle_blk_num, enc_blk_nums=cfg.model.enc_blks, dec_blk_nums=cfg.model.dec_blks)
    else: 
        raise NameError('Choose proper model name!!!')
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
