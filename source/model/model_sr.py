import torch
from .imdn_baseline import IMDN

def get_model_sr(cfg, device):
    if cfg.model.name == 'IMDN':
        model = IMDN(in_nc=3, out_nc=3, nc=64, nb=8, upscale=3, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05)
    else: 
        raise NameError('Choose proper model name!!!')
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
