import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='binary', ignore_index=250)
FocalLoss   = smp.losses.FocalLoss(mode='multilabel')
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

class BCEandTversky(nn.Module):
    def __init__(self):
        super(BCEandTversky, self).__init__()
        self.w1 = 0.5
        self.w2 = 0.5
    def forward(self, y_pred, y_true):
        return self.w1*BCELoss(y_pred, y_true) + self.w2*TverskyLoss(y_pred, y_true)
    
class FocalandTversky(nn.Module):
    def __init__(self):
        super(FocalandTversky, self).__init__()
        self.w1 = 0.5
        self.w2 = 0.5
    def forward(self, y_pred, y_true):
        return self.w1*FocalLoss(y_pred, y_true) + self.w2*TverskyLoss(y_pred, y_true)

class bootstrapped_cross_entropy2d(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=True, reduce=False, reduction='mean', ignore_index=250):
        super(bootstrapped_cross_entropy2d, self).__init__(size_average, reduce, reduction)
        self.size_average = size_average
        self.reduce = False
        
    def forward(self, input, target, min_K=4096, loss_th=0.3, weight=None, ignore_index=250):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        batch_size = input.size()[0]

        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        thresh = loss_th

        def _bootstrap_xentropy_single(input, target, K, thresh, weight=None, size_average=True):

            n, c, h, w = input.size()
            input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            target = target.view(-1)

            loss = F.cross_entropy(
                input, target, weight=weight, reduce=self.reduce, size_average=self.size_average, ignore_index=ignore_index
            )
            sorted_loss, _ = torch.sort(loss, descending=True)

            if sorted_loss[K] > thresh:
                loss = sorted_loss[sorted_loss > thresh]
            else:
                loss = sorted_loss[:K]
            reduced_topk_loss = torch.mean(loss)

            return reduced_topk_loss

        loss = 0.0
        # Bootstrap from each image not entire batch
        for i in range(batch_size):
            loss += _bootstrap_xentropy_single(
                input=torch.unsqueeze(input[i], 0),
                target=torch.unsqueeze(target[i], 0),
                K=min_K,
                thresh=thresh,
                weight=weight,
                size_average=self.size_average,
            )
        return loss / float(batch_size)
    
def get_criterion(cfg):
    if cfg.losses.name == 'l1':
        return nn.L1Loss()
    elif cfg.losses.name == 'CE':
        return nn.CrossEntropyLoss()
    if cfg.losses.name == 'BCEandTversky':
        return BCEandTversky()
    elif cfg.losses.name == 'FocalandTversky':
        return FocalandTversky()
    elif cfg.losses.name == 'bootstrapped_cross_entropy2d': 
        return bootstrapped_cross_entropy2d()
    elif cfg.losses.name == 'DICE':
        return smp.losses.DiceLoss(mode='multiclass', ignore_index=cfg.train_config.ignore_index)
    else: 
        raise NameError('Choose proper model name!!!')

if __name__ == "__main__":
    true = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    pred = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    
    #loss = criterion(pred, true)
    #print(loss)