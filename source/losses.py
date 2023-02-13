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

class bootstrapped_cross_entropy2d_l1_hybrid(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=True, reduce=False, reduction='mean', ignore_index=250):
        super(bootstrapped_cross_entropy2d_l1_hybrid, self).__init__(size_average, reduce, reduction)
        self.detection = bootstrapped_cross_entropy2d()
        self.recovery = torch.nn.L1Loss()
        self.w1 = 0.5
        self.w2 = 0.5
        
    def forward(self, input, target, input_rec, target_rec):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        batch_size = input.size()[0]

        loss1 = self.detection(input, target)
        loss2 = self.recovery(input_rec, target_rec)
        loss = self.w1 * loss1 + self.w2 * loss2
        return loss
    
class bootstrapped_cross_entropy2d_l2_hybrid(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=True, reduce=False, reduction='mean', ignore_index=250):
        super(bootstrapped_cross_entropy2d_l2_hybrid, self).__init__(size_average, reduce, reduction)
        self.detection = bootstrapped_cross_entropy2d()
        self.recovery = torch.nn.MSELoss()
        self.w1 = 0.5
        self.w2 = 0.5
        
    def forward(self, input, target, input_rec, target_rec):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        batch_size = input.size()[0]

        loss1 = self.detection(input, target)
        loss2 = self.recovery(input_rec, target_rec)
        loss = self.w1 * loss1 + self.w2 * loss2
        return loss

class MaskedL1Loss(torch.nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
    def forward(self, input, target, mask):
        # diff = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        # result = torch.sum(diff) / torch.sum(mask)
        return torch.sum((torch.abs(input-target)*mask))  / torch.sum(mask)
        
class bootstrapped_cross_entropy2d_ml1_hybrid(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=True, reduce=False, reduction='mean', ignore_index=250):
        super(bootstrapped_cross_entropy2d_ml1_hybrid, self).__init__(size_average, reduce, reduction)
        self.detection = bootstrapped_cross_entropy2d()
        self.recovery = MaskedL1Loss()
        self.w1 = 0.3
        self.w2 = 0.7
        
    def forward(self, input, target, input_rec, target_rec, mask):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        batch_size = input.size()[0]

        loss1 = self.detection(input, target)
        loss2 = self.recovery(input_rec, target_rec, mask)
        loss = self.w1 * loss1 + self.w2 * loss2
        return loss
    
class MaskedL2Loss(torch.nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()
    def forward(self, input, target, mask):
        # diff = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        # result = torch.sum(diff) / torch.sum(mask)
        return torch.sum(((input-target)*mask)**2.0)  / torch.sum(mask)
        
class bootstrapped_cross_entropy2d_ml2_hybrid(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=True, reduce=False, reduction='mean', ignore_index=250):
        super(bootstrapped_cross_entropy2d_ml2_hybrid, self).__init__(size_average, reduce, reduction)
        self.detection = bootstrapped_cross_entropy2d()
        self.recovery = MaskedL2Loss()
        self.w1 = 0.5
        self.w2 = 0.5
        
    def forward(self, input, target, input_rec, target_rec, mask):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        batch_size = input.size()[0]

        loss1 = self.detection(input, target)
        loss2 = self.recovery(input_rec, target_rec, mask)
        loss = self.w1 * loss1 + self.w2 * loss2
        return loss
    
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
    elif cfg.losses.name == 'bootstrapped_cross_entropy2d_hybrid_l1':
        return bootstrapped_cross_entropy2d_l1_hybrid()
    elif cfg.losses.name == 'bootstrapped_cross_entropy2d_hybrid_l2':
        return bootstrapped_cross_entropy2d_l2_hybrid()
    elif cfg.losses.name == 'bootstrapped_cross_entropy2d_hybrid_ml1':
        return bootstrapped_cross_entropy2d_ml1_hybrid()
    elif cfg.losses.name == 'bootstrapped_cross_entropy2d_hybrid_ml2':
        return bootstrapped_cross_entropy2d_ml2_hybrid()
    else: 
        raise NameError('Choose proper model name!!!')

if __name__ == "__main__":
    # true = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    # pred = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    # loss = get_criterion(pred, true)
    # print(loss)
    loss = MaskedL1Loss()
    
    predict = torch.tensor([1.0, 2, 3, 4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1.0, 1, 1, 1], dtype=torch.float64,  requires_grad=True)
    mask = torch.tensor([0, 0, 0, 1], dtype=torch.float64, requires_grad=True)
    out = loss(predict, target, mask)
    out.backward()
    print(out)