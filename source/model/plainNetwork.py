import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as summary_
#from source.utils.media_filter_torch import MedianPool2d

class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='relu', use_bn=False):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(self.out_planes)
        self.act  = None
        self.use_bn = use_bn

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.use_bn:
            y = self.bn(y)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

class Conv1X1(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='relu', use_bn=False):
        super(Conv1X1, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv1x1 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(self.out_planes)
        self.act  = None
        self.use_bn = use_bn

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv1x1(x)
        if self.use_bn:
            y = self.bn(y)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

class plainDP(nn.Module):
    def __init__(self, module_nums=4, channel_nums=32, num_class=2, act_type='relu', colors=3, use_bn=True, rec_mode = True):
        super(plainDP, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.use_bn = use_bn
        self.num_class = num_class
        self.rec_mode = rec_mode

        backbone = []
        backbone += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn)]
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn)]
        backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type='linear', use_bn=self.use_bn)]
        self.backbone = nn.Sequential(*backbone)

        self.seghead = torch.nn.Conv2d(self.channel_nums, self.num_class, kernel_size=1)
        
        if self.rec_mode:
            self.median_filter = MedianPool2d()
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        y = self.backbone(x) #+ _x
        y = self.seghead(y)
        
        if self.rec_mode:
            msk = y.data.max(1)[1].squeeze(axis=0)
            mask = torch.cat([msk.unsqueeze(axis=1),msk.unsqueeze(axis=1),msk.unsqueeze(axis=1)],dim=1)
            
            x_median = self.median_filter(x)
            x_median = torch.mul(x_median, mask)
            
            msk_inv = torch.ones_like(x_median)
            msk_inv[x_median>0] = 0
            
            y_rec = (torch.mul(x,msk_inv) + x_median)
            
            return y, y_rec
        return y, _
    
class plainRP(nn.Module):
    def __init__(self, module_nums=4, channel_nums=32, act_type='relu', colors=3, use_bn=True):
        super(plainRP, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.use_bn = use_bn
        self.mid_channel = 27

        backbone = []
        backbone += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn)]
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn)]
        backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.mid_channel, use_bn=self.use_bn)]
        self.backbone = nn.Sequential(*backbone)

        self.transision = nn.Sequential(
            Conv3X3(inp_planes=self.mid_channel, out_planes=self.mid_channel, use_bn=self.use_bn),
            Conv3X3(inp_planes=self.mid_channel, out_planes=self.mid_channel, act_type='linear', use_bn=self.use_bn)
        )
        self.seghead = torch.nn.Conv2d(self.mid_channel, colors, kernel_size=1)
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        y = self.backbone(x) #+ _x
        y = self.transision(y)
        y = self.seghead(y)
        return y

class plainHYBRID(nn.Module):
    def __init__(self, module_nums=4, channel_nums=32, num_class=2, act_type='relu', colors=3, use_bn_det=True, use_bn_rec=True, rec_mode = True):
        super(plainHYBRID, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.use_bn_det = use_bn_det
        self.use_bn_rec = use_bn_rec
        self.num_class = num_class
        self.rec_mode = rec_mode

        backbonePD = []
        backbonePD += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn_det)]
        for _ in range(self.module_nums):
            backbonePD += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn_det)]
        backbonePD += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums//2, act_type='linear', use_bn=self.use_bn_det)]
        self.backbonePD = nn.Sequential(*backbonePD)

        self.headPD = torch.nn.Conv2d(self.channel_nums//2, self.num_class, kernel_size=1)
        
        
        backbonePR = []
        backbonePR += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn_rec)]
        for _ in range(self.module_nums):
            backbonePR += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn_rec)]
        backbonePR += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type='linear', use_bn=self.use_bn_rec)]
        self.backbonePR = nn.Sequential(*backbonePR)

        self.trasitionPR = nn.Sequential(
                            Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums//2, act_type=self.act_type, use_bn=self.use_bn_rec),
                            Conv1X1(inp_planes=self.channel_nums//2, out_planes=self.colors, act_type=self.act_type, use_bn=self.use_bn_rec)
        )
            
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        y = self.backbonePD(x) #+ _x
        y = self.headPD(y)
        
        if self.rec_mode:
            #msk = y.data.max(1)[1].squeeze(axis=0)
            msk = y.data.max(1)[1]
            mask = torch.cat([msk.unsqueeze(axis=1),msk.unsqueeze(axis=1),msk.unsqueeze(axis=1)],dim=1)
            
            x_masked = torch.mul(x, mask)
            
            y1 = self.backbonePR(x_masked)
            y1 = self.trasitionPR(y1)
            y_rec = x + y1    
            return y, y_rec
        return y, _

if __name__ == "__main__":
    x = torch.rand(1,3,128,128).cuda()
    #model_dp = plainDP(module_nums=4, channel_nums=32, num_class=2, act_type='relu', colors=3, use_bn=True).cuda()
    #model_rp = plainRP(module_nums=4, channel_nums=32, act_type='relu', colors=3, use_bn=True).cuda()
    model_hybrid = plainHYBRID(module_nums=4, channel_nums=32, num_class=2, act_type='relu', colors=3, use_bn=True, rec_mode = True).cuda()

    #summary_(model_dp,(3,128,128), batch_size=16)
    #summary_(model_rp,(3,128,128), batch_size=16)
    summary_(model_hybrid,(3,128,128), batch_size=16)
    
    
    