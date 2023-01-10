import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as summary_

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

class plainDP(nn.Module):
    def __init__(self, module_nums=4, channel_nums=32, num_class=2, act_type='relu', colors=3, use_bn=True):
        super(plainDP, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.use_bn = use_bn
        self.num_class = num_class

        backbone = []
        backbone += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn)]
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type, use_bn=self.use_bn)]
        backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type='linear', use_bn=self.use_bn)]
        self.backbone = nn.Sequential(*backbone)

        self.seghead = torch.nn.Conv2d(self.channel_nums, self.num_class, kernel_size=1)
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        y = self.backbone(x) #+ _x
        y = self.seghead(y)
        return y
    
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

if __name__ == "__main__":
    x = torch.rand(1,3,128,128).cuda()
    model_dp = plainDP(module_nums=4, channel_nums=32, num_class=2, act_type='relu', colors=3, use_bn=True).cuda()
    model_rp = plainRP(module_nums=4, channel_nums=32, act_type='relu', colors=3, use_bn=True).cuda()
    #y = model(x)
    #print(model.summary())
    summary_(model_dp,(3,128,128), batch_size=16)
    summary_(model_rp,(3,128,128), batch_size=16)