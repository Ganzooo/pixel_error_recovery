import torch
import torch.nn as nn
import torchvision
#resnet = torchvision.models.resnet.resnet50(pretrained=True)

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.residual_block(x) #F(x)
        out = out + x #F(x) + x
        out = self.relu(out)
        return out
    
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) 
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, in_channel = 3, n_classes=3, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=pretrained)
        if in_channel == 1:
            resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + in_channel, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

class UNetWithResnet50Hybrid(nn.Module):
    DEPTH = 6

    def __init__(self, in_channel = 3, n_classes=3, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=pretrained)
        if in_channel == 1:
            resnet.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        
        ### Decoder branch for pixel detection
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + in_channel, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        
        ### Decoder branch for pixel recover
        up_blocks_recovery = []
        up_blocks_recovery.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks_recovery.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks_recovery.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks_recovery.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks_recovery.append(UpBlockForUNetWithResNet50(in_channels=64 + in_channel, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
        self.up_blocks_recovery = nn.ModuleList(up_blocks_recovery)
        
        #self.out_recovery = nn.Conv2d(64, in_channel, kernel_size=1, stride=1)
        self.trans = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        
        self.BasicResBlock = nn.Sequential(
            BasicResBlock(in_channels=32, out_channels=32),
            BasicResBlock(in_channels=32, out_channels=32),
            BasicResBlock(in_channels=32, out_channels=32),
            BasicResBlock(in_channels=32, out_channels=32),
        )
        self.out_recovery = nn.Conv2d(32, in_channel, kernel_size=1, stride=1)
        
    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        org_x = x 
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Hybrid.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x_bridge = self.bridge(x)
        
        x_det = x_bridge
        x_rec = x_bridge

        ### Detection
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Hybrid.DEPTH - 1 - i}"
            x_det = block(x_det, pre_pools[key])
        x_det = self.out(x_det)
        
        ### Recovery
        for i, block in enumerate(self.up_blocks_recovery, 1):
            key = f"layer_{UNetWithResnet50Hybrid.DEPTH - 1 - i}"
            x_rec = block(x_rec, pre_pools[key])
    
        x_rec = self.trans(x_rec)
        x_rec = self.BasicResBlock(x_rec)
        x_rec = self.out_recovery(x_rec)
        
        #Residual add
        x_rec = x_rec + org_x
        del pre_pools
        
        return x_det, x_rec
        
if __name__ == "__main__":
    model = UNetWithResnet50Hybrid(in_channel = 3, n_classes=2, pretrained=True).cuda()
    inp = torch.rand((10, 3, 512, 512)).cuda()
    out, out_rec = model(inp)
    print(out, out_rec)