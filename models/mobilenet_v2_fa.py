import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fa import fa_conv
from fa import fa_linear

def conv1x1_fa(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return fa_conv.FeedbackConvLayer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t = 6, downsample = False, pre_fa=False, post_fa=False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        
        # for main path:
        c  = t * input_channel
        # 1x1   point wise conv
        if pre_fa:
            self.conv1 = conv1x1_fa(input_channel, c)
        else:
            self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        if post_fa:
            self.conv3 = conv1x1_fa(c, output_channel)
        else:
            self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)
    
    def forward(self, inputs):
        # main path
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace = True)
        x = self.bn3(self.conv3(x))

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, alpha = 1, norm_layer=None, pre_fa=False, post_fa=False):
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        # TODO: norm_layer: group norm
        self.pre_fa = pre_fa
        self.post_fa = post_fa

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(16, 24, downsample = False, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(24, 24, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(24, 32, downsample = False, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(32, 32, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(32, 32, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(32, 64, downsample = True, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(64, 64, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(64, 64, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(64, 64, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(64, 96, downsample = False, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(96, 96, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(96, 96, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(96, 160, downsample = True, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(160, 160, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(160, 160, pre_fa=self.pre_fa, post_fa=self.post_fa),
            BaseBlock(160, 320, downsample = False, pre_fa=self.pre_fa, post_fa=self.post_fa))

        # last conv layers and fc layer
        if self.pre_fa or self.post_fa:
            self.conv1 = conv1x1_fa(int(320*alpha), 1280)
        else:
            self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        if self.pre_fa or self.post_fa:
            self.fc = fa_linear.FeedbackLinearLayer(1280, num_classes)
        else:
            self.fc = nn.Linear(1280, num_classes)
        
        # weights init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, fa_conv.FeedbackConvLayer) or isinstance(m, fa_linear.FeedbackLinearLayer):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                with torch.no_grad():
                    m.B.copy_(m.weight.data.clone())
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):

        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return None, None, x