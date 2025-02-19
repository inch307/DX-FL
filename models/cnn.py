import torch
import torch.nn as nn
import math
from fa import fa_conv, fa_linear

class CNN_2_32_fa(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_2_32_fa, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            fa_conv.FeedbackConvLayer(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            fa_conv.FeedbackConvLayer(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(p=0.05),
        )

        self.fc_layer = nn.Sequential(
            fa_linear.FeedbackLinearLayer(2048, 256),
            nn.ReLU(inplace=True),
            fa_linear.FeedbackLinearLayer(256, num_classes)
        )

    def forward(self, x):
        h = self.conv_layer(x)
        h = h.view(h.size(0), -1)
        y = self.fc_layer(h)
        return h, y
    
### Moderate size of CNN for CIFAR-10 dataset
class CNN_2_32(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_2_32, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(p=0.05),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv_layer(x)
        h = h.view(h.size(0), -1)
        y = self.fc_layer(h)
        return h, y
    