import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, last_layer=False):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)

        if last_layer:
            self.conv = DoubleConv(in_channels//4*3, out_channels)
        else:
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.size())
        # print(x2.size())

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetResNet50(nn.Module):
    def __init__(self, pretrained = True, n_channels = 1, n_classes = 1):
        super(UNetResNet50, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.resnet = resnet50(pretrained=True)
        self.dropout = nn.Dropout(p=0.5)
        self.inconv = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.resnet.maxpool
        )
        self.downconv1 = self.resnet.layer1
        self.downconv2 = self.resnet.layer2
        self.downconv3 = self.resnet.layer3
        self.downconv4 = self.resnet.layer4

        self.upconv4 = Up(2048, 1024)
        self.upconv3 = Up(1024, 512)
        self.upconv2 = Up(512, 256)
        self.upconv1 = Up(256, 128, True)

        self.outconv = nn.Sequential(
            nn.Conv2d(128, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
    def forward(self, x):
        x0 = self.inconv(x)
        x1 = self.downconv1(x0)
        x2 = self.downconv2(x1)
        x3 = self.downconv3(x2)
        x4 = self.downconv4(x3)

        x4 = self.dropout(x4)

        x = self.upconv4(x4, x3)
        x = self.upconv3(x, x2)
        x = self.upconv2(x, x1)
        x = self.upconv1(x, x0)

        return self.outconv(x)
