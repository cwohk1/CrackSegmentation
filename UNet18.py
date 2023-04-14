# Unet with resnet18 backbone
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models
from collections import OrderedDict

#    '''resnet model that outputs intermediate features'''
class myresnet(nn.Module):
    def __init__(self, resnet, hidden_layers):
        super(myresnet, self).__init__()
        self.backbone = resnet
        self.classifier = nn.Sequential(OrderedDict([   # 레이어의 이름을 지정하여 만들 수 있다.
            ('fc1', nn.Linear(512, hidden_layers)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layers, 2)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        self.lrs = (self.backbone.conv1, self.backbone.bn1, self.backbone.maxpool, self.backbone.layer1, self.backbone.layer2)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        l0 = self.backbone.maxpool(x)
        l1 = self.backbone.layer1(l0)
        l2 = self.backbone.layer2(l1)
        l3 = self.backbone.layer3(l2)
        l4 = self.backbone.layer4(l3)
        out = self.backbone.avgpool(l4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return {'layer0': l0, 'layer1': l1, 'layer2': l2,
                'layer3': l3, 'layer4': l4, 'class': out}


class UNet18(nn.Module):
    def __init__(self, hidden_layers = 1024, pretrained = True):
        super(UNet18, self).__init__()
        resnet = models.resnet18(pretrained = pretrained)
        resnet = myresnet(resnet, hidden_layers = hidden_layers)

        self.resnet = resnet
        self.ct2d = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c2d = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.ct2d2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c2d2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.ct2d3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c2d3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.ct2d4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c2d4 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.ct2d5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c2d5 = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        self.lrs = self.resnet.lrs

    def forward(self, x):
        g = self.resnet(x)
        l0 = g['layer0']
        l1 = g['layer1']
        l2 = g['layer2']
        l3 = g['layer3']
        l4 = g['layer4']
        out = g['class']
        up = self.ct2d(l4)  # increase size from IMG_SIZE/32 to IMG_SIZE/16
        up = torch.cat([l3, up], dim=1)  # size Nx512xIMG/SIZE/16**2
        up = self.c2d(up)  # size Nx256xIMG_SIZE/16**2
        up = self.bn1(up)
        up = self.relu(up)
        up = self.ct2d2(up)  # size Nx128xIMG_SIZE/8**2
        up = torch.cat([l2, up], dim=1)  # size Nx256xIMG_SIZE/8**2
        up = self.c2d2(up)  # size Nx128xIMG_SIZE/8**2
        up = self.bn2(up)
        up = self.relu(up)
        up = self.ct2d3(up)  # size Nx64xIMG_SIZE/4**2
        up = torch.cat([l1, up], dim=1)  # size Nx128xIMG_SIZE/4**2
        up = self.c2d3(up)  # size Nx64xIMG_SIZE/4**2
        up = self.bn3(up)
        up = self.relu(up)
        up = torch.cat([l0, up], dim=1)  # size Nx128xIMG_SIZE/4**2
        up = self.ct2d4(up)  # size Nx64xIMG_SIZE/2**2
        up = self.c2d4(up)  # size Nx32xIMG_SIZE/2**2
        up = self.bn4(up)
        up = self.relu(up)
        up = self.ct2d5(up)  # size Nx16xIMG_SIZE**2
        up = self.c2d5(up)
        up = torch.squeeze(up)   # 채널 수를 없애준다. 배치 크기가 1이라면 오류가 발생하므로 주의
        up = nn.LogSigmoid()(up)
        return up
