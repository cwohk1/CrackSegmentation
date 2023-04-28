import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, channels = 3, num_class = 1, init_f = 16):
        super().__init__()
        
        # encoder
        self.conv1 = nn.Conv2d(channels, init_f, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(8*init_f, 16*init_f, 3, stride=1, padding=1)
        
        # decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up1 = nn.Conv2d(16*init_f, 8*init_f, 3, stride=1, padding=1)
        self.conv_up2 = nn.Conv2d(8*init_f, 4*init_f, 3, stride=1, padding=1)
        self.conv_up3 = nn.Conv2d(4*init_f, 2*init_f, 3, stride=1, padding=1)
        self.conv_up4 = nn.Conv2d(2*init_f, 1*init_f, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(init_f, num_class, 3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        
    
    def forward(self, x):
      # encoder
      x = F.relu(self.conv1(x))
      x = self.pool(x)

      x = F.relu(self.conv2(x))
      x = self.pool(x)

      x = F.relu(self.conv3(x))
      x = self.pool(x)

      x = F.relu(self.conv4(x))
      x = self.pool(x)

      x = F.relu(self.conv5(x))

      # decoder
      x = self.upsample(x)
      x = F.relu(self.conv_up1(x))

      x = self.upsample(x)
      x = F.relu(self.conv_up2(x)) 

      x = self.upsample(x)
      x = F.relu(self.conv_up3(x))

      x = self.upsample(x)
      x = F.relu(self.conv_up4(x))

      x = self.conv_out(x)
      return x

def dice_loss(pred, target, smooth = 1e-5):
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss.sum(), dice.sum()
    

def loss_func(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='sum')
    div, _ =dice_loss(pred, target)
    loss = bce + div
    return loss