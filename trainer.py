import os
import torch
import torchsummary
from UNetResNet16 import UNet16
from dataloader import *
from torch import optim
from tqdm import tqdm
from UNet18 import UNet18, myresnet
from UNetResNet50 import UNetResNet50
from torch import nn

TRAIN_IMG = "./crack_segmentation_dataset/train/images"
TRAIN_MASK = "./crack_segmentation_dataset/train/masks"
TEST_IMG = "./crack_segmentation_dataset/test/images"
TEST_MASK = "./crack_segmentation_dataset/test/masks"
MODEL_NAME = "UNetResNet50"
#MODEL_NAME = "UNet16"
LEARNING_RATE = 0.001
BATCH_SIZE = 4
GPU = "cuda:1"
#LOSS = "dice_loss"
LOSS = "weighted_bce_loss"
#LOSS = "bce_dice_loss"

# class IoULoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, target):
#         print(input.shape)
#         N, H, W = input.shape
#         #p = input.exp()
#         intersection = (p*target).sum(1).sum(1)
#         union = (p+target).sum(1).sum(1)-intersection
#         iou = intersection/(union+1)
#         assert iou.shape == torch.Size([N]), 'iouloss shape failure'
#         loss = 1-iou
#         return loss.mean()

def iou_loss(outputs, targets, smooth=1):
    intersection = (outputs * targets).sum()
    union = (outputs + targets).sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    iou_loss = 1 - iou
    return iou_loss
    
def dice_loss(output, target, smooth=1):
    # Flatten the tensors
    output = output.view(-1)
    target = target.view(-1)

    # Compute the intersection and union
    intersection = (output * target).sum()
    union = output.sum() + target.sum()

    # Compute the Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Compute the Dice loss
    loss = 1.0 - dice
    return loss

# This loss works only for binary segmentation
class WeightedBCELoss(nn.Module):
    def __init__(self, smooth=10 ** -8, reduction='mean', pos_weight=[1, 1]):
        super(WeightedBCELoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], f"predict & target batch size don't match. predict.shape={predict.shape}"
        predict = predict.clamp(min=self.smooth)
        # target = target.contiguous().view(target.shape[0], -1)
        # predict = predict.squeeze(1)
        # target = target.squeeze(1)
        # print(predict.size())
        # print(target.size())
        # print(self.pos_weight[1])

        loss =  - (self.pos_weight[0]*target*torch.log(predict+self.smooth) + self.pos_weight[1]*(1-target)*torch.log(1-predict+self.smooth))/sum(self.pos_weight)

        if self.reduction == 'mean':
            #print(loss.mean(dim=(1,2,3)).size())
            #print(loss)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class BCE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='none', pos_weight=[1, 1], loss_weight = [1, 1]):
        super(BCE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.bce_loss = WeightedBCELoss(pos_weight=pos_weight).to('cuda')
        self.dice_loss =dice_loss
        self.loss_weight = loss_weight

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        loss = (self.loss_weight[0] * self.bce_loss(predict, target) + self.loss_weight[1] * self.dice_loss(predict, target)) / sum(self.loss_weight)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

import torch



def evaluate(model, testloader, loss):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()
    loss = test_loss/len(testloader)
    return loss

def train(model, trainloader, lr, loss_ftn, epoch, testloader = None, threshold=10, train_history = [], test_history = [], working_dir = "./", scheduler = False):
    model_dir = os.path.join(working_dir, "models")
    optimizer = optim.SGD(model.parameters(), lr = lr)
    print("Training %d images"%len(trainloader))
    for e in tqdm(range(1, epoch+1)):
        print()
        torch.cuda.empty_cache()       # GPU 메모리 정리
        model.train()
        running_loss = 0.0
        for idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_ftn(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % (len(trainloader)//5)== 0:
            #     print("\t epoch : %d, batch : %d, training_loss : %.4f"%(e, idx, running_loss/(idx+1)))
        train_history.append(running_loss/len(trainloader))              # 배치별 손실값의 평균을 저장
        if testloader is not None:
            test_loss = evaluate(model, testloader, loss_ftn)
            test_history.append(test_loss)
            if test_loss <= threshold:# save model if loss if less than threshold
                threshold = test_loss
                if not os.path.exists(model_dir): os.makedirs(model_dir)
                torch.save({'epoch': e,
                            'train_history': train_history,
                            "test_history" : test_history,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }, model_dir+'/%s_%d_%d.pt'%(MODEL_NAME, e, int(test_loss*1000)))
        print("\n[%d] training loss: %f"%(e, running_loss / len(trainloader)))
        if testloader is not None: print("[%d] test loss: %f"%(e, test_loss))

        # pytorch learning rate scheduling
        if scheduler and e % 10 == 0:
            lr /= 10
            optimizer = optim.SGD(model.parameters(), lr = lr)

    return train_history, test_history

if torch.cuda.is_available(): device = torch.device(GPU)
#elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")

if __name__ == "__main__":
    #criterion = torch.nn.BCEWithLogitsLoss()
    if MODEL_NAME == "UNetResNet50":
        unet = UNetResNet50(pretrained = True, n_channels=1, n_classes=1).to(device)
    elif MODEL_NAME == "UNet18":
        unet = UNet18(pretrained=True).to(device)
    else: 
        criterion = torch.nn.BCELoss()
        unet = UNet16(num_classes=1, pretrained = True).to(device)
    
    if LOSS == "dice_loss": criterion = dice_loss
    elif LOSS == "iou_loss": criterion = iou_loss
    elif LOSS == "weighted_bce_loss": criterion = WeightedBCELoss().to(device)
    elif LOSS == "bce_dice_loss": criterion = BCE_DiceLoss(pos_weight = torch.tensor([20.0, 1.0]).to(device), loss_weight = [1.0, 1.0]).to(device)
    else: criterion = torch.nn.BCELoss()

    train_transform = TrainImageTransforms()
    test_transform = TestImageTransforms()
    mask_transforms = MaskTransforms()
    trainset = CrackDataSet(image_dir=TRAIN_IMG, mask_dir=TRAIN_IMG, image_transforms=train_transform, mask_transforms=mask_transforms)
    testset = CrackDataSet(image_dir=TEST_IMG, mask_dir=TEST_MASK, image_transforms=test_transform, mask_transforms=mask_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False)
    train_history, test_history = train(unet, trainloader, LEARNING_RATE, criterion, 30, testloader, scheduler=False)