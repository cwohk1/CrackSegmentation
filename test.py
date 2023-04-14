import torch, argparse, os
import torchsummary
from dataloader import *
import matplotlib.pyplot as plt
from UNet18 import UNet18, myresnet
from UNetResNet50 import UNetResNet50

TRAIN_IMG = "./crack_segmentation_dataset/train/images"
TRAIN_MASK = "./crack_segmentation_dataset/train/masks"
TEST_IMG = "./crack_segmentation_dataset/test/images"
TEST_MASK = "./crack_segmentation_dataset/test/masks"
MODEL_NAME = "UNetResNet50"
MODEL = "./models"
GPU = "cuda:1"
BATCH_SIZE = 4

class IoU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, mask):
        l = input.shape[0]
        iou_sum = 0
        for i in range(l):
            p = input[i]
            target = mask[i]
            N, H, W = p.shape
            intersection = (p*target).sum(1).sum(1)
            union = (p+target).sum(1).sum(1)-intersection
            iou = intersection/(union+1)
            assert iou.shape == torch.Size([N]), 'iouloss shape failure'
            iou_sum += iou.mean()
        return iou_sum / l


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

#def display_loss(train_loss_history, test_loss_history)
def display(*display_list):
    plt.figure(figsize = (15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.axis('off')
        plt.imshow(display_list[i])
    plt.show()

def show_test_pred(model, testloader, threshold=0.3):
    with torch.no_grad():
        data, labels = next(iter(testloader))
        data = data.to(device)
        masks = labels.permute(0, 2, 3, 1).to("cpu").numpy()
        preds = model(data).permute(0, 2, 3, 1).to("cpu").numpy()
        images = (data/2 + 0.5).permute(0, 2, 3, 1).to("cpu").numpy()
    for image, mask, pred in zip(images, masks, preds):
        display(image, mask, pred)
    return images, masks, preds

def load_model(model, weight):
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint["model"])
    return checkpoint["train_history"], checkpoint["test_history"]

if torch.cuda.is_available(): device = torch.device(GPU)
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type = str, help = "name of the model weight file")

    args = parser.parse_args()
    weight = os.path.join(MODEL, args.weight)

    criterion = IoU()

    if MODEL_NAME == "UNetResNet50":
        unet = UNetResNet50(pretrained = False, n_channels=1, n_classes=1).to(device)
    elif MODEL_NAME == "UNet18":
        unet = UNet18(pretrained=False).to(device)

    train_transform = TrainImageTransforms()
    test_transform = TestImageTransforms()
    mask_transforms = MaskTransforms()
    trainset = CrackDataSet(image_dir=TRAIN_IMG, mask_dir=TRAIN_IMG, image_transforms=train_transform, mask_transforms=mask_transforms)
    testset = CrackDataSet(image_dir=TEST_IMG, mask_dir=TEST_MASK, image_transforms=test_transform, mask_transforms=mask_transforms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
    testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False)
    train_history, test_history = load_model(unet, weight)

    with torch.no_grad():
        test_loss = 0
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = unet(data)
            test_loss += criterion.forward(data, target)

    print("test loss = ", test_loss.item()/len(testloader))
    images, masks, preds = show_test_pred(unet, testloader, 0.5)