import os
import numpy as np
import cv2
import torch
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

def flatten(image):
    return image[:,:,0]

class CrackDataSet(Dataset):
    def __init__(self, image_dir =  "./crack_segmentation_dataset/test/images", mask_dir = "./crack_segmentation_dataset/test/masks", image_transforms=None, mask_transforms=None):      
        print("image_dir", image_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fnames = [path.name for path in Path(image_dir).glob('*.jpg')]
        self.img_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __getitem__(self, i):
        fname = self.fnames[i]
        fpath = os.path.join(self.image_dir, fname)
        img = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
        spath = os.path.join(self.mask_dir, fname)
        mask = cv2.cvtColor(cv2.imread(spath), cv2.COLOR_BGR2RGB)

        if self.img_transforms is not None:
            img = self.img_transforms(img)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        return img, mask
    def __len__(self):
        return len(self.fnames)


# class CrackDataSetTest(Dataset):
#     def __init__(self, image_dir =  "./crack_segmentation_dataset/train/images", mask_dir = "./crack_segmentation_dataset/train/masks", image_transforms=None, mask_transforms=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.fnames = [path.name for path in Path(image_dir).glob('*.jpg')]
#         self.img_transforms = image_transforms
#         self.mask_transforms = mask_transforms

#     def __getitem__(self, i):
#         fname = self.fnames[i]
#         fpath = os.path.join(self.image_dir, fname)
#         img = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
#         spath = os.path.join(self.mask_dir, fname)
#         mask = cv2.cvtColor(cv2.imread(spath), cv2.COLOR_BGR2RGB)

#         if image_transforms is not None:
#             img = self.img_transforms(img)
#         if mask_transforms is not None:
#             mask = self.mask_transforms(mask)
#         return img, mask, fname

#     def __len__(self):
#         return len(self.fnames)

class TrainImageTransforms:
    def __init__(self):
        self.augment = tr.Compose([
            # tr.ConvertFromInts(),
            # tr.RandomMirror(),
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # tr.RandomGrayscale(0.25),
            # tr.RandomVerticalFlip(0.25),
        ])
    
    def __call__(self, image):
        return self.augment(image)

class TestImageTransforms:
    def __init__(self):
        self.augment = tr.Compose([
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __call__(self, image):
        return self.augment(image)

class MaskTransforms:
    def __init__(self):
        self.augment = tr.Compose([
            flatten,
            tr.ToTensor()
        ])
    
    def __call__(self, image):
        image = self.augment(image)
        return image/255.
