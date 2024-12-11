# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:46:42 2023

@author: sourg
"""

from torch.utils.data import Dataset
import torch.nn as nn
import torch
import glob
import nibabel as nib
import cv2 
from torch import from_numpy
import numpy as np

class CamusEDImageDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transform=None,test=False):
        self.transform = transform
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
            
    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        
        # and read the associated mask from disk in grayscale mode
        image = nib.load(imagePath).get_fdata().astype(np.float32)
        
        
        mask = nib.load(self.maskPaths[idx]).get_fdata()

        if self.transform:
            image = self.transform(image)
        #resize the mask
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        #convert to tensor
        mask = from_numpy(mask).long() #convert now for skorch
        return image,mask