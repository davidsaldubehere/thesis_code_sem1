"""
Script to test the U-Net model on the test set given by the challenge
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
torch.manual_seed(1907)
from CamusEDImageDataset import CamusEDImageDataset
from model import greedyvig_b_feat
import config
from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.functional import dice
from tqdm import tqdm
from datasets import load_metric
import glob
from torchmetrics.functional import dice
import os


imagePaths = sorted(glob.glob(os.path.join(config.IMAGE_DATASET_PATH_TEST, "*.nii")))
maskPaths = sorted(glob.glob(os.path.join(config.MASK_DATASET_PATH_TEST, "*.nii")))

test_data =CamusEDImageDataset(
    transform=Compose([ToPILImage(),Resize((256,256)),ToTensor()]),
    imagePaths = imagePaths,
    maskPaths = maskPaths
)

test_dataloader = DataLoader(test_data, batch_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idxImgTest = 4
inputs, labels = test_dataloader.dataset[idxImgTest]
inputs = inputs.unsqueeze(0)
labels = labels.unsqueeze(0)
net = greedyvig_b_feat().to(device)
with torch.no_grad():
    outputs = net(inputs.to(device))

    new_labels = torch.squeeze(labels)
    print(outputs.shape)