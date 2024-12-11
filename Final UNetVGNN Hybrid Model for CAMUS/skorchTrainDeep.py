import torch
torch.manual_seed(1907)
import torch.nn as nn
from skorch import NeuralNet
from skorch.dataset import Dataset
from skorch.callbacks import EarlyStopping, ProgressBar, PrintLog, EpochScoring, Checkpoint
from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor,RandomRotation
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchmetrics.functional import dice
from skorch.dataset import ValidSplit
from sklearn.model_selection import train_test_split
from CamusEDImageDataset import CamusEDImageDataset
from model import greedyvig_b_feat

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from FocalLoss import FocalLoss


from tqdm import tqdm

import glob
import config
import os

class DeepCrossEntropy(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, y_pred, y_true):
        # Compute weighted loss for each output
        losses = []
        for pred, weight in zip(y_pred, self.weights):
            losses.append(weight * self.criterion(pred, y_true))
        return sum(losses)


def dice_score(net, X, y):
    """Higher is better"""

    #Shape will be (164, 4, 4, 256, 256) so we need to extract the outer segmentation map
    y_pred = net.predict(X)[:, 0, :, :]
    return dice(torch.tensor(y_pred), torch.tensor(y, dtype=torch.long), average="micro", ignore_index=0)

def setup_training(greedy, train_data, device, save_dir='./weights'):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup callbacks
    dice_scoring = EpochScoring(dice_score, name='dice', lower_is_better=False)
    checkpoint = Checkpoint(
        f_params=os.path.join(save_dir, 'Unet.pt'),  # Save best model parameters
        monitor='dice_best',  # Monitor dice score
        f_history=os.path.join(save_dir, 'history.json'),  # Save training history
        f_pickle=None,  # Don't pickle the entire model
    )

    net = NeuralNet(
        module=greedy,
        max_epochs=40,
        lr=0.01,
        callbacks=[
            dice_scoring,
            checkpoint,  # Add checkpoint callback
            PrintLog(keys_ignored=None),
            ProgressBar()
        ],
        train_split=ValidSplit(cv=5),
        iterator_train__shuffle=True,
        iterator_train__batch_size=4,
        criterion=DeepCrossEntropy([1, .7, .3, .1]),
        batch_size=4,
        optimizer=Adam,
        device=device
    )
    
    return net

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Your existing dataset setup code here
    imagePaths = sorted(glob.glob(os.path.join(config.IMAGE_DATASET_PATH, "*.nii")))
    maskPaths = sorted(glob.glob(os.path.join(config.MASK_DATASET_PATH, "*.nii")))
    
    train_data = CamusEDImageDataset(
        imagePaths=imagePaths,
        maskPaths=maskPaths,
        transform=Compose([ToPILImage(), Resize((256,256)), RandomRotation(10), ToTensor()]),
    )
    
    greedy = greedyvig_b_feat()
    total_params = sum(p.numel() for p in greedy.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup and train the model
    net = setup_training(greedy, train_data, device)
    net.fit(train_data)
    
    # The best model is automatically saved during training,
    # but you can still save the final model if desired
    torch.save(net.module_.state_dict(), './weights/final_model.pt')
