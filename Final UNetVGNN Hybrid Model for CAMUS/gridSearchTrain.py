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
from sklearn.model_selection import train_test_split, GridSearchCV
from CamusEDImageDataset import CamusEDImageDataset
from model import greedyvig_b_feat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from FocalLoss import FocalLoss
from sklearn.metrics import make_scorer
from tqdm import tqdm
import glob
import config
import os

class FocalLossWithParams(FocalLoss):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(alpha=alpha, gamma=gamma)

def dice_score(net, X, y):
    """Higher is better"""
    y_pred = net.predict(X)
    return dice(torch.tensor(y_pred), torch.tensor(y, dtype=torch.long), average="micro", ignore_index=0)

def setup_training(greedy, train_data, device, save_dir='./weights'):
    os.makedirs(save_dir, exist_ok=True)
    
    net = NeuralNet(
        module=greedy,
        max_epochs=30,
        lr=0.01,
        train_split=False,
        criterion=FocalLossWithParams,  # Changed to parameterized version
        optimizer=Adam,
        device=device
    )
    return net

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #removing the last index for even 3 fold splits
    imagePaths = sorted(glob.glob(os.path.join(config.IMAGE_DATASET_PATH, "*.nii")))
    maskPaths = sorted(glob.glob(os.path.join(config.MASK_DATASET_PATH, "*.nii")))
    dice_scorer = make_scorer(dice_score, greater_is_better=True)

    train_data = CamusEDImageDataset(
        imagePaths=imagePaths,
        maskPaths=maskPaths,
        transform=Compose([ToPILImage(), Resize((256,256)), RandomRotation(10), ToTensor()]),
    )
    
    greedy = greedyvig_b_feat()
    total_params = sum(p.numel() for p in greedy.parameters())
    print(f"Total parameters: {total_params:,}")

    # Setup base neural net for grid search
    net = setup_training(greedy, train_data, device)
    
    # Define parameter grid
    param_grid = {
        'criterion__alpha': [0.25, 0.5, 0.75, 1.0],
        'criterion__gamma': [1.0, 2.0, 3.0]
    }
    
    # Initialize and perform grid search
    gs = GridSearchCV(
        estimator=net,
        param_grid=param_grid,
        cv=3,
        scoring=dice_scorer,
        n_jobs=1,  # Set to higher number if multiple GPUs available
        verbose=10
    )
    # Fit grid search
    gs.fit(train_data)
    
    # Print results
    print("Best parameters:", gs.best_params_)
    print("Best score:", gs.best_score_)
    
    # Save best model
    torch.save(gs.best_estimator_.module_.state_dict(), './weights/best_model.pt')