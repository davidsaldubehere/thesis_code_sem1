import torch
torch.manual_seed(1907)
import torch.nn as nn

from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor,RandomRotation
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.model_selection import train_test_split
from CamusEDImageDataset import CamusEDImageDataset
from model import SegmentationVGNN

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


from tqdm import tqdm

from torchmetrics.functional import dice
import glob
import config
import os


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NB_EPOCHS = 40
    VALID_SIZE = 5

    #Use a "light" version if True (3.7M params) or the paper version if False (31M params)
    lightUnet = False
    
    imagePaths = sorted(glob.glob(os.path.join(config.IMAGE_DATASET_PATH, "*.nii")))
    maskPaths = sorted(glob.glob(os.path.join(config.MASK_DATASET_PATH, "*.nii")))
    NBSAMPLES = len(imagePaths)

    #Load camus dataset
    train_data =CamusEDImageDataset(
        imagePaths=imagePaths,
        maskPaths=maskPaths,
        transform=Compose([ToPILImage(),Resize((224,224)),RandomRotation(10),ToTensor()]),
    )

    valid_data =CamusEDImageDataset(
        imagePaths=imagePaths,
        maskPaths=maskPaths,
        transform=Compose([ToPILImage(),Resize((224,224)),ToTensor()]),
    )

    #Split with validation set
    train_indices, val_indices = train_test_split(np.arange(0,NBSAMPLES,1),test_size=VALID_SIZE,random_state=1907)

    train_data = torch.utils.data.Subset(train_data,train_indices)
    valid_data =torch.utils.data.Subset(valid_data,val_indices)
    
    #Turn the dataset into DataLoader
    train_dataloader = DataLoader(train_data, batch_size=5)
    valid_dataloader = DataLoader(valid_data, batch_size=5)
    
    
    net = SegmentationVGNN().to(device)

    optimizer = optim.Adam(net.parameters(),lr=1e-3)
    
    criterion = nn.CrossEntropyLoss()
    criterion.requires_grad = True
    
    lossEvolve = []
    valEvolve = []
    diceEvolve = []

    #For animation
    imgs = []
    for epoch in tqdm(range(NB_EPOCHS)):  # loop over the dataset multiple times
        net.train()
        print("################# EPOCH:",epoch+1,"#################")

        #Train
        train_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs,labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.type(torch.LongTensor).to(device))
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        #Validation
        net.eval()
        val_loss = 0.0
        dice_curr = 0.0
        with torch.no_grad():
            for j, data in enumerate(valid_dataloader, 0):
                inputs, labels = data
                inputs,labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                labels = labels.int()
                #You just need to replace the transformation or something for the masks
                loss = criterion(outputs, labels.type(torch.LongTensor).to(device))
                #loss.requires_grad = True
                val_loss += loss.item()
                dice_curr += dice(outputs,labels,average="micro",ignore_index=0)

            #For animation
            inputs, labels = valid_dataloader.dataset[0]
            if epoch == 0:
                baseImage = inputs[0]
                baseImage = baseImage.detach().cpu().numpy()

            inputs = inputs.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            outputs = net(inputs)
            outputs = torch.softmax(outputs,1)
            pred = torch.Tensor(torch.argmax(outputs,1).float())
            imgs.append(pred.detach().cpu().numpy()[0])

        lossEvolve.append(train_loss/(i+1))
        valEvolve.append(val_loss/(j+1))
        diceEvolve.append(dice_curr.cpu()/(j+1))
        print("Training Loss: %f \tValid Loss: %f \tDice: %f"%(train_loss/(i+1),val_loss/(j+1),dice_curr/(j+1)))

        if val_loss/(j+1) == min(valEvolve):
            torch.save(net.state_dict(),'./weights/Unet.pt')
    print('Finished Training')
    plt.figure(figsize=(5,5))
    plt.plot(lossEvolve,label="Train set loss")
    plt.plot(valEvolve,label="Validation set loss")
    plt.title("Evolution of loss for validation and train dataset")
    plt.legend()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.plot(diceEvolve)
    plt.title("Evolution of Dice metric on valdiation set")
    plt.show()


    palette = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]])
    def updateSeg(i,imgBase,segEvolve):
        plt.clf()
        plt.axis("off")
        plt.title(f"Evolution of segmentation with Unet: Epochs {i+1}")
        seg = segEvolve[i]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]
        plt.imshow(imgBase,cmap="gray")
        plt.imshow(color_seg,alpha=0.3)
    
    #Animation
    frames = [] # for storing the generated images
    fig = plt.figure()
    plt.axis("off")
    plt.title(f"Evolution of segmentation with Unet during {NB_EPOCHS} Epochs")
    ani = animation.FuncAnimation(fig, updateSeg,frames=len(imgs), interval=1000,repeat_delay=300,fargs=(baseImage,imgs))
    writergif = animation.PillowWriter(fps=30) 
    ani.save('SimplifiedUnet.gif', writer=writergif)
    plt.show()
