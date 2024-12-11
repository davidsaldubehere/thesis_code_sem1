from dataset import SegmentationDataset
from Unet import Unet, print_model_parameters
import config
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import glob

if __name__ == "__main__":

    # load the image and mask filepaths in a sorted manner
    
    imagePaths = sorted(glob.glob(os.path.join(config.IMAGE_DATASET_PATH, "*.nii")))
    maskPaths = sorted(glob.glob(os.path.join(config.MASK_DATASET_PATH, "*.nii")))

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths,
        test_size=config.TEST_SPLIT, random_state=42)
    # unpack the data split
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]
    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model
    print("[INFO] saving testing image paths...")
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()

    # define transformations
    dataTransforms = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])
    
    # create the train and test datasets
    
    #TODO: Try cross entropy loss if 1 channel is bad or try to add 4 channels to the mask dataset?
    
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=dataTransforms)
    
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=dataTransforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())

    # initialize our UNet model
    unet = Unet(1, 4, light=False).to(config.DEVICE)    
    print_model_parameters(unet)

    # initialize loss function and optimizer
    
    #Try dice loss in future
    lossFunc = CrossEntropyLoss(reduction="mean")
    opt = Adam(unet.parameters(), lr=config.INIT_LR)
    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    testSteps = len(testDS) // config.BATCH_SIZE
    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        unet.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))    
            # perform a forward pass and calculate the training loss
            pred = unet(x) #The prediction apparently should be raw logits
            
            #Convert y to long tensor
            y = y.long()          

                                            
            loss = lossFunc(pred, y) 
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                pred = unet(x)
                y = y.squeeze(1)
                y = y.long()
                totalTestLoss += lossFunc(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)
    # serialize the model to disk
    torch.save(unet, config.MODEL_PATH)
