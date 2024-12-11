import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import nibabel as nib
import glob
from Unet import Unet

def prepare_plot(origImage, origMask, predMask, filename):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(5, 5))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.savefig(filename)
    plt.close(figure)

def make_predictions(model, imagePath):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        # load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = nib.load(imagePath).get_fdata()
        
        image = image.astype("float32") / 255.0
        # resize the image and make a copy of it for visualization
        image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
        orig = image.copy()
        # find the filename and generate the path to ground truth
        # mask
        filename = imagePath.split(os.path.sep)[-1]
        #add _gt to the filename before the .nii extension
        filename = filename.replace(".nii", "_gt.nii")
        
        groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
            filename)
        # load the ground-truth segmentation mask in grayscale mode
        # and resize it
        gtMask = nib.load(groundTruthPath).get_fdata()
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

                
        # create a channel axis, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        print(image.shape)
        predMask = model(image).squeeze(0)
        print(predMask.shape)
        #print the channel output of the model        
        
        
        predMask = torch.argmax(predMask, dim=0)
        predMask = predMask.cpu().numpy()
        print(predMask.shape)
        #The shape of the predMask is (1, 4, 256, 256), we need to remove the first two dimensions by taking the argmax of the second dimension
        # resize the predicted mask to the original dimensions        
        # prepare a plot for visualization
        save_filename = filename.replace("_gt.nii", "_pred.png")
        prepare_plot(orig, gtMask, predMask, save_filename)
        # load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
#imagePaths = sorted(glob.glob(os.path.join(config.IMAGE_DATASET_PATH, "*.nii")))
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
model = Unet(1,4,light=False)  # Initialize the model architecture
model.load_state_dict(torch.load(config.MODEL_PATH))  # Load the state dictionary into the model
model = model.to(config.DEVICE)  # Move the model to the specified device

# iterate over the randomly selected test image paths
for path in imagePaths:
    # make predictions and visualize the results
    make_predictions(model, path)
    
plt.show()
