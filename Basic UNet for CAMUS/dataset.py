# import the necessary packages
from torch.utils.data import Dataset
import nibabel as nib
import cv2 
import numpy as np
from torch import from_numpy

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        
        # and read the associated mask from disk in grayscale mode
        image = nib.load(imagePath).get_fdata().astype(np.float32)
        
        
        mask = nib.load(self.maskPaths[idx]).get_fdata()
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to image
            image = self.transforms(image)
        
        #resize the mask
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        #convert to tensor
        mask = from_numpy(mask)
        
        # return a tuple of the image and its mask
        return (image, mask)
