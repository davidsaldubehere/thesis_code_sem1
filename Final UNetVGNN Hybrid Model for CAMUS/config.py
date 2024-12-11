import torch
import os
# base path of the dataset
DATASET_PATH = os.path.join("dataset", "train")
DATASET_PATH_TEST = os.path.join("dataset", "test")
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
IMAGE_DATASET_PATH_TEST = os.path.join(DATASET_PATH_TEST, "images")
MASK_DATASET_PATH_TEST = os.path.join(DATASET_PATH_TEST, "masks")

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_camus.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
