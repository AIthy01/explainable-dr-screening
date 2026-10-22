import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5
MODEL_NAME = "resnet50"  
# 可选: vgg16, resnet50, densenet121, efficientnet_b3

TRAIN_CSV = "data/train.csv"
IMAGE_DIR = "data/train_images"

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
NUM_WORKERS = 8

SAVE_PATH = f"results/checkpoints/{MODEL_NAME}_best.pth"