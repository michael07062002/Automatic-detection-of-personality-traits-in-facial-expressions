import torch
NUM_FRAMES = 16
SEG_LEN    = 10
STRIDE     = 5
IMG_SIZE   = 112
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"