import torch, torch.nn as nn
from ..constants import *


def concordance_cc(pred, target):
    vx, vy = pred.var(0), target.var(0)
    mx, my = pred.mean(0), target.mean(0)
    sxy    = ((pred-mx)*(target-my)).mean(0)
    return (2*sxy) / (vx + vy + (mx-my).pow(2) + 1e-8)

class LossCCC_MSE(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()

    def forward(self, pred, y):
        mse = self.mse(pred, y)
        ccc = concordance_cc(pred, y).mean()
        return self.alpha*mse + (1-self.alpha)*(1-ccc)/2