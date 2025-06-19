import torch.nn as nn
from .mamba_ssm import Mamba, ModelArgs 


class TemporalMamba(nn.Module):
    def __init__(self, d_in, d_model=512, n_layer=4):
        super().__init__()
        self.proj   = nn.Linear(d_in, d_model)
        self.mamba  = Mamba(ModelArgs(d_model=d_model, n_layer=n_layer))
        self.out_dim= d_model*2

    def forward(self, x):                      
        x = self.proj(x); m = self.mamba(x)
        return torch.cat([m.mean(1), m.std(1)], 1)  
