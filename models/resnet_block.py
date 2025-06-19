import torch.nn as nn, torchvision.models as models, torch


class ResNetBlock1(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet50(pretrained=pretrained)
        self.block1 = nn.Sequential(*list(res.children())[:5])  
        self.out_dim = 256

    def forward(self, x):                     
        f = self.block1(x)                    
        μ  = f.mean(dim=(2,3))
        σ  = f.std(dim=(2,3))
        mx = f.amax(dim=(2,3))
        return torch.cat([μ, σ, mx], dim=1)   