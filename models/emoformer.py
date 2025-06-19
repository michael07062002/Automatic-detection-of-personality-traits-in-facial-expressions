import torch.nn as nn
from .resnet_block   import ResNetBlock1
from .temporal_mamba import TemporalMamba
from .mt_chafs       import MT_CHAFS
from ..constants     import SEG_LEN


class EmoFormer(nn.Module):
    def __init__(self, feat_dim=768, d_model=256, n_layer=2):
        super().__init__()
        self.cnn   = ResNetBlock1(pretrained=True)  
        self.templ = TemporalMamba(feat_dim, d_model, n_layer)
        self.tempr = TemporalMamba(feat_dim, d_model, n_layer)
        self.fuse  = MT_CHAFS(self.templ.out_dim)
        self.head = nn.Linear(self.templ.out_dim * 2 + 2, 5)
        
    def forward(self, segL, segR):                             
        B,S = segL.shape[:2]
        segL, segR = segL.flatten(0,1), segR.flatten(0,1)
        featL = self.cnn(segL.flatten(0,1)).view(B*S, SEG_LEN, -1)
        featR = self.cnn(segR.flatten(0,1)).view(B*S, SEG_LEN, -1)
        tL = self.templ(featL).view(B,S,-1)
        tR = self.tempr(featR).view(B,S,-1)
        fused = self.fuse(tL, tR)
        return self.head(fused)