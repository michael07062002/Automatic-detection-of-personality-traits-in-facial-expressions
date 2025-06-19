import math, torch.nn as nn, torch


class MT_CHAFS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_L, self.fc_R = nn.Linear(dim,dim), nn.Linear(dim,dim)
        self.q_L, self.k_L, self.v_L = nn.Linear(dim,dim,False), nn.Linear(dim,dim,False), nn.Linear(dim,dim,False)
        self.q_R, self.k_R, self.v_R = nn.Linear(dim,dim,False), nn.Linear(dim,dim,False), nn.Linear(dim,dim,False)

    @staticmethod
    def _attn(Q,K,V):
        w = (Q*K).sum(-1,keepdim=True) / math.sqrt(Q.size(-1))   
        w = w.softmax(1)
        return (w*V).sum(1)                                     

    def forward(self, seqL, seqR):                              
        L, R = self.fc_L(seqL), self.fc_R(seqR)
        ctx_L2R = self._attn(self.q_L(L), self.k_R(R), self.v_R(R))
        ctx_R2L = self._attn(self.q_R(R), self.k_L(L), self.v_L(L))
        fuse = torch.cat([ctx_L2R, ctx_R2L], 1)                 
        μ = fuse.mean(dim=1, keepdim=False, dtype=fuse.dtype).unsqueeze(1) 
        σ = fuse.std(dim=1, keepdim=False).unsqueeze(1)                   
        return torch.cat([fuse, μ, σ], dim=1)                      
