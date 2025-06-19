import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    d_state: int = 16
    expand: int = 2
    dt_rank: int | str = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

    def forward(self, x):  # x: (B, T, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x  # (B, T, d_model)

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        b, l, _ = x.shape
        x_and_res = self.in_proj(x)
        x_frag, res = x_and_res.split([self.args.d_inner, self.args.d_inner], dim=-1)
        x_t = rearrange(x_frag, 'b l d -> b d l')
        x_t = self.conv1d(x_t)[..., :l]
        x_t = rearrange(x_t, 'b d l -> b l d')
        x_t = F.silu(x_t)
        y = self.ssm(x_t)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        d_in, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        b, l, d_in = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, u, 'b l d, b l n, b l d -> b l d n')
        x = torch.zeros((b, d_in, n), device=u.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d n, b n -> b d')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + u * D

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
