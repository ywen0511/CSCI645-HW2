"""Flow-matching action-chunk policy with a DiT-style denoiser.

This is the modern counterpart to the 2023 methods the assignment names, and is
included as an ADDITIONAL baseline (the required baseline/ablation arms all use
ACT, as Part 3 asks for one algorithm held fixed).

Method: conditional/rectified flow matching (Lipman et al., ICLR 2023,
arXiv:2210.02747; Liu et al., arXiv:2209.03003). Rather than learning a
multi-step denoiser as in Diffusion Policy (Chi et al., RSS 2023,
arXiv:2303.04137), we regress the straight-line velocity field between noise and
data, which permits very few integration steps at inference. This is the action
representation used by recent VLA action experts such as pi-0
(Black et al., arXiv:2410.24164).

Training:  x0 ~ N(0, I), x1 = action chunk, t ~ U(0,1),
           x_t = (1-t) x0 + t x1,  target velocity = x1 - x0,  loss = MSE.
Sampling:  x <- x0; x <- x + (1/K) v(x, t, obs) for K uniform Euler steps.

Backbone: a transformer over the H action tokens whose blocks are modulated by
adaLN-Zero (Peebles & Xie, DiT, arXiv:2212.09748) from the conditioning vector
(timestep embedding + observation embedding).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowConfig:
    obs_dim: int
    act_dim: int
    chunk_size: int = 30
    hidden_dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    n_sample_steps: int = 10

    def __post_init__(self):
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads "
                f"({self.n_heads}); nn.MultiheadAttention asserts this.")

    def as_dict(self) -> dict:
        return asdict(self)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    """Sinusoidal embedding of a continuous flow time in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().reshape(-1, 1) * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        # 6 modulation signals: shift/scale/gate for attention and for the MLP
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1.unsqueeze(1)) + sh1.unsqueeze(1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * a
        h = self.norm2(x) * (1 + sc2.unsqueeze(1)) + sh2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class FlowPolicy(nn.Module):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        D, H = cfg.hidden_dim, cfg.chunk_size
        self.act_in = nn.Linear(cfg.act_dim, D)
        self.pos = nn.Parameter(torch.zeros(1, H, D))
        self.obs_emb = nn.Sequential(nn.Linear(cfg.obs_dim, D), nn.SiLU(), nn.Linear(D, D))
        self.t_emb = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.blocks = nn.ModuleList(
            [DiTBlock(D, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        self.head = nn.Linear(D, cfg.act_dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.zeros_(self.ada_out[1].weight); nn.init.zeros_(self.ada_out[1].bias)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def velocity(self, x: torch.Tensor, t: torch.Tensor, obs: torch.Tensor):
        """v(x_t, t, obs) with x: (B,H,act_dim), t: (B,), obs: (B,obs_dim)."""
        c = self.t_emb(timestep_embedding(t, self.cfg.hidden_dim)) + self.obs_emb(obs)
        h = self.act_in(x) + self.pos
        for blk in self.blocks:
            h = blk(h, c)
        sh, sc = self.ada_out(c).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        return self.head(h)

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, pad: torch.Tensor):
        B = obs.shape[0]
        x0 = torch.randn_like(actions)
        t = torch.rand(B, device=actions.device, dtype=actions.dtype)
        tb = t.view(B, 1, 1)
        xt = (1 - tb) * x0 + tb * actions
        target = actions - x0
        v = self.velocity(xt, t, obs)
        valid = (1.0 - pad).unsqueeze(-1)
        mse = ((v - target) ** 2 * valid).sum() / valid.sum().clamp(min=1.0) / actions.shape[-1]
        return mse, {"loss": float(mse.detach()), "mse": float(mse.detach())}

    @torch.no_grad()
    def predict_chunk(self, obs: torch.Tensor, n_steps: Optional[int] = None):
        """Euler-integrate the learned velocity field from noise to an action chunk."""
        self.eval()
        K = int(n_steps or self.cfg.n_sample_steps)
        B = obs.shape[0]
        x = torch.randn(B, self.cfg.chunk_size, self.cfg.act_dim,
                        device=obs.device, dtype=obs.dtype)
        dt = 1.0 / K
        for k in range(K):
            t = torch.full((B,), k * dt, device=obs.device, dtype=obs.dtype)
            x = x + dt * self.velocity(x, t, obs)
        return x
