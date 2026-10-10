"""ACT -- Action Chunking with Transformers, state-based variant.

Reference: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
Low-Cost Hardware" (RSS 2023), https://arxiv.org/abs/2304.13705
Reference implementation: https://github.com/tonyzhaozh/act (DETR-derived).

This is a self-contained re-implementation (no third-party library vendored, as
the submission instructions ask). It keeps the parts that define the method:

  * a transformer encoder-decoder that emits a whole chunk of H actions from one
    observation, with H learned decoder queries (action chunking);
  * a CVAE whose encoder sees the ground-truth action chunk at training time and
    produces a style latent z, regularised toward N(0, I); at test time z = 0;
  * an L1 reconstruction loss on the chunk plus a KL term.

Deviations from the 2023 reference, all deliberate and reported in the writeup:
  * No ResNet image backbone -- observations are the environment state vector,
    so the encoder memory is built from a linear projection of the state.
  * Pre-layer-norm blocks with GELU (the reference is post-LN with ReLU);
    pre-LN trains more stably without the warmup-sensitive tuning of post-LN.
  * AdamW with cosine decay and linear warmup, bf16 autocast, and an EMA copy of
    the weights used for evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ACTConfig:
    obs_dim: int
    act_dim: int
    chunk_size: int = 30
    hidden_dim: int = 512
    dim_feedforward: int = 2048
    n_heads: int = 8
    n_enc_layers: int = 4          # decoder-side transformer encoder
    n_dec_layers: int = 6
    n_cvae_enc_layers: int = 4     # CVAE posterior encoder
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0

    def __post_init__(self):
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads "
                f"({self.n_heads}); nn.MultiheadAttention asserts this.")

    def as_dict(self) -> dict:
        return asdict(self)


def _sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(d, dtype=torch.float32)[None, :]
    angle = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


class ACT(nn.Module):
    def __init__(self, cfg: ACTConfig):
        super().__init__()
        self.cfg = cfg
        H, D = cfg.chunk_size, cfg.hidden_dim

        # ---- CVAE posterior encoder: [CLS, obs, a_1..a_H] -> (mu, logvar) ----
        self.cvae_cls = nn.Parameter(torch.zeros(1, 1, D))
        self.cvae_obs_proj = nn.Linear(cfg.obs_dim, D)
        self.cvae_act_proj = nn.Linear(cfg.act_dim, D)
        self.register_buffer("cvae_pos", _sinusoidal(H + 2, D)[None], persistent=False)
        self.cvae_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, cfg.n_heads, cfg.dim_feedforward,
                                       cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True),
            num_layers=cfg.n_cvae_enc_layers, enable_nested_tensor=False,
        )
        self.to_latent = nn.Linear(D, 2 * cfg.latent_dim)

        # ---- transformer encoder over the conditioning tokens ----
        self.obs_proj = nn.Linear(cfg.obs_dim, D)
        self.latent_proj = nn.Linear(cfg.latent_dim, D)
        self.register_buffer("mem_pos", _sinusoidal(2, D)[None], persistent=False)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, cfg.n_heads, cfg.dim_feedforward,
                                       cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True),
            num_layers=cfg.n_enc_layers, enable_nested_tensor=False,
        )

        # ---- transformer decoder: H learned queries -> H actions ----
        self.query_embed = nn.Parameter(torch.zeros(1, H, D))
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(D, cfg.n_heads, cfg.dim_feedforward,
                                       cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True),
            num_layers=cfg.n_dec_layers,
        )
        self.head = nn.Linear(D, cfg.act_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        nn.init.trunc_normal_(self.cvae_cls, std=0.02)

    # ------------------------------------------------------------------
    def encode_posterior(self, obs: torch.Tensor, actions: torch.Tensor,
                         pad: Optional[torch.Tensor] = None):
        """q(z | obs, a_1..a_H) -> mu, logvar."""
        B = obs.shape[0]
        toks = torch.cat([
            self.cvae_cls.expand(B, -1, -1),
            self.cvae_obs_proj(obs).unsqueeze(1),
            self.cvae_act_proj(actions),
        ], dim=1)
        toks = toks + self.cvae_pos[:, : toks.shape[1]].to(toks.dtype)
        key_pad = None
        if pad is not None:
            # CLS and obs are always valid; padded chunk steps are masked out.
            key_pad = torch.cat(
                [torch.zeros(B, 2, device=pad.device, dtype=torch.bool),
                 pad.bool()], dim=1)
        h = self.cvae_encoder(toks, src_key_padding_mask=key_pad)
        mu, logvar = self.to_latent(h[:, 0]).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        mem = torch.stack([self.latent_proj(z), self.obs_proj(obs)], dim=1)
        mem = mem + self.mem_pos.to(mem.dtype)
        mem = self.encoder(mem)
        q = self.query_embed.expand(B, -1, -1)
        h = self.decoder(q, mem)
        return self.head(h)

    def forward(self, obs: torch.Tensor, actions: Optional[torch.Tensor] = None,
                pad: Optional[torch.Tensor] = None):
        """Training: pass `actions` to sample z from the posterior.
        Inference: omit `actions`; z is set to the prior mean (zeros)."""
        if actions is None:
            z = torch.zeros(obs.shape[0], self.cfg.latent_dim,
                            device=obs.device, dtype=obs.dtype)
            return self.decode(obs, z), None, None
        mu, logvar = self.encode_posterior(obs, actions, pad)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(obs, z), mu, logvar

    # ------------------------------------------------------------------
    def loss(self, obs: torch.Tensor, actions: torch.Tensor,
             pad: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        pred, mu, logvar = self.forward(obs, actions, pad)
        valid = (1.0 - pad).unsqueeze(-1)                       # (B,H,1)
        l1 = (F.l1_loss(pred, actions, reduction="none") * valid).sum() / \
             valid.sum().clamp(min=1.0) / actions.shape[-1]
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
        total = l1 + self.cfg.kl_weight * kl
        return total, {"loss": float(total.detach()), "l1": float(l1.detach()),
                       "kl": float(kl.detach())}

    @torch.no_grad()
    def predict_chunk(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, obs_dim) -> (B, H, act_dim), z = prior mean."""
        self.eval()
        return self.forward(obs, None, None)[0]
