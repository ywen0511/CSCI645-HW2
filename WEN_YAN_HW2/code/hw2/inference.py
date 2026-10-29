"""Action selection from chunk predictions (numpy only, unit-testable)."""
from __future__ import annotations

from typing import Optional

import numpy as np


class ActionSelector:
    """Turns per-step chunk predictions into executed actions for a batch."""

    def __init__(self, B: int, H: int, A: int, T: int, exec_mode: str,
                 exec_horizon: Optional[int] = None, ensemble_m: float = 0.01,
                 ensemble_order: str = "newest"):
        if ensemble_order not in ("newest", "oldest"):
            raise ValueError(f"unknown ensemble_order {ensemble_order!r}")
        self.order = ensemble_order
        if exec_mode not in ("ensemble", "chunk"):
            raise ValueError(f"unknown exec_mode {exec_mode!r}")
        self.mode, self.H = exec_mode, H
        self.k = int(exec_horizon or H)
        self.m = ensemble_m
        self.sum = np.zeros((B, T + H, A), np.float64)
        self.wt = np.zeros((B, T + H, 1), np.float64)
        self.pending = np.zeros((B, H, A), np.float32)

    def needs_prediction(self, t: int) -> bool:
        return self.mode == "ensemble" or t % self.k == 0

    def act(self, t: int, chunk: Optional[np.ndarray]) -> np.ndarray:
        if self.mode == "ensemble":
            # The chunk predicted at step s covers steps s..s+H-1 and its
            # entry for step t gets weight exp(-m (t-s)): the NEWEST prediction
            # weighs most. This is the historical protocol, kept unchanged.
            # Zhao et al. (2023) instead weight the OLDEST prediction most
            # (`ensemble_order="oldest"`, a supplementary diagnostic only).
            # With m = 0.01 and H = 30 the weights span 1 .. 0.75 either way.
            j = np.arange(self.H)
            w = np.exp(-self.m * (j if self.order == "newest"
                                  else (self.H - 1 - j)))[None, :, None]
            self.sum[:, t:t + self.H] += chunk.astype(np.float64) * w
            self.wt[:, t:t + self.H] += w
            a = self.sum[:, t] / np.maximum(self.wt[:, t], 1e-9)
        else:
            if chunk is not None:
                self.pending = chunk
            a = self.pending[:, t % self.k]
        return np.clip(a, -1.0, 1.0).astype(np.float32)
