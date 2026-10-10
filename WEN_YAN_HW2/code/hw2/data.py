"""Demonstration storage, normalization and action-chunk dataset.

Demonstrations are stored as a single flat .npz per dataset with explicit
episode boundaries so that chunking never crosses an episode edge.

Normalization statistics are ALWAYS computed from the episodes actually used
for training a given condition (never from the full pool), so that the
25-demo condition cannot leak information from the 100-demo pool.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
@dataclass
class DemoSet:
    """Flat demonstration buffer with episode boundaries."""

    obs: np.ndarray            # (T_total, obs_dim) float32
    actions: np.ndarray        # (T_total, act_dim) float32
    rewards: np.ndarray        # (T_total,) float32
    episode_ends: np.ndarray   # (n_ep,) int64, exclusive
    success: np.ndarray        # (n_ep,) bool
    meta: dict

    # -- episode indexing -------------------------------------------------
    @property
    def n_episodes(self) -> int:
        return int(len(self.episode_ends))

    @property
    def episode_starts(self) -> np.ndarray:
        return np.concatenate([[0], self.episode_ends[:-1]]).astype(np.int64)

    @property
    def episode_lengths(self) -> np.ndarray:
        return (self.episode_ends - self.episode_starts).astype(np.int64)

    def episode_slice(self, i: int) -> slice:
        return slice(int(self.episode_starts[i]), int(self.episode_ends[i]))

    # -- io ---------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            obs=self.obs.astype(np.float32),
            actions=self.actions.astype(np.float32),
            rewards=self.rewards.astype(np.float32),
            episode_ends=self.episode_ends.astype(np.int64),
            success=self.success.astype(np.bool_),
            meta=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path: str) -> "DemoSet":
        z = np.load(path, allow_pickle=False)
        return cls(
            obs=z["obs"],
            actions=z["actions"],
            rewards=z["rewards"],
            episode_ends=z["episode_ends"],
            success=z["success"],
            meta=json.loads(str(z["meta"])),
        )

    # -- subsetting -------------------------------------------------------
    def subset(self, idx: Sequence[int]) -> "DemoSet":
        """Return a DemoSet containing only the episodes in `idx` (in order)."""
        idx = [int(i) for i in idx]
        if len(idx) == 0:
            raise ValueError(
                "DemoSet.subset() called with an empty episode list. This "
                "usually means the teacher produced zero successful episodes, "
                "which is a teacher-quality problem, not a data-plumbing one."
            )
        obs, act, rew, ends, suc = [], [], [], [], []
        running = 0
        for i in idx:
            sl = self.episode_slice(i)
            n = sl.stop - sl.start
            obs.append(self.obs[sl])
            act.append(self.actions[sl])
            rew.append(self.rewards[sl])
            running += n
            ends.append(running)
            suc.append(bool(self.success[i]))
        meta = dict(self.meta)
        meta["subset_of"] = meta.get("name", "?")
        meta["subset_episode_indices"] = idx
        return DemoSet(
            obs=np.concatenate(obs, 0).astype(np.float32),
            actions=np.concatenate(act, 0).astype(np.float32),
            rewards=np.concatenate(rew, 0).astype(np.float32),
            episode_ends=np.asarray(ends, dtype=np.int64),
            success=np.asarray(suc, dtype=np.bool_),
            meta=meta,
        )

    def successful_only(self) -> "DemoSet":
        return self.subset(np.flatnonzero(self.success).tolist())

    # -- summary ----------------------------------------------------------
    def summary(self) -> dict:
        L = self.episode_lengths
        return {
            "n_episodes": self.n_episodes,
            "n_transitions": int(self.obs.shape[0]),
            "obs_dim": int(self.obs.shape[1]),
            "act_dim": int(self.actions.shape[1]),
            "ep_len_mean": float(L.mean()) if len(L) else 0.0,
            "ep_len_std": float(L.std()) if len(L) else 0.0,
            "ep_len_min": int(L.min()) if len(L) else 0,
            "ep_len_max": int(L.max()) if len(L) else 0,
            "success_fraction": float(self.success.mean()) if len(L) else 0.0,
            "n_success": int(self.success.sum()),
            "return_mean": float(
                np.mean([self.rewards[self.episode_slice(i)].sum()
                         for i in range(self.n_episodes)])
            ) if len(L) else 0.0,
        }


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
@dataclass
class Normalizer:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    @classmethod
    def fit(cls, demos: DemoSet, eps: float = 1e-6) -> "Normalizer":
        o, a = demos.obs, demos.actions
        # A constant obs/action dimension would otherwise blow up to +-inf.
        o_std = o.std(0)
        a_std = a.std(0)
        o_std = np.where(o_std < eps, 1.0, o_std)
        a_std = np.where(a_std < eps, 1.0, a_std)
        return cls(o.mean(0).astype(np.float32), o_std.astype(np.float32),
                   a.mean(0).astype(np.float32), a_std.astype(np.float32))

    def norm_obs(self, o: np.ndarray) -> np.ndarray:
        return (o - self.obs_mean) / self.obs_std

    def norm_act(self, a: np.ndarray) -> np.ndarray:
        return (a - self.act_mean) / self.act_std

    def denorm_act(self, a: np.ndarray) -> np.ndarray:
        return a * self.act_std + self.act_mean

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **{k: v for k, v in asdict(self).items()})

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        z = np.load(path)
        return cls(z["obs_mean"], z["obs_std"], z["act_mean"], z["act_std"])


# --------------------------------------------------------------------------
# chunked dataset
# --------------------------------------------------------------------------
class ActionChunkDataset:
    """Maps a timestep t to (obs_t, actions[t:t+H], pad_mask).

    Chunks never cross an episode boundary: past the end of an episode the
    chunk is padded by repeating the final action and the pad mask is set,
    exactly as in the reference ACT implementation.
    """

    def __init__(self, demos: DemoSet, horizon: int, normalizer: Normalizer,
                 obs_history: int = 1):
        self.demos = demos
        self.horizon = int(horizon)
        self.norm = normalizer
        self.obs_history = int(obs_history)

        # Flat index -> (episode, offset within episode)
        idx = []
        for e in range(demos.n_episodes):
            n = int(demos.episode_lengths[e])
            for t in range(n):
                idx.append((e, t))
        self._index = np.asarray(idx, dtype=np.int64)

        self._obs_n = self.norm.norm_obs(demos.obs).astype(np.float32)
        self._act_n = self.norm.norm_act(demos.actions).astype(np.float32)

    def __len__(self) -> int:
        return int(len(self._index))

    def __getitem__(self, i: int):
        e, t = self._index[i]
        s = int(self.demos.episode_starts[e])
        n = int(self.demos.episode_lengths[e])
        H = self.horizon

        # observation (optionally a short history, zero-padded at episode start)
        if self.obs_history == 1:
            obs = self._obs_n[s + t]
        else:
            lo = t - self.obs_history + 1
            rows = []
            for k in range(lo, t + 1):
                rows.append(self._obs_n[s + max(k, 0)])
            obs = np.stack(rows, 0)

        avail = min(H, n - t)
        chunk = np.empty((H, self._act_n.shape[1]), dtype=np.float32)
        chunk[:avail] = self._act_n[s + t: s + t + avail]
        if avail < H:
            chunk[avail:] = self._act_n[s + n - 1]  # repeat final action
        pad = np.zeros((H,), dtype=np.float32)
        pad[avail:] = 1.0  # 1 == padded == excluded from the loss
        return obs.astype(np.float32), chunk, pad


def make_torch_loader(dataset: "ActionChunkDataset", batch_size: int,
                      shuffle: bool = True, num_workers: int = 4,
                      seed: Optional[int] = None):
    """Wrap ActionChunkDataset in a torch DataLoader (imported lazily)."""
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset

    class _TD(TorchDataset):
        def __len__(self_):
            return len(dataset)

        def __getitem__(self_, i):
            o, a, p = dataset[i]
            return (torch.from_numpy(np.ascontiguousarray(o)),
                    torch.from_numpy(np.ascontiguousarray(a)),
                    torch.from_numpy(np.ascontiguousarray(p)))

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))
    return DataLoader(_TD(), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=True, generator=g,
                      persistent_workers=num_workers > 0)
