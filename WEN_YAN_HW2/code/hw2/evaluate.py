"""Closed-loop evaluation of a trained student policy in the simulator.

The student is a torch module; the environment is JAX/MJX. We vmap the env over
all evaluation episodes and run them in lockstep, exchanging numpy arrays at the
chunk boundaries, so 100 episodes cost one batched rollout.

Two inference modes from the ACT paper:
  chunk     -- predict H actions, execute `exec_horizon` of them open loop,
               then re-plan. `exec_horizon == H` is pure open-loop chunking.
  ensemble  -- predict at every step and temporally ensemble all predictions
               that cover the current timestep with exponential weights
               w_i ∝ exp(-m i) (ACT's "temporal ensembling").

Evaluation seeds are fixed and shared across every condition, and are disjoint
from the demonstration-collection seeds, so no student is evaluated on an
initial condition it was trained on.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch


def load_policy(ckpt_path: str, device: str = "cuda", use_ema: bool = True):
    from hw2.train_il import build_policy
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model, _ = build_policy(
        ck["algo"], cfg["obs_dim"], cfg["act_dim"], cfg["chunk_size"],
        {k: v for k, v in cfg.items()
         if k not in ("obs_dim", "act_dim", "chunk_size")},
    )
    model.load_state_dict(ck["state_dict"] if use_ema else ck["raw_state_dict"])
    model.to(device).eval()
    n = ck["normalizer"]
    return model, ck, {k: np.asarray(v, dtype=np.float32) for k, v in n.items()}


@torch.no_grad()
def rollout_students(env, model, norm: Dict[str, np.ndarray], rng_keys,
                     episode_length: int, exec_mode: str = "ensemble",
                     exec_horizon: Optional[int] = None,
                     ensemble_m: float = 0.01,
                     target_rel_slice=(49, 52),
                     box_rel_slice=(46, 49),
                     device: str = "cuda",
                     keep_states_for: Optional[List[int]] = None):
    """Run a batch of evaluation episodes under the student policy."""
    import jax
    import jax.numpy as jp

    lo, hi = target_rel_slice
    blo, bhi = box_rel_slice
    B = rng_keys.shape[0]
    H = int(model.cfg.chunk_size)
    A = int(model.cfg.act_dim)
    k_exec = int(exec_horizon or H)

    vreset = jax.jit(jax.vmap(env.reset))
    vstep = jax.jit(jax.vmap(env.step))

    om = torch.as_tensor(norm["obs_mean"], device=device)
    os_ = torch.as_tensor(norm["obs_std"], device=device)
    am = torch.as_tensor(norm["act_mean"], device=device)
    as_ = torch.as_tensor(norm["act_std"], device=device)

    def predict(obs_np: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
        o = (o - om) / os_
        chunk = model.predict_chunk(o)              # (B, H, A) normalised
        chunk = chunk * as_ + am
        return chunk.float().cpu().numpy()

    state = vreset(rng_keys)
    dists = [np.linalg.norm(np.asarray(state.obs[:, lo:hi]), axis=-1)]
    dones = np.zeros((B,), dtype=bool)
    done_flags, rewards, grip = [], [], []
    kept: Dict[int, list] = {i: [] for i in (keep_states_for or [])}

    # temporal-ensemble buffer: predictions for absolute timesteps
    ens_sum = np.zeros((B, episode_length + H, A), dtype=np.float64)
    ens_wt = np.zeros((B, episode_length + H, 1), dtype=np.float64)
    pending = np.zeros((B, H, A), dtype=np.float32)

    for t in range(episode_length):
        obs_np = np.asarray(state.obs, dtype=np.float32)
        if exec_mode == "ensemble":
            chunk = predict(obs_np)                                  # (B,H,A)
            w = np.exp(-ensemble_m * np.arange(H))[None, :, None]    # (1,H,1)
            ens_sum[:, t:t + H] += chunk.astype(np.float64) * w
            ens_wt[:, t:t + H] += w
            act = ens_sum[:, t] / np.maximum(ens_wt[:, t], 1e-9)
        elif exec_mode == "chunk":
            if t % k_exec == 0:
                pending = predict(obs_np)
            act = pending[:, t % k_exec]
        else:
            raise ValueError(f"unknown exec_mode {exec_mode!r}")

        act = np.clip(act, -1.0, 1.0).astype(np.float32)
        state = vstep(state, jp.asarray(act))
        d = np.asarray(state.obs[:, lo:hi])
        dists.append(np.linalg.norm(d, axis=-1))
        rewards.append(np.asarray(state.reward))
        grip.append(np.linalg.norm(np.asarray(state.obs[:, blo:bhi]), axis=-1))
        dones |= np.asarray(state.done).astype(bool)
        done_flags.append(dones.copy())
        for i in kept:
            kept[i].append(jax.tree_util.tree_map(lambda x: x[i], state))

    return {
        "dist": np.stack(dists, 1),                  # (B, T+1)
        "rewards": np.stack(rewards, 1),             # (B, T)
        "done": np.stack(done_flags, 1),             # (B, T)
        "grip_dist": np.stack(grip, 1),              # (B, T)
        "kept_states": kept,
    }


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = float(z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d)
    # plain python floats: np.float64 is not JSON-serialisable and would only
    # blow up at the very end of a long job, when the report is written.
    return (float(p), float(max(0.0, c - h)), float(min(1.0, c + h)))


def summarize(res: Dict[str, np.ndarray], radius: float, hold: int) -> dict:
    from hw2.envs import success_from_distances
    suc = success_from_distances(res["dist"], radius, hold)
    n = int(len(suc))
    k = int(suc.sum())
    p, lo, hi = wilson_ci(k, n)
    return {
        "n_episodes": n, "n_success": k,
        "success_rate": float(p), "success_ci95": [float(lo), float(hi)],
        "min_dist_mean": float(res["dist"].min(1).mean()),
        "final_dist_mean": float(res["dist"][:, -1].mean()),
        "return_mean": float(res["rewards"].sum(1).mean()),
        "return_std": float(res["rewards"].sum(1).std()),
        "out_of_bounds_frac": float(res["done"][:, -1].mean()),
        "reached_box_frac": (float((res["grip_dist"].min(1) < 0.012).mean())
                             if "grip_dist" in res else None),
        "min_grip_dist_mean": (float(res["grip_dist"].min(1).mean())
                               if "grip_dist" in res else None),
        "success_radius": radius, "success_hold": hold,
        "per_episode_success": suc.astype(bool).tolist(),
    }


@torch.no_grad()
def rollout_student_states(env, model, norm: Dict[str, np.ndarray], rng_key,
                           episode_length: int, exec_mode: str = "ensemble",
                           exec_horizon: Optional[int] = None,
                           ensemble_m: float = 0.01,
                           target_rel_slice=(49, 52), box_rel_slice=(46, 49),
                           device: str = "cuda"):
    """Single-episode student rollout that KEEPS env States, for rendering."""
    import jax
    import jax.numpy as jp

    lo, hi = target_rel_slice
    blo, bhi = box_rel_slice
    H = int(model.cfg.chunk_size)
    A = int(model.cfg.act_dim)
    k_exec = int(exec_horizon or H)
    reset, step = jax.jit(env.reset), jax.jit(env.step)

    om = torch.as_tensor(norm["obs_mean"], device=device)
    os_ = torch.as_tensor(norm["obs_std"], device=device)
    am = torch.as_tensor(norm["act_mean"], device=device)
    as_ = torch.as_tensor(norm["act_std"], device=device)

    def predict(obs_np):
        o = torch.as_tensor(obs_np[None], device=device, dtype=torch.float32)
        o = (o - om) / os_
        c = model.predict_chunk(o) * as_ + am
        return c[0].float().cpu().numpy()

    state = reset(rng_key)
    states = [state]
    dists = [float(jp.linalg.norm(state.obs[lo:hi]))]
    grips = []
    ens_sum = np.zeros((episode_length + H, A), dtype=np.float64)
    ens_wt = np.zeros((episode_length + H, 1), dtype=np.float64)
    pending = np.zeros((H, A), dtype=np.float32)
    for t in range(episode_length):
        obs_np = np.asarray(state.obs, dtype=np.float32)
        if exec_mode == "ensemble":
            chunk = predict(obs_np)
            w = np.exp(-ensemble_m * np.arange(H))[:, None]
            ens_sum[t:t + H] += chunk.astype(np.float64) * w
            ens_wt[t:t + H] += w
            act = ens_sum[t] / np.maximum(ens_wt[t], 1e-9)
        else:
            if t % k_exec == 0:
                pending = predict(obs_np)
            act = pending[t % k_exec]
        act = np.clip(act, -1.0, 1.0).astype(np.float32)
        state = step(state, jp.asarray(act))
        states.append(state)
        dists.append(float(jp.linalg.norm(state.obs[lo:hi])))
        grips.append(float(jp.linalg.norm(state.obs[blo:bhi])))
    return states, np.asarray(dists), np.asarray(grips)
