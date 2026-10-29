"""Closed-loop evaluation of trained student policies (protocol v2).

The student is a torch module; the environment is JAX/MJX. We vmap the env over
all evaluation episodes and run them in lockstep, exchanging numpy arrays every
control step, so 100 episodes cost one batched rollout.

Inference modes (ACT paper):
  ensemble  predict a chunk at every step and average all predictions that
            cover the current step with weights exp(-m * age), m = 0.01, i.e.
            the NEWEST prediction weighs most -- the historical headline
            protocol, unchanged. (ACT's reference weights the OLDEST most; see
            ActionSelector.)
  chunk     predict H actions and execute `exec_horizon` of them open loop.
            exec_horizon = 1 is per-step replanning (first action only).

Scoring follows `hw2.metrics` (termination-aware). Every rollout returns the
raw per-state traces so the metrics can be recomputed without simulation; see
`save_episodes`.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from hw2 import metrics as MT
from hw2.inference import ActionSelector  # noqa: F401  (re-exported)
from hw2.scoring import score, summarize  # noqa: F401  (re-exported)

MODES = {
    "ensemble": dict(exec_mode="ensemble"),
    "replan1": dict(exec_mode="chunk", exec_horizon=1),
}
# diagnostics only (never headline numbers)
DIAG_MODES = {
    "ensemble_oldest": dict(exec_mode="ensemble", ensemble_order="oldest"),
    "chunk_full": dict(exec_mode="chunk"),
    "chunk5": dict(exec_mode="chunk", exec_horizon=5),
}


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


def make_predictor(model, norm: Dict[str, np.ndarray], device: str = "cuda"):
    """obs (B, obs_dim) numpy -> de-normalised action chunk (B, H, A) numpy."""
    om = torch.as_tensor(norm["obs_mean"], device=device)
    os_ = torch.as_tensor(norm["obs_std"], device=device)
    am = torch.as_tensor(norm["act_mean"], device=device)
    as_ = torch.as_tensor(norm["act_std"], device=device)

    @torch.no_grad()
    def predict(obs_np: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(np.asarray(obs_np, np.float32), device=device)
        o = (o - om) / os_
        return (model.predict_chunk(o) * as_ + am).float().cpu().numpy()
    return predict


def _stack_diag(rows: List[dict]) -> Dict[str, np.ndarray]:
    keys = rows[0].keys()
    return {k: np.stack([np.asarray(r[k]) for r in rows], 1) for k in keys}


@torch.no_grad()
def rollout_students(env, spec, model, norm: Dict[str, np.ndarray], rng_keys,
                     exec_mode: str = "ensemble",
                     exec_horizon: Optional[int] = None,
                     ensemble_m: float = 0.01, ensemble_order: str = "newest",
                     device: str = "cuda",
                     keep_states_for: Optional[List[int]] = None):
    """Batched student rollout. Returns raw per-state traces (see metrics.py).

    The batch keeps stepping after an episode terminates (the env has no
    auto-reset); those states are recorded but masked out by the metrics.
    """
    import jax
    import jax.numpy as jp
    from hw2.envs import state_diagnostics_fn

    B = int(rng_keys.shape[0])
    T = int(spec.episode_length)
    H, A = int(model.cfg.chunk_size), int(model.cfg.act_dim)
    vreset = jax.jit(jax.vmap(env.reset))
    vstep = jax.jit(jax.vmap(env.step))
    vdiag = jax.jit(jax.vmap(state_diagnostics_fn(env, spec)))
    predict = make_predictor(model, norm, device)
    sel = ActionSelector(B, H, A, T, exec_mode, exec_horizon, ensemble_m,
                         ensemble_order)

    state = vreset(rng_keys)
    diags = [jax.device_get(vdiag(state))]
    done, rewards, actions = [], [], []
    kept: Dict[int, list] = {i: [jax.tree_util.tree_map(lambda x: x[i], state)]
                             for i in (keep_states_for or [])}
    for t in range(T):
        obs_np = np.asarray(state.obs, dtype=np.float32)
        chunk = predict(obs_np) if sel.needs_prediction(t) else None
        act = sel.act(t, chunk)
        state = vstep(state, jp.asarray(act))
        diags.append(jax.device_get(vdiag(state)))
        done.append(np.asarray(state.done))
        rewards.append(np.asarray(state.reward))
        actions.append(act)
        for i in kept:
            kept[i].append(jax.tree_util.tree_map(lambda x: x[i], state))
    raw = _stack_diag(diags)                       # each (B, T+1)
    raw["done"] = np.stack(done, 1).astype(bool)   # (B, T) raw, not sticky
    raw["rewards"] = np.stack(rewards, 1)
    raw["actions"] = np.stack(actions, 1)          # (B, T, A) executed
    raw["kept_states"] = kept
    return raw


def evaluate(env, spec, model, norm, rng_keys, radius, hold, mode: str,
             device: str = "cuda", keep_states_for=None):
    """Rollout + score + summary for one inference mode."""
    kw = MODES[mode] if mode in MODES else DIAG_MODES[mode]
    raw = rollout_students(env, spec, model, norm, rng_keys, device=device,
                           keep_states_for=keep_states_for, **kw)
    ep = score(raw, radius, hold)
    ep.update({"_dist": raw["dist"], "_done": raw["done"],
               "_nonfinite": raw["nonfinite"]})
    s = summarize(ep, radius, hold)
    s["mode"] = mode
    return raw, ep, s


def key_data(rng_keys) -> np.ndarray:
    """Raw uint32 data of a batch of jax PRNG keys (legacy or typed)."""
    import jax
    k = rng_keys
    try:
        if jax.dtypes.issubdtype(k.dtype, jax.dtypes.prng_key):
            k = jax.random.key_data(k)
    except Exception:
        pass
    return np.asarray(k, dtype=np.uint32)


def save_episodes(path: str, rng_keys, raw: Dict[str, np.ndarray],
                  ep: Dict[str, np.ndarray], meta: dict) -> str:
    """Everything needed to recompute the metrics without simulation."""
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrs = {f"raw_{k}": np.asarray(v) for k, v in raw.items()
            if k != "kept_states" and k != "actions"}
    arrs["raw_actions"] = np.asarray(raw["actions"], np.float16)
    for k, v in ep.items():
        if k.startswith("_"):
            continue
        arrs[f"ep_{k}"] = np.asarray(v).astype(str) if k in ("label", "term_cause") \
            else np.asarray(v)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, reset_keys=key_data(rng_keys),
                        meta=json.dumps(meta), **arrs)
    os.replace(tmp, path)
    return path
