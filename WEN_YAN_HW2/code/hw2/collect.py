"""Batched teacher rollouts: demonstration collection and policy evaluation.

Everything here runs as a single `jax.lax.scan` over a vmapped batch of
environments, so 100+ episodes are collected in one GPU pass.

Two details that matter for correctness:

* The raw playground env has NO auto-reset, so once `done` fires (box out of
  bounds / NaN) the remaining steps of that episode are meaningless. We record
  the done flags and truncate each episode at its first `done` when building the
  dataset, rather than storing post-termination garbage.
* The stored action is the action actually EXECUTED in the simulator (including
  injected noise, for the noisy-teacher condition), paired with the observation
  it was selected from -- which is what the assignment requires.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def rollout(env, policy_fn, rng_keys, episode_length: int,
            action_noise: float = 0.0, target_rel_slice: Tuple[int, int] = (49, 52),
            box_rel_slice: Tuple[int, int] = (46, 49),
            clip_actions: bool = True, spec=None,
            noise_key=None) -> Dict[str, np.ndarray]:
    """Roll out a batch of episodes under `policy_fn`.

    Returns arrays with leading shape (B, T) / (B, T, dim):
      obs      (B, T, obs_dim)  observation the action was selected from
      actions  (B, T, act_dim)  action executed in the simulator
      clean    (B, T, act_dim)  teacher action before noise (diagnostics)
      rewards  (B, T)
      dist     (B, T+1)         box->target distance, including the reset state
      grip_dist(B, T)           gripper->box distance (grasp diagnostic)
      done     (B, T)           raw env done of each transition (not sticky)

    With `spec` given, the protocol-v2 ground-truth traces of
    `hw2.envs.state_diagnostics_fn` are added as (B, T+1) arrays under their
    own names (`dist` then becomes the SIMULATOR distance and the observation
    distance moves to `dist_obs`), so `hw2.evaluate.score` applies unchanged.
    `noise_key` seeds the action noise (historically PRNGKey(0)).
    """
    import jax
    import jax.numpy as jp

    lo, hi = target_rel_slice
    blo, bhi = box_rel_slice
    B = rng_keys.shape[0]
    vreset, vstep = jax.vmap(env.reset), jax.vmap(env.step)
    vdiag = None
    if spec is not None:
        from hw2.envs import state_diagnostics_fn
        vdiag = jax.vmap(state_diagnostics_fn(env, spec))

    def step_fn(carry, _):
        state, rng = carry
        rng, k_pol, k_noise = jax.random.split(rng, 3)
        clean, _ = policy_fn(state.obs, k_pol)
        if action_noise > 0.0:
            noise = jax.random.normal(k_noise, clean.shape) * action_noise
            act = clean + noise
        else:
            act = clean
        if clip_actions:
            act = jp.clip(act, -1.0, 1.0)
        nstate = vstep(state, act)
        out = {
            "obs": state.obs,
            "actions": act,
            "clean": clean,
            "rewards": nstate.reward,
            "dist": jp.linalg.norm(nstate.obs[:, lo:hi], axis=-1),
            "grip_dist": jp.linalg.norm(nstate.obs[:, blo:bhi], axis=-1),
            "done": nstate.done,
        }
        if vdiag is not None:
            out["diag"] = vdiag(nstate)
        return (nstate, rng), out

    state0 = vreset(rng_keys)
    d0 = jp.linalg.norm(state0.obs[:, lo:hi], axis=-1)
    if noise_key is None:
        noise_key = jax.random.PRNGKey(0)
    (_, _), traj = jax.lax.scan(
        step_fn, (state0, noise_key), None, length=episode_length
    )
    diag = traj.pop("diag", None)
    out = {k: np.asarray(jax.device_get(jp.swapaxes(v, 0, 1))) for k, v in traj.items()}
    out["dist"] = np.concatenate([np.asarray(jax.device_get(d0))[:, None], out["dist"]], axis=1)
    out["done"] = out["done"].astype(bool)
    if diag is not None:
        d0g = jax.device_get(vdiag(state0))
        out["dist_obs_v1"] = out.pop("dist")
        for k, v in diag.items():
            v = np.asarray(jax.device_get(jp.swapaxes(v, 0, 1)))
            out[k] = np.concatenate([np.asarray(d0g[k])[:, None], v], axis=1)
    return out


def rollout_states(env, policy_fn, rng_key, episode_length: int,
                   action_noise: float = 0.0):
    """Single-episode rollout that KEEPS the env States, for rendering."""
    import jax
    import jax.numpy as jp

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(rng_key)
    states = [state]
    rng = rng_key
    for _ in range(episode_length):
        rng, k_pol, k_noise = jax.random.split(rng, 3)
        act, _ = policy_fn(state.obs[None], k_pol)
        act = act[0]
        if action_noise > 0.0:
            act = jp.clip(act + jax.random.normal(k_noise, act.shape) * action_noise,
                          -1.0, 1.0)
        state = step(state, act)
        states.append(state)
    return states


def truncate_at_done(traj: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-episode valid length: index of first `done`, else full length."""
    done = traj["done"].astype(bool)          # (B, T)
    B, T = done.shape
    first = np.where(done.any(1), done.argmax(1) + 1, T)
    return first.astype(np.int64)


def to_demoset(traj: Dict[str, np.ndarray], radius: float, hold: int,
               meta: Optional[dict] = None):
    """Convert rollout arrays into a DemoSet with per-episode success labels."""
    from hw2.data import DemoSet
    from hw2 import metrics as MT

    lengths = truncate_at_done(traj)
    # protocol v2: success only over valid states (terminal state excluded)
    v2 = MT.episode_metrics(traj["dist"], traj["done"], radius, hold,
                            nonfinite=~np.isfinite(traj["dist"]))["success"]
    B = lengths.shape[0]
    obs, act, rew, ends, suc = [], [], [], [], []
    running = 0
    for b in range(B):
        n = int(lengths[b])
        obs.append(traj["obs"][b, :n])
        act.append(traj["actions"][b, :n])
        rew.append(traj["rewards"][b, :n])
        running += n
        ends.append(running)
        suc.append(bool(v2[b]))
    m = {"success_radius": radius, "success_hold": hold}
    if meta:
        m.update(meta)
    return DemoSet(
        obs=np.concatenate(obs, 0).astype(np.float32),
        actions=np.concatenate(act, 0).astype(np.float32),
        rewards=np.concatenate(rew, 0).astype(np.float32),
        episode_ends=np.asarray(ends, dtype=np.int64),
        success=np.asarray(suc, dtype=np.bool_),
        meta=m,
    )


def eval_seeds(n: int, base: int = 7777):
    """Deterministic held-out evaluation keys, shared by teacher and all students."""
    import jax
    return jax.random.split(jax.random.PRNGKey(base), n)


def demo_seeds(n: int, base: int = 1000):
    """Demonstration-collection keys, disjoint from the evaluation keys."""
    import jax
    return jax.random.split(jax.random.PRNGKey(base), n)
