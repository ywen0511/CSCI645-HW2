"""RL teacher: brax PPO on PandaPickCube, with intermediate checkpoints.

Follows the upstream recipe in mujoco_playground/learning/train_jax_ppo.py:
hyperparameters come from `manipulation_params.brax_ppo_config`, the env is
wrapped by `wrapper.wrap_for_brax_training`, and the policy/value network sizes
come from the config's `network_factory` block.

Intermediate parameter snapshots are written at every evaluation point so that a
"teacher quality" study (HW2 Part 3 Option C) remains possible without retraining.
"""
from __future__ import annotations

import functools
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np


def ppo_config(task: str):
    from mujoco_playground.config import manipulation_params
    return manipulation_params.brax_ppo_config(task)


def train_teacher(task: str, out_dir: str, seed: int = 0,
                  num_timesteps: Optional[int] = None) -> Dict[str, Any]:
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from hw2 import envs as E

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    env = E.make_env(task, impl="jax")
    eval_env = E.make_env(task, impl="jax")

    cfg = ppo_config(task)
    training_params = dict(cfg)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in training_params:
        nf = training_params.pop("network_factory")
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf)
    if num_timesteps is not None:
        training_params["num_timesteps"] = int(num_timesteps)

    print("[teacher] PPO params:", json.dumps(
        {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
         for k, v in training_params.items()}, indent=2))

    progress: List[Dict[str, float]] = []
    t_start = time.time()

    def progress_fn(step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": int(step), "wall_s": time.time() - t_start}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except Exception:
                pass
        progress.append(row)
        print(f"[teacher] step={step:>10,} "
              f"reward={row.get('eval/episode_reward', float('nan')):.3f} "
              f"({row['wall_s']:.0f}s)", flush=True)
        with open(os.path.join(out_dir, "ppo_progress.json"), "w") as f:
            json.dump(progress, f, indent=2)

    snapshots: List[int] = []

    def policy_params_fn(current_step: int, make_policy, params) -> None:
        """Snapshot parameters at each eval so teacher quality can be varied."""
        del make_policy
        p = os.path.join(ckpt_dir, f"params_step{int(current_step):010d}.pkl")
        with open(p, "wb") as f:
            pickle.dump(jax.device_get(params), f)
        snapshots.append(int(current_step))

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    make_inference_fn, params, _ = train_fn(
        environment=env, eval_env=eval_env,
        progress_fn=progress_fn, policy_params_fn=policy_params_fn,
    )

    final = os.path.join(out_dir, "teacher_params.pkl")
    with open(final, "wb") as f:
        pickle.dump(jax.device_get(params), f)

    info = {
        "task": task, "seed": seed,
        "num_timesteps": int(training_params["num_timesteps"]),
        "ppo_params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                       for k, v in training_params.items()},
        "final_params": final,
        "snapshots": sorted(snapshots),
        "wall_s": time.time() - t_start,
        "final_eval_reward": progress[-1].get("eval/episode_reward") if progress else None,
    }
    with open(os.path.join(out_dir, "teacher_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[teacher] done in {info['wall_s']:.0f}s -> {final}")
    return info


def load_teacher_policy(task: str, params_path: str, deterministic: bool = True,
                        normalize_observations: Optional[bool] = None):
    """Rebuild the PPO inference function from a saved parameter pickle.

    `normalize_observations` must match what training used, otherwise the policy
    silently receives unnormalised inputs. When None we read it back from the
    task's PPO config (the same source training used).
    """
    import functools as ft
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training import types as brax_types

    from hw2 import envs as E

    env = E.make_env(task, impl="jax")
    cfg = dict(ppo_config(task))
    nf = cfg.get("network_factory", {})
    make_networks = ft.partial(ppo_networks.make_ppo_networks, **nf) if nf \
        else ppo_networks.make_ppo_networks

    if normalize_observations is None:
        normalize_observations = bool(cfg.get("normalize_observations", False))
    preprocess = (running_statistics.normalize if normalize_observations
                  else brax_types.identity_observation_preprocessor)

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    networks = make_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=preprocess,
    )
    make_policy = ppo_networks.make_inference_fn(networks)
    return env, jax.jit(make_policy(params, deterministic=deterministic))
