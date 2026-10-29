"""Imitation-learning training loop (ACT or flow matching), state-based.

Conditions are built by taking a NESTED prefix of the demonstration pool, so the
25-demo set is a strict subset of the 50-demo set, which is a strict subset of
the 100-demo set. That is what makes the quantity ablation interpretable: the
only thing that changes is how many of the same demonstrations the student sees.

Normalisation statistics are fit on the selected subset only (never the pool).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch


def build_policy(algo: str, obs_dim: int, act_dim: int, chunk: int,
                 overrides: Optional[dict] = None):
    overrides = dict(overrides or {})
    if algo == "act":
        from hw2.policies.act import ACT, ACTConfig
        cfg = ACTConfig(obs_dim=obs_dim, act_dim=act_dim, chunk_size=chunk, **overrides)
        return ACT(cfg), cfg
    if algo == "flow":
        from hw2.policies.flow import FlowPolicy, FlowConfig
        cfg = FlowConfig(obs_dim=obs_dim, act_dim=act_dim, chunk_size=chunk, **overrides)
        return FlowPolicy(cfg), cfg
    raise ValueError(f"unknown algo {algo!r}")


class EMA:
    """Exponential moving average of parameters, used for evaluation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1.0 - self.decay)

    def state_dict(self, model: torch.nn.Module) -> dict:
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in self.shadow.items():
            sd[k] = v.to(sd[k].dtype)
        return sd


def cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float,
                       min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def train(demos_path: str, out_dir: str, algo: str = "act", n_demos: int = 100,
          chunk: int = 30, steps: int = 6000, batch_size: int = 128,
          lr: float = 1e-4, weight_decay: float = 1e-4, warmup: int = 200,
          seed: int = 0, ema_decay: float = 0.999, num_workers: int = 4,
          successful_only: bool = True, policy_overrides: Optional[dict] = None,
          log_every: int = 50) -> Dict:
    from hw2.data import DemoSet, Normalizer, ActionChunkDataset, make_torch_loader

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    pool = DemoSet.load(demos_path)
    if successful_only:
        pool = pool.successful_only()
    if n_demos > pool.n_episodes:
        raise ValueError(f"requested {n_demos} demos but pool has {pool.n_episodes}")
    # NESTED prefix: condition k is a strict subset of condition k+1.
    demos = pool.subset(list(range(n_demos)))

    norm = Normalizer.fit(demos)
    norm.save(os.path.join(out_dir, "normalizer.npz"))
    ds = ActionChunkDataset(demos, chunk, norm)
    loader = make_torch_loader(ds, batch_size, shuffle=True,
                               num_workers=num_workers, seed=seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = build_policy(algo, demos.obs.shape[1], demos.actions.shape[1],
                              chunk, policy_overrides)
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            betas=(0.9, 0.95))
    ema = EMA(model, ema_decay)
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()

    summary = demos.summary()
    print(f"[train] algo={algo} n_demos={n_demos} transitions={summary['n_transitions']} "
          f"obs={demos.obs.shape[1]} act={demos.actions.shape[1]} chunk={chunk} "
          f"params={n_params/1e6:.2f}M dev={dev} bf16={use_bf16}", flush=True)

    history = []
    t0 = time.time()
    step = 0
    model.train()
    while step < steps:
        for obs, act, pad in loader:
            if step >= steps:
                break
            for g in opt.param_groups:
                g["lr"] = cosine_with_warmup(step, steps, warmup, lr)
            obs = obs.to(dev, non_blocking=True)
            act = act.to(dev, non_blocking=True)
            pad = pad.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss, logs = model.loss(obs, act, pad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            if step % log_every == 0 or step == steps - 1:
                row = {"step": step, "lr": opt.param_groups[0]["lr"],
                       "grad_norm": float(gn), "wall_s": time.time() - t0, **logs}
                history.append(row)
                print(f"[train] step {step:>6d}/{steps} "
                      + " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                      + f" lr={row['lr']:.2e}", flush=True)
            step += 1

    ckpt = {
        "algo": algo, "cfg": cfg.as_dict(), "state_dict": ema.state_dict(model),
        "raw_state_dict": model.state_dict(),
        "normalizer": {"obs_mean": norm.obs_mean, "obs_std": norm.obs_std,
                       "act_mean": norm.act_mean, "act_std": norm.act_std},
        "n_demos": n_demos, "chunk": chunk, "seed": seed, "steps": steps,
        "n_params": n_params, "demos_path": os.path.basename(demos_path),
        "demo_summary": summary,
        "episode_indices": demos.meta.get("subset_episode_indices"),
    }
    torch.save(ckpt, os.path.join(out_dir, "policy.pt"))
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"history": history, "config": {
            "algo": algo, "n_demos": n_demos, "chunk": chunk, "steps": steps,
            "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
            "warmup": warmup, "seed": seed, "ema_decay": ema_decay,
            "successful_only": successful_only,
            "policy_cfg": cfg.as_dict(), "n_params": n_params,
            "demos_path": os.path.basename(demos_path), "demo_summary": summary,
        }, "wall_s": time.time() - t0}, f, indent=2)
    print(f"[train] done in {time.time()-t0:.0f}s -> {out_dir}/policy.pt", flush=True)
    return {"out_dir": out_dir, "history": history, "n_params": n_params}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--algo", default="act", choices=["act", "flow"])
    ap.add_argument("--n_demos", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--include_failures", action="store_true")
    a = ap.parse_args()
    train(a.demos, a.out_dir, algo=a.algo, n_demos=a.n_demos, chunk=a.chunk,
          steps=a.steps, batch_size=a.batch_size, lr=a.lr,
          weight_decay=a.weight_decay, seed=a.seed, num_workers=a.num_workers,
          successful_only=not a.include_failures)


if __name__ == "__main__":
    main()
