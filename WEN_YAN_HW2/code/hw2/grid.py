"""The 21-model experiment grid (7 conditions x training seeds 0, 1, 2).

Student hyperparameters are the resolved values recorded in
configs/trainlog_*.json for the historical run. The class defaults in
policies/act.py (hidden 512, ff 2048) and policies/flow.py (6 heads) are NOT
what was trained; the stage-2 command line always overrode them, so the
overrides live here explicitly.
"""
from __future__ import annotations

from typing import Dict, List

SEEDS = (0, 1, 2)
TRAIN = dict(steps=6000, batch_size=128, lr=1e-4, weight_decay=1e-4,
             warmup=200, ema_decay=0.999, chunk=30)
ACT_OVERRIDES = {"hidden_dim": 256, "dim_feedforward": 1024}
FLOW_OVERRIDES = {"hidden_dim": 256, "n_layers": 4, "n_heads": 4}


def build_grid() -> List[Dict]:
    g = []
    # Option A, quantity: nested prefixes of the success-filtered clean pool
    for n in (25, 50, 100):
        for s in SEEDS:
            g.append({"name": f"act_clean_n{n}_s{s}", "algo": "act",
                      "dataset": "demos_sigma0.npz", "n_demos": n, "seed": s,
                      "group": "quantity", "successful_only": True,
                      "noise": 0.0, "policy_overrides": ACT_OVERRIDES})
    # Option B, teacher action noise: first 100 episodes, failures INCLUDED
    for sig in ("0.5", "1", "2"):
        for s in SEEDS:
            g.append({"name": f"act_noise{sig}_n100_s{s}", "algo": "act",
                      "dataset": f"demos_sigma{sig}.npz", "n_demos": 100,
                      "seed": s, "group": "noise", "successful_only": False,
                      "noise": float(sig), "policy_overrides": ACT_OVERRIDES})
    # supplementary: flow matching on the clean 100-demo set
    for s in SEEDS:
        g.append({"name": f"flow_clean_n100_s{s}", "algo": "flow",
                  "dataset": "demos_sigma0.npz", "n_demos": 100, "seed": s,
                  "group": "algorithm", "successful_only": True, "noise": 0.0,
                  "policy_overrides": FLOW_OVERRIDES})
    return g


def by_name() -> Dict[str, Dict]:
    return {c["name"]: c for c in build_grid()}
