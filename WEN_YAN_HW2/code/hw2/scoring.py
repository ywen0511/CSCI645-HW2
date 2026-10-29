"""Per-episode scoring from saved rollout traces (numpy only, no simulator).

`score` turns the raw per-state traces of a batched rollout into protocol-v2
metrics, physical evidence and an outcome label; `summarize` aggregates them.
Both run on the arrays saved in results/episodes/*.npz, so every reported
metric can be recomputed without simulation (scripts/rescore_episodes.py).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from hw2 import metrics as MT


def score(raw: Dict[str, np.ndarray], radius: float, hold: int) -> Dict[str, np.ndarray]:
    """Per-episode protocol-v2 metrics, physical evidence and outcome labels."""
    m = MT.episode_metrics(raw["dist"], raw["done"], radius, hold,
                           nonfinite=raw.get("nonfinite"),
                           rewards=raw.get("rewards"))
    T = raw["dist"].shape[1] - 1
    mask = MT.valid_mask(m["n_valid_states"], T)
    # The reset state (k = 0) is returned before MJX computes body positions:
    # its box/gripper positions read as 0. Position-derived diagnostics
    # therefore use valid states k >= 1, and lift is measured from the box
    # height after the first control step (the box is still resting there).
    pmask = mask.copy()
    pmask[:, 0] = False
    contact = np.asarray(raw["contact_l"]) | np.asarray(raw["contact_r"])
    both = np.asarray(raw["contact_l"]) & np.asarray(raw["contact_r"])
    z = np.asarray(raw["box_z"], np.float64)
    lift = z - z[:, 1:2]
    max_lift = -MT.masked_min(-lift, pmask)
    ev = {
        "any_contact": MT.masked_any(contact, pmask),
        "both_finger_contact": MT.masked_any(both, pmask),
        "contact_steps": (contact & pmask).sum(1),
        "max_lift": max_lift,
        "lifted": np.nan_to_num(max_lift, nan=0.0) >= MT.LIFT_M,
        "min_grip": MT.masked_min(raw["grip"], pmask),
        "within_12mm": np.nan_to_num(MT.masked_min(raw["grip"], pmask),
                                     nan=np.inf) < MT.PROXIMITY_M,
        "obs_vs_sim_dist_maxdev": np.nan_to_num(MT.masked_min(
            -np.abs(np.asarray(raw["dist"]) - np.asarray(raw["dist_obs"])), pmask),
            nan=0.0) * -1.0,
        # protocol v1 on the SAME rollout: obs distance, full trace incl.
        # post-termination states. Used only to count changed outcomes.
        "success_v1": MT.legacy_success(raw["dist_obs"], radius, hold),
    }
    labels = np.array([MT.classify_episode(
        bool(m["success"][i]), bool(m["terminated"][i]),
        bool(ev["any_contact"][i]), bool(ev["lifted"][i]),
        float(m["min_dist"][i]), radius) for i in range(len(m["success"]))])
    return {**m, **ev, "label": labels}


def summarize(ep: Dict[str, np.ndarray], radius: float, hold: int) -> dict:
    """Aggregate a `score` table into a JSON-able summary."""
    suc = np.asarray(ep["success"], bool)
    n, k = int(suc.size), int(suc.sum())
    p, lo, hi = MT.wilson_ci(k, n)
    fail_labels = [str(l) for l in ep["label"] if l != "success"]
    tax = {c: fail_labels.count(c) for c in MT.FAILURE_ORDER}
    assert sum(tax.values()) == n - k, "failure categories must cover failures"
    causes = [str(c) for c in ep["term_cause"]]
    return {
        "protocol": MT.PROTOCOL_VERSION,
        "n_episodes": n, "n_success": k,
        "success_rate": float(p), "success_ci95": [float(lo), float(hi)],
        "min_dist_mean": float(np.nanmean(ep["min_dist"])),
        "final_dist_mean": float(np.nanmean(ep["final_dist"])),
        "return_mean": float(np.mean(ep["return"])),
        "return_std": float(np.std(ep["return"])),
        "terminated_frac": float(np.mean(ep["terminated"])),
        "term_causes": {c: causes.count(c) for c in sorted(set(causes))},
        "within_12mm_frac": float(np.mean(ep["within_12mm"])),
        "contact_frac": float(np.mean(ep["any_contact"])),
        "lifted_frac": float(np.mean(ep["lifted"])),
        "left_after_success": int(np.sum(ep["left_after_success"])),
        "final_inside_frac": float(np.mean(ep["final_inside"])),
        "obs_vs_sim_dist_maxdev": float(np.max(ep["obs_vs_sim_dist_maxdev"])),
        "n_success_v1_same_rollout": int(np.sum(ep["success_v1"])),
        "n_changed_vs_v1_same_rollout": int(np.sum(ep["success_v1"] != suc)),
        "failure_taxonomy": tax,
        "success_by_radius": {
            f"{r:g}": float(MT.episode_metrics(
                ep["_dist"], ep["_done"], r, hold,
                nonfinite=ep.get("_nonfinite"))["success"].mean())
            for r in (0.03, 0.05, 0.08, 0.10, 0.15)} if "_dist" in ep else {},
        "success_radius": radius, "success_hold": hold,
        "per_episode_success": suc.tolist(),
        "per_episode_label": [str(l) for l in ep["label"]],
    }
