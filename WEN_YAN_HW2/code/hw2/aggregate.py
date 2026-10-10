"""Aggregate per-condition results into tables.

Every number that appears in the report is computed here from the raw JSON
produced by the evaluation stage; nothing is transcribed by hand.

Because all conditions are evaluated on the SAME held-out initial conditions,
condition-vs-condition comparisons are PAIRED: we use a paired bootstrap over
episodes (and McNemar's exact test for the per-episode success indicators),
which is far tighter than comparing two independent Wilson intervals.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_conditions(res_dir: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = float(z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d)
    return (float(p), float(max(0.0, c - h)), float(min(1.0, c + h)))


def group_key(rec: dict) -> Tuple:
    return (rec["group"], rec["algo"], int(rec["n_demos"]), float(rec["noise"]))


def aggregate(records: List[dict], eval_key: str = "eval") -> List[dict]:
    """Collapse seeds within each condition cell, using `eval_key`'s block."""
    cells: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in records:
        cells[group_key(r)].append(r)

    rows = []
    for key, recs in sorted(cells.items()):
        grp, algo, n_demos, noise = key
        recs = [r for r in recs if eval_key in r]
        if not recs:
            continue
        per_seed = [r[eval_key]["success_rate"] for r in recs]
        seeds = [r["seed"] for r in recs]
        # pooled over every evaluation episode of every seed
        succ = np.concatenate([np.asarray(r[eval_key]["per_episode_success"], bool)
                               for r in recs])
        p, lo, hi = wilson_ci(int(succ.sum()), int(succ.size))
        rows.append({
            "group": grp, "algo": algo, "n_demos": n_demos, "noise": noise,
            "seeds": seeds, "n_seeds": len(recs),
            "success_per_seed": per_seed,
            "success_mean": float(np.mean(per_seed)),
            "success_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
            "success_pooled": p, "ci95_lo": lo, "ci95_hi": hi,
            "eval_key": eval_key,
            "n_eval_per_seed": int(recs[0][eval_key]["n_episodes"]),
            "n_eval_total": int(succ.size),
            "min_dist_mean": float(np.mean([r[eval_key]["min_dist_mean"] for r in recs])),
            "return_mean": float(np.mean([r[eval_key]["return_mean"] for r in recs])),
            "oob_frac": float(np.mean([r[eval_key]["out_of_bounds_frac"] for r in recs])),
            "reached_box_frac": float(np.mean(
                [r[eval_key].get("reached_box_frac") or 0.0 for r in recs])),
            "n_params": int(recs[0]["n_params"]),
            "demo_transitions": int(recs[0]["demo_summary"]["n_transitions"]),
            "exec_mode": recs[0]["exec_mode"],
            "names": [r["name"] for r in recs],
        })
    return rows


def _succ_matrix(recs: List[dict], eval_key: str = "eval") -> np.ndarray:
    """(n_seeds, n_eval) boolean success matrix, episodes aligned across seeds."""
    return np.stack([np.asarray(r[eval_key]["per_episode_success"], bool)
                     for r in recs])


def paired_compare(records: List[dict], key_a: Tuple, key_b: Tuple,
                   n_boot: int = 20000, seed: int = 0,
                   eval_key: str = "eval") -> dict:
    """Paired comparison of two condition cells on the shared evaluation seeds.

    The statistic is (mean success of B) - (mean success of A), averaged over
    seeds. The bootstrap resamples EPISODE INDICES (the shared axis), keeping
    the pairing intact, which removes the variance due to which initial
    conditions were drawn.
    """
    cells: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in records:
        cells[group_key(r)].append(r)
    A, B = cells.get(key_a), cells.get(key_b)
    if not A or not B:
        return {"error": f"missing cell {key_a if not A else key_b}"}

    SA, SB = _succ_matrix(A, eval_key), _succ_matrix(B, eval_key)      # (sA, n), (sB, n)
    n = min(SA.shape[1], SB.shape[1])
    SA, SB = SA[:, :n], SB[:, :n]
    obs = float(SB.mean() - SA.mean())

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = SB[:, idx].mean() - SA[:, idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # McNemar on the seed-pooled per-episode indicators (paired by episode).
    a = SA.mean(0) > 0.5
    b = SB.mean(0) > 0.5
    b01 = int(np.sum(~a & b))     # A fails, B succeeds
    b10 = int(np.sum(a & ~b))     # A succeeds, B fails
    from math import comb
    nn = b01 + b10
    if nn == 0:
        pval = 1.0
    else:
        k = min(b01, b10)
        tail = sum(comb(nn, i) for i in range(0, k + 1)) / (2 ** nn)
        pval = min(1.0, 2 * tail)

    return {
        "a": list(key_a), "b": list(key_b),
        "success_a": float(SA.mean()), "success_b": float(SB.mean()),
        "delta": obs, "ci95": [float(lo), float(hi)],
        "n_episodes_paired": int(n),
        "mcnemar_b01": b01, "mcnemar_b10": b10, "mcnemar_p": float(pval),
        "significant_at_05": bool(not (lo <= 0.0 <= hi)),
    }


def write_csv(rows: List[dict], path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = ["group", "algo", "n_demos", "noise", "n_seeds", "success_mean",
            "success_std", "success_pooled", "ci95_lo", "ci95_hi",
            "n_eval_per_seed", "n_eval_total", "min_dist_mean", "return_mean",
            "oob_frac", "demo_transitions", "n_params", "exec_mode"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def main_table_tex(rows: List[dict], teacher: Dict[float, dict],
                   path: str) -> None:
    """The table HW2 Part 4 requires, as a standalone LaTeX float.

    Generated as a complete `table` environment (never a bare \\input inside a
    tabular, which silently opens a TeX group and breaks column counting).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    order = {"quantity": 0, "noise": 1, "algorithm": 2}
    rows = sorted(rows, key=lambda r: (order.get(r["group"], 9), r["noise"], r["n_demos"]))
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Condition & Algo. & \#Demos & $\sigma$ & Teacher succ. & "
        r"Student succ. (95\% CI) & Eval. eps. \\", r"\midrule",
    ]
    last = None
    for r in rows:
        if last is not None and r["group"] != last:
            lines.append(r"\midrule")
        last = r["group"]
        t = teacher.get(float(r["noise"]))
        tstr = fmt_pct(t["success_rate"]) + r"\%" if t else "--"
        label = {"quantity": "Quantity", "noise": "Noisy teacher",
                 "algorithm": "Flow matching"}[r["group"]]
        lines.append(
            f"{label} & {r['algo'].upper()} & {r['n_demos']} & {r['noise']:g} & "
            f"{tstr} & {fmt_pct(r['success_pooled'])}\\% "
            f"[{fmt_pct(r['ci95_lo'])}, {fmt_pct(r['ci95_hi'])}] & "
            f"{r['n_eval_total']} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Teacher and student task success. Student success is pooled "
        r"over all training seeds (" +
        str(rows[0]["n_seeds"] if rows else 0) +
        r" seeds $\times$ " + str(rows[0]["n_eval_per_seed"] if rows else 0) +
        r" held-out initial conditions) with a Wilson 95\% interval. Teacher "
        r"success is measured on the same held-out initial conditions, at the "
        r"action-noise level used to collect that condition's data.}",
        r"\label{tab:main}", r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
