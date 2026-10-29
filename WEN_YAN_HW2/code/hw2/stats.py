"""Statistics for repeated evaluation on SHARED initial conditions.

Each condition has 3 trained models (training seeds 0, 1, 2) and every model
of every condition is evaluated on the same N initial conditions (ICs). The
3 x N outcomes are therefore NOT 3N independent trials: outcomes on the same
IC are correlated across seeds (some ICs are simply hard), and the 3 seeds are
a tiny sample of the training-seed distribution. We report:

  mean      the average of the 3 per-seed success rates (= pooled rate here,
            because every seed has the same N);
  ci_ic     95% percentile interval from resampling the N ICs with the models
            held fixed -- uncertainty about the IC distribution, CONDITIONAL on
            these trained models;
  ci_2way   95% interval from resampling training seeds within the condition
            AND ICs -- also reflects seed-to-seed variation, but with only 3
            seeds it is crude (a resample can contain one seed three times);
  seed_sd   the sample SD of the 3 per-seed rates.

Paired contrasts between conditions use the same IC resample for both arms.
`p_signflip` is an exact-in-distribution sign-flip test of the per-IC
difference of seed-averaged success (conditional on the models). The
majority-vote McNemar test compares per-IC majority outcomes (>= 2 of 3 seeds
succeed); it tests a DIFFERENT statistic from the mean-rate difference and is
reported only for continuity with the historical report.
"""
from __future__ import annotations

from math import comb
from typing import Dict, List

import numpy as np

N_BOOT = 20000


def _pct(x, q=(2.5, 97.5)):
    lo, hi = np.percentile(x, q)
    return [float(lo), float(hi)]


def cell_summary(S: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> Dict:
    """S: (n_seeds, n_ic) 0/1 outcomes on shared ICs."""
    S = np.asarray(S, dtype=float)
    k, n = S.shape
    rng = np.random.default_rng(seed)
    ic = rng.integers(0, n, (n_boot, n))
    b_ic = S[:, ic].mean(axis=(0, 2))                       # (n_boot,)
    sd = rng.integers(0, k, (n_boot, k))
    b_2 = np.array([S[sd[i]][:, ic[i]].mean() for i in range(n_boot)])
    per_seed = S.mean(1)
    return {"n_seeds": int(k), "n_ic": int(n),
            "per_seed": per_seed.tolist(),
            "mean": float(per_seed.mean()),
            "seed_sd": float(per_seed.std(ddof=1)) if k > 1 else 0.0,
            "ci_ic": _pct(b_ic), "ci_2way": _pct(b_2),
            "n_success_total": int(S.sum()), "n_episodes_total": int(S.size)}


def mcnemar_exact(b01: int, b10: int) -> float:
    nn = b01 + b10
    if nn == 0:
        return 1.0
    k = min(b01, b10)
    return float(min(1.0, 2 * sum(comb(nn, i) for i in range(k + 1)) / 2 ** nn))


def paired_contrast(SA: np.ndarray, SB: np.ndarray, n_boot: int = N_BOOT,
                    n_flip: int = 100000, seed: int = 0) -> Dict:
    """Contrast B - A of mean success on the same ICs (columns aligned)."""
    SA, SB = np.asarray(SA, float), np.asarray(SB, float)
    assert SA.shape[1] == SB.shape[1], "conditions must share the ICs"
    kA, n = SA.shape
    kB = SB.shape[0]
    rng = np.random.default_rng(seed)
    delta = float(SB.mean() - SA.mean())
    ic = rng.integers(0, n, (n_boot, n))
    d_ic = SB[:, ic].mean(axis=(0, 2)) - SA[:, ic].mean(axis=(0, 2))
    sa = rng.integers(0, kA, (n_boot, kA))
    sb = rng.integers(0, kB, (n_boot, kB))
    d_2 = np.array([SB[sb[i]][:, ic[i]].mean() - SA[sa[i]][:, ic[i]].mean()
                    for i in range(n_boot)])
    # sign-flip test on per-IC differences of seed-averaged success
    d = SB.mean(0) - SA.mean(0)
    flips = rng.choice([-1.0, 1.0], size=(n_flip, n))
    null = (flips * d).mean(1)
    p_flip = float((np.sum(np.abs(null) >= abs(d.mean()) - 1e-12) + 1) / (n_flip + 1))
    a_maj, b_maj = SA.mean(0) > 0.5, SB.mean(0) > 0.5
    b01, b10 = int(np.sum(~a_maj & b_maj)), int(np.sum(a_maj & ~b_maj))
    per_seed_index = None
    if kA == kB:
        per_seed_index = (SB.mean(1) - SA.mean(1)).tolist()
    ci_ic = _pct(d_ic)
    return {"delta": delta, "ci_ic": ci_ic, "ci_2way": _pct(d_2),
            "p_signflip": p_flip,
            "ic_ci_excludes_zero": bool(ci_ic[0] > 0 or ci_ic[1] < 0),
            "per_seed_index_delta": per_seed_index,
            "n_ic": int(n), "n_seeds": [int(kA), int(kB)],
            "mcnemar_majority_b01": b01, "mcnemar_majority_b10": b10,
            "mcnemar_majority_p": mcnemar_exact(b01, b10),
            "mean_a": float(SA.mean()), "mean_b": float(SB.mean())}
