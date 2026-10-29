"""Shared-initial-condition statistics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from hw2 import stats as ST  # noqa: E402


def test_identical_conditions_give_zero_contrast():
    rng = np.random.default_rng(1)
    S = (rng.random((3, 100)) < 0.3).astype(float)
    c = ST.paired_contrast(S, S, n_boot=500, n_flip=2000)
    assert c["delta"] == 0 and c["ci_ic"] == [0.0, 0.0]
    assert c["p_signflip"] == 1.0


def test_clear_effect_is_detected_and_two_way_is_wider():
    rng = np.random.default_rng(2)
    hard = rng.random(100) < 0.5                     # shared IC difficulty
    A = np.stack([(rng.random(100) < 0.2) & ~hard for _ in range(3)]).astype(float)
    B = np.stack([(rng.random(100) < 0.8) & ~hard for _ in range(3)]).astype(float)
    c = ST.paired_contrast(A, B, n_boot=2000, n_flip=5000)
    assert c["delta"] > 0.2 and c["ic_ci_excludes_zero"]
    assert c["p_signflip"] < 0.001
    s = ST.cell_summary(B, n_boot=2000)
    assert s["ci_2way"][1] - s["ci_2way"][0] >= s["ci_ic"][1] - s["ci_ic"][0] - 1e-9


def test_mcnemar_exact():
    assert ST.mcnemar_exact(0, 0) == 1.0
    assert abs(ST.mcnemar_exact(0, 10) - 2 / 2 ** 10) < 1e-12
