"""Temporal-ensembling and replanning action selection."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from hw2.inference import ActionSelector  # noqa: E402

H, A, T = 30, 1, 60


def _drive(sel, steps=40):
    # every chunk is constant and equal to (prediction time)/100
    for t in range(steps):
        chunk = np.full((1, H, A), t / 100.0, np.float32) if sel.needs_prediction(t) else None
        a = sel.act(t, chunk)
    return float(a[0, 0])


def test_newest_vs_oldest_ordering():
    newest = _drive(ActionSelector(1, H, A, T, "ensemble", ensemble_m=0.5))
    oldest = _drive(ActionSelector(1, H, A, T, "ensemble", ensemble_m=0.5,
                                   ensemble_order="oldest"))
    # at t=39 the covering predictions were made at t=10..39
    assert 0.35 < newest <= 0.39       # dominated by the newest (0.39)
    assert 0.10 <= oldest < 0.14       # dominated by the oldest (0.10)


def test_uniform_when_m_zero():
    a = _drive(ActionSelector(1, H, A, T, "ensemble", ensemble_m=0.0))
    assert abs(a - np.mean(np.arange(10, 40) / 100.0)) < 1e-6


def test_replan_every_step_uses_first_action():
    sel = ActionSelector(1, H, A, T, "chunk", exec_horizon=1)
    for t in range(5):
        chunk = np.arange(H, dtype=np.float32).reshape(1, H, 1) / 100 + t
        assert sel.needs_prediction(t)
        assert sel.act(t, chunk)[0, 0] == np.clip(t, -1, 1)


def test_open_loop_chunk_and_clipping():
    sel = ActionSelector(1, H, A, T, "chunk")
    chunk = np.linspace(-2, 2, H, dtype=np.float32).reshape(1, H, 1)
    acts = []
    for t in range(H):
        acts.append(sel.act(t, chunk if sel.needs_prediction(t) else None)[0, 0])
    assert not sel.needs_prediction(1) and sel.needs_prediction(H)
    assert min(acts) == -1.0 and max(acts) == 1.0
