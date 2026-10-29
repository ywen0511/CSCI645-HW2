"""Regression tests for the termination-aware metrics (protocol v2).

Synthetic trajectories only: these run on CPU in a second and need neither
MuJoCo nor a GPU.   python -m pytest tests/ -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from hw2 import metrics as M  # noqa: E402

R, H, T = 0.05, 10, 30          # radius, hold, horizon (short, for readability)


def traj(inside_states, T=T, done_at=None, nan_at=None):
    """One episode: distance 0.01 at the given state indices, else 0.3."""
    d = np.full(T + 1, 0.3)
    d[list(inside_states)] = 0.01
    done = np.zeros(T, bool)
    if done_at is not None:
        done[done_at:] = True          # raw done may stay up after termination
    if nan_at is not None:
        d[nan_at:] = np.nan
    return d, done


def run(d, done, rewards=None):
    return M.episode_metrics(d[None], done[None], R, H,
                             rewards=None if rewards is None else rewards[None])


def test_normal_success():
    d, done = traj(range(5, 20))
    m = run(d, done)
    assert m["success"][0]
    assert m["term_cause"][0] == M.TERM_HORIZON
    assert m["n_valid_states"][0] == T + 1
    assert m["first_success_state"][0] == 5 + H - 1


def test_insufficient_hold():
    d, done = traj(range(5, 5 + H - 1))        # 9 consecutive steps
    assert not run(d, done)["success"][0]
    # two 9-step runs separated by one outside step never add up to 10
    d, done = traj(list(range(2, 11)) + list(range(12, 21)))
    assert not run(d, done)["success"][0]


def test_success_only_after_termination_does_not_count():
    # terminates on transition 8 (state 9 is terminal); "inside" only afterwards
    d, done = traj(range(12, 28), done_at=8)
    m = run(d, done)
    assert not m["success"][0]
    assert m["terminated"][0] and m["term_cause"][0] == M.TERM_OOB
    assert m["n_valid_states"][0] == 9
    # the v1 metric would have counted it: this is the bug being fixed
    assert M.legacy_success(d[None], R, H)[0]


def test_valid_success_before_later_termination_counts():
    d, done = traj(range(3, 16), done_at=20)
    m = run(d, done)
    assert m["success"][0]
    assert m["terminated"][0]
    assert m["left_after_success"][0]          # states 16..20 are outside


def test_hold_cannot_straddle_termination():
    # 6 inside before the terminal state + 10 inside after it: not a success
    d, done = traj(range(3, 25), done_at=8)
    assert not run(d, done)["success"][0]


def test_nan_termination():
    d, done = traj(range(2, 9), nan_at=9)
    done[8:] = True                             # env raises done on NaN
    m = run(d, done)
    assert not m["success"][0]
    assert m["term_cause"][0] == M.TERM_NAN
    assert np.isfinite(m["min_dist"][0])
    # NaN without done still ends the valid prefix and is never "inside"
    d2, done2 = traj(range(2, 30), nan_at=6)
    m2 = run(d2, done2)
    assert not m2["success"][0]
    assert m2["n_valid_states"][0] == 6
    assert m2["term_cause"][0] == M.TERM_NAN


def test_no_cross_episode_accumulation():
    # episode A ends inside for 6 steps, episode B starts inside for 6 steps
    dA, doneA = traj(range(T - 5, T + 1))
    dB, doneB = traj(range(0, 6))
    m = M.episode_metrics(np.stack([dA, dB]), np.stack([doneA, doneB]), R, H)
    assert not m["success"].any()
    # and the concatenation trap the row layout prevents:
    assert M.longest_run(np.concatenate([dA, dB])[None] < R)[0] == 12


def test_returns_exclude_post_termination_rewards():
    d, done = traj([], done_at=4)
    r = np.ones(T)
    r[4] = np.nan                               # terminal transition reward NaN
    m = run(d, done, r)
    assert m["return"][0] == 4.0                # transitions 0..3 only
    assert m["nonfinite_reward_in_window"][0]
    d, done = traj([], done_at=4)
    m = run(d, done, np.ones(T))
    assert m["return"][0] == 5.0                # terminal transition 4 counts


def test_classifier_success_first_and_exclusive():
    assert M.classify_episode(True, True, False, False, 1.0, R) == "success"
    labels = {
        M.classify_episode(False, True, True, True, 0.0, R),
        M.classify_episode(False, False, False, False, 0.5, R),
        M.classify_episode(False, False, True, False, 0.5, R),
        M.classify_episode(False, False, True, True, 0.5, R),
        M.classify_episode(False, False, True, True, 0.01, R),
    }
    assert labels == set(M.FAILURE_ORDER)


def test_scoring_ignores_degenerate_reset_state():
    """MJX returns the reset state before kinematics: positions read as 0.

    Lift must be measured from the first stepped state, and the reset state's
    zero gripper-box distance must not count as 'within 12 mm'.
    """
    from hw2.scoring import score
    T = 20
    z = np.full((1, T + 1), 0.025); z[0, 0] = 0.0          # box resting, reset=0
    grip = np.full((1, T + 1), 0.10); grip[0, 0] = 0.0     # never near, reset=0
    raw = {"dist": np.full((1, T + 1), 0.3), "dist_obs": np.full((1, T + 1), 0.3),
           "done": np.zeros((1, T), bool), "nonfinite": np.zeros((1, T + 1), bool),
           "rewards": np.zeros((1, T)), "box_z": z, "grip": grip,
           "contact_l": np.ones((1, T + 1), bool), "contact_r": np.zeros((1, T + 1), bool)}
    ep = score(raw, R, H)
    assert abs(ep["max_lift"][0]) < 1e-12 and not ep["lifted"][0]
    assert not ep["within_12mm"][0] and abs(ep["min_grip"][0] - 0.10) < 1e-12
    assert ep["label"][0] == "touched the box, never lifted it"
    # negative control: the old reference (k = 0) would have called it lifted
    assert (z[0, 1:] - z[0, 0]).max() >= 0.02
