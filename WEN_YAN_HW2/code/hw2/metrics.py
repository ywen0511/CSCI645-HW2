"""Termination-aware episode metrics (evaluation protocol v2).

Every rollout in this project (teacher evaluation, demonstration collection,
student evaluation, diagnostics, videos) produces the same raw arrays for a
batch of B episodes with horizon T control steps:

    dist      (B, T+1)  box->target distance of state k; k = 0 is the reset state
    done      (B, T)    raw environment `done` of transition t (state t -> t+1)
    nonfinite (B, T+1)  state k contains NaN/inf (optional)
    rewards   (B, T)    reward of transition t (optional)

The raw MuJoCo Playground env has no auto-reset: after `done` fires, the batch
keeps stepping a state that is out of bounds or NaN. Protocol v1 (the
historical submission) scored success over the whole distance trace, including
those post-termination states. Protocol v2 fixes one convention and applies it
everywhere:

* The horizon limit (T steps) is NOT termination; termination means the env
  raised `done` (box out of bounds, or NaN in the physics state).
* Let t* be the first transition with done = 1. Its resulting state k = t*+1 is
  the TERMINAL state. The terminal state and every later state are invalid:
  they are excluded from success, distances and all diagnostics. A non-finite
  state is invalid even if `done` did not fire, and ends the valid prefix.
* Rewards: the reward of the terminal transition t* is counted (the agent
  earned it on a valid transition) unless it is non-finite; rewards of every
  later transition are excluded.
* Success: the box is within `radius` of the target for `hold` consecutive
  VALID states. Valid states form a prefix of each episode, so a hold streak
  cannot cross termination, an invalid state, or another episode (episodes are
  separate rows and are never concatenated).
* "Ever achieved the hold" is the criterion; leaving the target radius later
  does not erase a success. Such later violations are reported separately
  (`left_after_success`, `final_inside`).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

PROTOCOL_VERSION = "v2-termination-aware"

TERM_HORIZON = "horizon"
TERM_OOB = "out_of_bounds"
TERM_NAN = "nonfinite"


def valid_lengths(done: np.ndarray, nonfinite: Optional[np.ndarray] = None,
                  T: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Number of valid states per episode and the cause of the episode end.

    k_done = (first t with done) + 1   index of the terminal state
    k_nf   = first non-finite state index
    k_end  = min(k_done, k_nf, T+1);   states 0 .. k_end-1 are valid

    Returns
      n_valid        (B,) states 0 .. n_valid-1 are valid (0 <= n_valid <= T+1)
      n_transitions  (B,) transitions whose reward counts: k_end if the
                     episode terminated (incl. the terminal transition), else T
      t_term         (B,) the terminating transition k_end-1, or -1 (horizon)
      cause          (B,) horizon / out_of_bounds / nonfinite
    """
    done = np.asarray(done).astype(bool)
    B, Tn = done.shape
    T = Tn if T is None else int(T)
    big = T + 1
    k_done = np.where(done.any(1), done.argmax(1) + 1, big)
    if nonfinite is not None:
        nf = np.asarray(nonfinite).astype(bool)
        k_nf = np.where(nf.any(1), nf.argmax(1), big)
    else:
        k_nf = np.full(B, big)
    k_end = np.minimum(np.minimum(k_done, k_nf), big).astype(np.int64)
    term = k_end < big
    cause = np.where(~term, TERM_HORIZON,
                     np.where(k_nf <= k_end, TERM_NAN, TERM_OOB))
    return {"n_valid": k_end,
            "n_transitions": np.where(term, k_end, T).astype(np.int64),
            "t_term": np.where(term, k_end - 1, -1).astype(np.int64),
            "cause": cause.astype(str)}


def valid_mask(n_valid: np.ndarray, T: int) -> np.ndarray:
    """(B, T+1) boolean mask of valid states from per-episode valid lengths."""
    k = np.arange(T + 1)[None, :]
    return k < np.asarray(n_valid)[:, None]


def longest_run(x: np.ndarray) -> np.ndarray:
    """Longest run of True per row of a (B, L) boolean array."""
    x = np.asarray(x).astype(bool)
    B, L = x.shape
    best = np.zeros(B, dtype=np.int64)
    cur = np.zeros(B, dtype=np.int64)
    for k in range(L):
        cur = np.where(x[:, k], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def first_hold_completion(inside: np.ndarray, hold: int) -> np.ndarray:
    """Index of the state at which the first `hold`-long run completes, or -1."""
    inside = np.asarray(inside).astype(bool)
    B, L = inside.shape
    out = np.full(B, -1, dtype=np.int64)
    cur = np.zeros(B, dtype=np.int64)
    for k in range(L):
        cur = np.where(inside[:, k], cur + 1, 0)
        hit = (cur >= hold) & (out < 0)
        out[hit] = k
    return out


def episode_metrics(dist: np.ndarray, done: np.ndarray,
                    radius: float, hold: int,
                    nonfinite: Optional[np.ndarray] = None,
                    rewards: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-episode metrics under protocol v2. All outputs have leading dim B."""
    dist = np.asarray(dist, dtype=np.float64)
    if dist.ndim == 1:
        dist = dist[None]
    done = np.asarray(done)
    if done.ndim == 1:
        done = done[None]
    B, T1 = dist.shape
    T = T1 - 1
    if done.shape != (B, T):
        raise ValueError(f"done has shape {done.shape}, expected {(B, T)}")
    if nonfinite is None:
        nonfinite = ~np.isfinite(dist)
    else:
        nonfinite = np.asarray(nonfinite).astype(bool) | ~np.isfinite(dist)
    vl = valid_lengths(done, nonfinite, T)
    mask = valid_mask(vl["n_valid"], T)

    inside = mask & (np.where(np.isfinite(dist), dist, np.inf) < radius)
    run = longest_run(inside)
    success = run >= hold
    first = first_hold_completion(inside, hold)

    dmask = np.where(mask, dist, np.inf)
    min_dist = dmask.min(1)
    has_valid = mask.any(1)
    min_dist = np.where(has_valid, min_dist, np.nan)
    last = np.maximum(vl["n_valid"] - 1, 0)
    final_dist = np.where(has_valid, dist[np.arange(B), last], np.nan)
    final_inside = has_valid & (final_dist < radius)
    left_after = np.zeros(B, dtype=bool)
    for b in np.flatnonzero(success):
        after = np.arange(T1) > first[b]
        left_after[b] = bool(np.any(after & mask[b] & ~inside[b]))

    out = {
        "success": success,
        "n_valid_states": vl["n_valid"],
        "t_term": vl["t_term"],
        "term_cause": vl["cause"],
        "terminated": vl["cause"] != TERM_HORIZON,
        "longest_hold": run,
        "first_success_state": first,
        "min_dist": min_dist,
        "final_dist": final_dist,
        "final_inside": final_inside,
        "left_after_success": left_after,
    }
    if rewards is not None:
        r = np.asarray(rewards, dtype=np.float64)
        n_tr = vl["n_transitions"]
        tmask = np.arange(T)[None, :] < n_tr[:, None]
        rfin = np.isfinite(r)
        out["return"] = np.where(tmask & rfin, r, 0.0).sum(1)
        out["nonfinite_reward_in_window"] = (tmask & ~rfin).any(1)
    return out


def masked_min(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise min over valid entries; NaN for a row with no valid entry."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask).astype(bool) & np.isfinite(x)
    v = np.where(m, x, np.inf).min(1)
    return np.where(m.any(1), v, np.nan)


def masked_any(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (np.asarray(x).astype(bool) & np.asarray(mask).astype(bool)).any(1)


def legacy_success(dist: np.ndarray, radius: float, hold: int) -> np.ndarray:
    """Protocol v1 success: hold over the FULL trace, ignoring termination.

    Kept only to count how many outcomes the v2 fix changes.
    """
    dist = np.asarray(dist, dtype=np.float64)
    inside = np.where(np.isfinite(dist), dist, np.inf) < radius
    return longest_run(inside) >= hold


# --------------------------------------------------------------------------
# failure taxonomy
# --------------------------------------------------------------------------
# Thresholds are validated against teacher and student trajectories and the
# rendered videos (scripts/stage2b_failure.py writes the validation record).
PROXIMITY_M = 0.012     # the env's own `reached_box` radius (gripper site->box)
LIFT_M = 0.02           # box raised at least 2 cm above its initial height

FAILURE_ORDER = [
    "terminated early (box out of bounds / non-finite)",
    "never touched the box",
    "touched the box, never lifted it",
    "lifted the box, never within target radius",
    "within target radius, held < required steps",
]


def classify_episode(success: bool, terminated: bool, any_contact: bool,
                     lifted: bool, min_dist: float, radius: float) -> str:
    """Mutually exclusive outcome label with a fixed decision order.

    Successful episodes are classified FIRST and never receive a failure label.
    Every quantity must already be restricted to the episode's valid states.
    """
    if success:
        return "success"
    if terminated:
        return FAILURE_ORDER[0]
    if not any_contact:
        return FAILURE_ORDER[1]
    if not lifted:
        return FAILURE_ORDER[2]
    if not (min_dist < radius):
        return FAILURE_ORDER[3]
    return FAILURE_ORDER[4]


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = float(z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d)
    return (float(p), float(max(0.0, c - h)), float(min(1.0, c + h)))
