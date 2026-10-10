"""All report figures. Every figure is computed from the stored artifacts and
carries a provenance stamp naming the files it was built from.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

C_BASE, C_ALT, C_ACC = "#2E5C8A", "#C25E00", "#4A7C59"


def _stamp(fig, sources: List[str]) -> None:
    txt = "generated " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") \
          + " from " + ", ".join(os.path.basename(s) for s in sources[:4])
    if len(sources) > 4:
        txt += f" (+{len(sources)-4} more)"
    fig.text(0.005, 0.004, txt, fontsize=5, color="#777777", ha="left", va="bottom")


def _save(fig, path: str, sources: List[str]) -> str:
    _stamp(fig, sources)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
    return path


# ---------------------------------------------------------------- PPO teacher
def fig_ppo_curve(progress_path: str, out: str) -> Optional[str]:
    if not os.path.exists(progress_path):
        return None
    rows = json.load(open(progress_path))
    steps = [r["step"] for r in rows]
    key = "eval/episode_reward"
    rew = [r.get(key, np.nan) for r in rows]
    std = [r.get("eval/episode_reward_std", 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(steps, rew, "-o", ms=3, color=C_BASE)
    ax.fill_between(steps, np.array(rew) - np.array(std), np.array(rew) + np.array(std),
                    alpha=0.15, color=C_BASE, lw=0)
    ax.set_xlabel("environment steps"); ax.set_ylabel("episode reward")
    ax.set_title("PPO teacher learning curve (PandaPickCube)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    return _save(fig, out, [progress_path])


# ------------------------------------------------------------ demonstrations
def fig_demo_stats(demos_path: str, spec: dict, out: str,
                   n_traj: int = 25) -> str:
    from hw2.data import DemoSet
    from hw2.envs import decode_positions, EnvSpec

    d = DemoSet.load(demos_path)
    sp = EnvSpec(**{k: (tuple(v) if isinstance(v, list) else v)
                    for k, v in spec.items()})
    L = d.episode_lengths
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4))

    ax = axes[0, 0]
    ax.hist(L, bins=min(20, max(3, len(set(L.tolist())))), color=C_BASE, alpha=0.85)
    ax.set_xlabel("episode length (control steps)"); ax.set_ylabel("count")
    ax.set_title(f"Lengths: mean {L.mean():.1f} $\\pm$ {L.std():.1f}  (n={d.n_episodes})")

    ax = axes[0, 1]
    tr = min(n_traj, d.n_episodes)
    for i in range(tr):
        sl = d.episode_slice(i)
        pos = decode_positions(d.obs[sl], sp)
        dist = np.linalg.norm(pos["target"] - pos["box"], axis=-1)
        ax.plot(dist, color=C_BASE, alpha=0.3, lw=0.9)
    ax.axhline(d.meta.get("success_radius", 0.05), color=C_ALT, ls="--", lw=1.2,
               label=f"success radius {d.meta.get('success_radius', 0.05)} m")
    ax.set_xlabel("control step"); ax.set_ylabel("box$\\rightarrow$target distance (m)")
    ax.set_title(f"Box-to-target distance ({tr} demonstrations)"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    b0, t0 = [], []
    for i in range(d.n_episodes):
        o = d.obs[d.episode_starts[i]]
        p = decode_positions(o, sp)
        b0.append(p["box"]); t0.append(p["target"])
    b0, t0 = np.asarray(b0), np.asarray(t0)
    ax.scatter(b0[:, 0], b0[:, 1], s=16, color=C_BASE, label="initial box (x,y)")
    ax.scatter(t0[:, 0], t0[:, 1], s=16, color=C_ALT, marker="^", label="target (x,y)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(fontsize=7)
    ax.set_title("Coverage of initial conditions"); ax.set_aspect("equal", "datalim")

    ax = axes[1, 1]
    ax.boxplot([d.actions[:, j] for j in range(d.actions.shape[1])],
               showfliers=False, widths=0.6)
    ax.set_xlabel("action dimension"); ax.set_ylabel("value")
    ax.set_title("Executed action distribution")

    fig.suptitle(f"Demonstration dataset: {os.path.basename(demos_path)}  "
                 f"({d.n_episodes} episodes, {d.success.mean()*100:.0f}% success)",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, out, [demos_path])


def fig_demo_frames(frame_dir: str, out: str, n_frames: int = 6) -> Optional[str]:
    """Film-strip of expert demonstrations, read from the saved frame arrays."""
    packs = sorted(glob.glob(os.path.join(frame_dir, "expert_demo*_frames.npz")))
    if not packs:
        return None
    packs = packs[:3]
    fig, axes = plt.subplots(len(packs), n_frames,
                             figsize=(1.75 * n_frames, 1.5 * len(packs)),
                             squeeze=False)
    for r, pk in enumerate(packs):
        z = np.load(pk)
        frames, fidx, ntot = z["frames"], z["frame_index"], int(z["n_total"])
        sel = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
        for c, k in enumerate(sel):
            ax = axes[r][c]
            ax.imshow(frames[k]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"frame {int(fidx[k])}/{ntot}", fontsize=6)
        lbl = os.path.basename(pk).replace("expert_demo", "demo ").replace("_frames.npz", "")
        axes[r][0].set_ylabel(lbl, fontsize=6)
    fig.suptitle("Expert demonstrations (frames sampled uniformly in time)", fontsize=10)
    fig.tight_layout()
    return _save(fig, out, packs)


# --------------------------------------------------------------- loss curves
def fig_loss_curves(runs_dir: str, out: str, pattern: str = "act_*") -> Optional[str]:
    logs = sorted(glob.glob(os.path.join(runs_dir, pattern, "train_log.json")))
    if not logs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    cmap = plt.get_cmap("viridis")
    groups: Dict[str, List[str]] = {}
    for p in logs:
        name = os.path.basename(os.path.dirname(p))
        groups.setdefault(name.rsplit("_s", 1)[0], []).append(p)
    keys = sorted(groups)
    for i, k in enumerate(keys):
        col = cmap(i / max(len(keys) - 1, 1))
        for j, p in enumerate(sorted(groups[k])):
            h = json.load(open(p))["history"]
            st = [r["step"] for r in h]
            axes[0].plot(st, [r["loss"] for r in h], color=col, alpha=0.8, lw=1.0,
                         label=k if j == 0 else None)
            if "l1" in h[0]:
                axes[1].plot(st, [r["l1"] for r in h], color=col, alpha=0.8, lw=1.0,
                             label=k if j == 0 else None)
    axes[0].set_xlabel("gradient step"); axes[0].set_ylabel("total loss")
    axes[0].set_title("ACT training loss (all seeds)"); axes[0].set_yscale("log")
    axes[1].set_xlabel("gradient step"); axes[1].set_ylabel("action $L_1$")
    axes[1].set_title("Action reconstruction $L_1$"); axes[1].set_yscale("log")
    axes[1].legend(fontsize=6, ncol=1, loc="upper right")
    fig.tight_layout()
    return _save(fig, out, logs)


# ------------------------------------------------------------------- results
def _errbars(rows, xkey):
    x = [r[xkey] for r in rows]
    y = [r["success_pooled"] for r in rows]
    lo = [r["success_pooled"] - r["ci95_lo"] for r in rows]
    hi = [r["ci95_hi"] - r["success_pooled"] for r in rows]
    return np.asarray(x), np.asarray(y), np.vstack([lo, hi])


def fig_quantity(rows: List[dict], teacher: Dict[float, dict], out: str,
                 sources: List[str]) -> Optional[str]:
    rs = sorted([r for r in rows if r["group"] == "quantity"],
                key=lambda r: r["n_demos"])
    if not rs:
        return None
    x, y, err = _errbars(rs, "n_demos")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.errorbar(x, 100 * y, yerr=100 * err, fmt="-o", color=C_BASE, capsize=3,
                label="ACT student (pooled, 95% CI)")
    for r in rs:
        ax.scatter([r["n_demos"]] * len(r["success_per_seed"]),
                   100 * np.asarray(r["success_per_seed"]), s=12,
                   color=C_BASE, alpha=0.45, zorder=3)
    t = teacher.get(0.0)
    if t:
        ax.axhline(100 * t["success_rate"], color=C_ALT, ls="--", lw=1.3,
                   label=f"PPO teacher ({100*t['success_rate']:.0f}%)")
    ax.set_xlabel("number of demonstrations"); ax.set_ylabel("task success (%)")
    ax.set_xticks(x); ax.set_ylim(-3, 103)
    ax.set_title("Effect of demonstration quantity")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, out, sources)


def fig_noise(rows: List[dict], teacher: Dict[float, dict], out: str,
              sources: List[str]) -> Optional[str]:
    rs = sorted([r for r in rows
                 if (r["group"] == "noise") or
                    (r["group"] == "quantity" and r["n_demos"] == 100)],
                key=lambda r: r["noise"])
    if len(rs) < 2:
        return None
    x, y, err = _errbars(rs, "noise")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.errorbar(x, 100 * y, yerr=100 * err, fmt="-o", color=C_BASE, capsize=3,
                label="ACT student (pooled, 95% CI)")
    for r in rs:
        ax.scatter([r["noise"]] * len(r["success_per_seed"]),
                   100 * np.asarray(r["success_per_seed"]), s=12,
                   color=C_BASE, alpha=0.45, zorder=3)
    tx = sorted(teacher)
    ax.plot(tx, [100 * teacher[k]["success_rate"] for k in tx], "--^",
            color=C_ALT, lw=1.3, ms=4, label="teacher at same noise")
    ax.set_xlabel("teacher action noise $\\sigma$"); ax.set_ylabel("task success (%)")
    ax.set_ylim(-3, 103); ax.set_xticks(x)
    ax.set_title("Effect of demonstration quality (teacher action noise)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return _save(fig, out, sources)


def fig_algorithms(rows: List[dict], out: str, sources: List[str]) -> Optional[str]:
    rs = [r for r in rows if r["n_demos"] == 100 and r["noise"] == 0.0]
    if len(rs) < 2:
        return None
    rs = sorted(rs, key=lambda r: r["algo"])
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    names = [r["algo"].upper() for r in rs]
    y = np.array([100 * r["success_pooled"] for r in rs])
    lo = y - np.array([100 * r["ci95_lo"] for r in rs])
    hi = np.array([100 * r["ci95_hi"] for r in rs]) - y
    ax.bar(names, y, yerr=np.vstack([lo, hi]), capsize=4,
           color=[C_BASE, C_ACC][: len(rs)], alpha=0.9, width=0.55)
    for i, r in enumerate(rs):
        ax.scatter([i] * len(r["success_per_seed"]),
                   100 * np.asarray(r["success_per_seed"]), s=14, color="k", alpha=0.5)
    ax.set_ylabel("task success (%)"); ax.set_ylim(0, 103)
    ax.set_title("ACT vs flow matching (100 clean demos)")
    fig.tight_layout()
    return _save(fig, out, sources)


def fig_failure(fail_json: str, out: str) -> Optional[str]:
    """Failure-mode analysis from the stored per-episode diagnostics."""
    if not os.path.exists(fail_json):
        return None
    d = json.load(open(fail_json))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

    ax = axes[0]
    for lbl, key, col in (("success", "dist_success", C_ACC),
                          ("failure", "dist_failure", C_ALT)):
        arr = np.asarray(d.get(key, []), dtype=float)
        if arr.size:
            m = arr.mean(0)
            ax.plot(m, color=col, label=f"{lbl} (n={arr.shape[0]})")
            ax.fill_between(np.arange(len(m)), np.percentile(arr, 25, 0),
                            np.percentile(arr, 75, 0), color=col, alpha=0.18, lw=0)
    ax.axhline(d.get("success_radius", 0.05), color="k", ls="--", lw=1)
    ax.set_xlabel("control step"); ax.set_ylabel("box$\\rightarrow$target distance (m)")
    ax.set_title("Distance traces by outcome"); ax.legend(fontsize=7)

    ax = axes[1]
    md = np.asarray(d.get("min_dist_all", []), dtype=float)
    if md.size:
        ax.hist(md, bins=24, color=C_BASE, alpha=0.85)
        ax.axvline(d.get("success_radius", 0.05), color=C_ALT, ls="--", lw=1.3,
                   label="success radius")
        ax.legend(fontsize=7)
    ax.set_xlabel("closest box$\\rightarrow$target distance (m)"); ax.set_ylabel("episodes")
    ax.set_title("How close did it get?")

    ax = axes[2]
    cats = d.get("failure_taxonomy", {})
    if cats:
        k = list(cats); v = [cats[x] for x in k]
        ax.barh(k, v, color=C_ALT, alpha=0.9)
        ax.set_xlabel("episodes"); ax.set_title("Failure taxonomy")
        ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return _save(fig, out, [fail_json])
