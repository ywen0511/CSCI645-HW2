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
def fig_ppo_curve(progress_path: str, out: str,
                  historical_path: Optional[str] = None) -> Optional[str]:
    if not os.path.exists(progress_path):
        return None
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    srcs = [progress_path]
    for path, lbl, col, ls in ((progress_path, "retrained teacher (this run)", C_BASE, "-"),
                               (historical_path, "original teacher (archived log)", C_ALT, "--")):
        if not path or not os.path.exists(path):
            continue
        rows = json.load(open(path))
        st = np.array([r["step"] for r in rows])
        rw = np.array([r.get("eval/episode_reward", np.nan) for r in rows])
        sd = np.array([r.get("eval/episode_reward_std", 0.0) for r in rows])
        ax.plot(st, rw, ls, marker="o", ms=3, color=col, label=lbl)
        ax.fill_between(st, rw - sd, rw + sd, alpha=0.12, color=col, lw=0)
        if path != progress_path:
            srcs.append(path)
    ax.set_xlabel("environment steps"); ax.set_ylabel("PPO eval episode reward")
    ax.set_title("PPO teacher learning curve (PandaPickCube)")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, out, srcs)


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
    # state 0 of every episode is the reset observation, returned before MJX
    # computes body positions (box/gripper read as 0): plot from state 1.
    for i in range(tr):
        sl = d.episode_slice(i)
        pos = decode_positions(d.obs[sl][1:], sp)
        dist = np.linalg.norm(pos["target"] - pos["box"], axis=-1)
        ax.plot(dist, color=C_BASE, alpha=0.3, lw=0.9)
    ax.axhline(d.meta.get("success_radius", 0.05), color=C_ALT, ls="--", lw=1.2,
               label=f"success radius {d.meta.get('success_radius', 0.05)} m")
    ax.set_xlabel("control step (from 1)"); ax.set_ylabel("box$\\rightarrow$target distance (m)")
    ax.set_title(f"Box-to-target distance ({tr} demonstrations)"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    b0, t0 = [], []
    for i in range(d.n_episodes):
        o = d.obs[d.episode_starts[i] + 1]
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


def fig_demo_frames(videos: List[str], out: str, n_frames: int = 6) -> Optional[str]:
    """Film-strip of rollout videos, decoded from the mp4 files."""
    import imageio.v2 as iio
    videos = [v for v in videos if os.path.exists(v)][:4]
    if not videos:
        return None
    fig, axes = plt.subplots(len(videos), n_frames,
                             figsize=(1.75 * n_frames, 1.45 * len(videos)), squeeze=False)
    for r, v in enumerate(videos):
        frames = [f for f in iio.mimread(v, memtest=False)]
        sel = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        for c, k in enumerate(sel):
            ax = axes[r][c]
            ax.imshow(frames[k]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"t={2*k}", fontsize=6)
        lbl = os.path.basename(v).replace(".mp4", "")
        lbl = lbl.replace("student_", "").replace("_replan1", "\nreplan").replace("expert_", "")
        axes[r][0].set_ylabel(lbl[:34], fontsize=5)
    fig.tight_layout()
    return _save(fig, out, videos)


# --------------------------------------------------------------- loss curves
def fig_loss_curves(logs: List[str], out: str) -> Optional[str]:
    logs = sorted(logs)
    if not logs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    cmap = plt.get_cmap("viridis")
    groups: Dict[str, List[str]] = {}
    for p in logs:
        name = os.path.basename(p).replace("trainlog_", "").replace(".json", "")
        groups.setdefault(name.rsplit("_s", 1)[0], []).append(p)
    keys = sorted(groups)
    for i, k in enumerate(keys):
        col = cmap(i / max(len(keys) - 1, 1))
        for j, p in enumerate(sorted(groups[k])):
            h = json.load(open(p))["history"]
            st = [r["step"] for r in h]
            axes[0].plot(st, [r["loss"] for r in h], color=col, alpha=0.8, lw=0.9,
                         label=k if j == 0 else None)
            key = "l1" if "l1" in h[0] else ("mse" if "mse" in h[0] else None)
            if key:
                axes[1].plot(st, [r[key] for r in h], color=col, alpha=0.8, lw=0.9,
                             label=k if j == 0 else None)
    axes[0].set_xlabel("gradient step"); axes[0].set_ylabel("training loss")
    axes[0].set_title("Training loss, all 21 models (3 seeds each)"); axes[0].set_yscale("log")
    axes[1].set_xlabel("gradient step"); axes[1].set_ylabel("ACT action $L_1$ / flow MSE")
    axes[1].set_title("Reconstruction term"); axes[1].set_yscale("log")
    axes[1].legend(fontsize=6, ncol=2, loc="upper right")
    fig.tight_layout()
    return _save(fig, out, logs)


# ------------------------------------------------------------------- results
MODE_STYLE = {"ensemble": (C_BASE, "o", "temporal ensembling"),
              "replan1": (C_ACC, "s", "replan every step")}


def _series(ax, cells, keys, xs, mode, offset=0.0):
    col, mk, lbl = MODE_STYLE[mode]
    y = np.array([100 * cells[k]["mean"] for k in keys])
    lo = np.array([100 * cells[k]["ci_ic"][0] for k in keys])
    hi = np.array([100 * cells[k]["ci_ic"][1] for k in keys])
    x = np.asarray(xs, float) + offset
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="-" + mk, color=col,
                capsize=3, ms=4, label=f"{lbl} (mean of 3 seeds, IC 95% CI)")
    for xi, k in zip(x, keys):
        ax.scatter([xi] * len(cells[k]["per_seed"]), 100 * np.asarray(cells[k]["per_seed"]),
                   s=11, color=col, alpha=0.45, zorder=3)


def fig_quantity(cells: Dict[str, Dict[str, dict]], teacher_rate: float, out: str,
                 sources: List[str]) -> Optional[str]:
    keys = ["act_clean_n25", "act_clean_n50", "act_clean_n100"]
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for i, mode in enumerate(("ensemble", "replan1")):
        _series(ax, cells[mode], keys, np.log2([25, 50, 100]), mode, (i - 0.5) * 0.06)
    ax.axhline(100 * teacher_rate, color=C_ALT, ls="--", lw=1.2,
               label=f"retrained PPO teacher ({100*teacher_rate:.0f}%)")
    ax.set_xticks(np.log2([25, 50, 100])); ax.set_xticklabels(["25", "50", "100"])
    ax.set_xlabel("number of clean demonstrations (nested subsets)")
    ax.set_ylabel("task success (%)"); ax.set_ylim(-3, 103)
    ax.set_title("Option A: demonstration quantity (ACT)")
    ax.legend(fontsize=6, loc="center right")
    fig.tight_layout()
    return _save(fig, out, sources)


def fig_noise(cells, teacher: Dict[float, float], teacher_hist: Dict[float, float],
              out: str, sources: List[str]) -> Optional[str]:
    keys = ["act_clean_n100", "act_noise0.5_n100", "act_noise1_n100", "act_noise2_n100"]
    xs = np.arange(4)
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for i, mode in enumerate(("ensemble", "replan1")):
        _series(ax, cells[mode], keys, xs, mode, (i - 0.5) * 0.08)
    sig = [0.0, 0.5, 1.0, 2.0]
    ax.plot(xs, [100 * teacher[s] for s in sig], "--^", color=C_ALT, ms=4, lw=1.2,
            label="retrained teacher at same $\\sigma$")
    if teacher_hist:
        hx = [i for i, s in enumerate(sig) if s in teacher_hist]
        ax.plot(hx, [100 * teacher_hist[sig[i]] for i in hx], "x", color="k", ms=5,
                label="original teacher (archived), = data-collection teacher")
    ax.set_xticks(xs); ax.set_xticklabels(["0 (clean,\nfiltered)", "0.5", "1", "2"])
    ax.set_xlabel("teacher action-noise $\\sigma$ of the 100 demonstrations")
    ax.set_ylabel("task success (%)"); ax.set_ylim(-3, 103)
    ax.set_title("Option B: demonstration quality (ACT)")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    return _save(fig, out, sources)


def fig_algorithms(cells, out: str, sources: List[str]) -> Optional[str]:
    keys = ["act_clean_n100", "flow_clean_n100"]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    w = 0.36
    for i, mode in enumerate(("ensemble", "replan1")):
        col, _, lbl = MODE_STYLE[mode]
        y = np.array([100 * cells[mode][k]["mean"] for k in keys])
        lo = np.array([100 * cells[mode][k]["ci_ic"][0] for k in keys])
        hi = np.array([100 * cells[mode][k]["ci_ic"][1] for k in keys])
        x = np.arange(2) + (i - 0.5) * w
        ax.bar(x, y, w * 0.9, yerr=np.vstack([y - lo, hi - y]), capsize=3, color=col,
               alpha=0.85, label=lbl)
        for xi, k in zip(x, keys):
            ax.scatter([xi] * 3, 100 * np.asarray(cells[mode][k]["per_seed"]), s=10,
                       color="k", alpha=0.5, zorder=3)
    ax.set_xticks(range(2)); ax.set_xticklabels(["ACT", "flow matching\n(supplementary)"])
    ax.set_ylabel("task success (%)"); ax.set_ylim(0, 103)
    ax.set_title("100 clean demonstrations"); ax.legend(fontsize=6)
    fig.tight_layout()
    return _save(fig, out, sources)


def fig_fresh(fresh: Dict[str, dict], out: str, sources: List[str]) -> Optional[str]:
    """Suite B: per-model success on the 100 fresh initial conditions."""
    if not fresh:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, (cond, col) in enumerate((("act_clean_n100", C_BASE), ("act_noise0.5_n100", C_ALT))):
        c = fresh[cond]
        ax.bar(i, 100 * c["mean"], 0.55, color=col, alpha=0.85,
               yerr=[[100 * (c["mean"] - c["ci_ic"][0])], [100 * (c["ci_ic"][1] - c["mean"])]],
               capsize=4)
        ax.scatter([i] * 3, 100 * np.asarray(c["per_seed"]), color="k", s=14, zorder=3)
        for s, v in enumerate(c["per_seed"]):
            ax.annotate(f"s{s}", (i + 0.08, 100 * v), fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["clean (100 demos)", "$\\sigma=0.5$ (100 demos)"])
    ax.set_ylabel("task success (%)"); ax.set_ylim(0, 103)
    ax.set_title("Fresh initial conditions (frozen protocol), replan every step")
    fig.tight_layout()
    return _save(fig, out, sources)


def fig_failure(fa: dict, traces: Dict[str, np.ndarray], out: str,
                sources: List[str]) -> Optional[str]:
    """Outcome categories (pooled over seeds) + distance traces by outcome."""
    cats = ["success"] + list(fa["category_order"])
    short = ["success", "terminated early", "never touched box", "touched, not lifted",
             "lifted, not near target", "near target, short hold"]
    cols = [C_ACC, "#999999", "#7B3294", "#C2A5CF", C_ALT, "#F1A340"]
    rows = [("act_clean_n100", "ensemble"), ("act_clean_n100", "replan1"),
            ("act_noise0.5_n100", "ensemble"), ("act_noise0.5_n100", "replan1")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    labels = []
    for r, (stem, mode) in enumerate(rows):
        t = fa["taxonomy"][f"{stem}__{mode}"]
        vals = [t["n_success"]] + [t["pooled"][c] for c in fa["category_order"]]
        left = 0
        for v, c, nm in zip(vals, cols, short):
            ax.barh(r, v, left=left, color=c, label=nm if r == 0 else None)
            if v >= 12:
                ax.text(left + v / 2, r, str(v), ha="center", va="center", fontsize=6,
                        color="white")
            left += v
        labels.append(("clean" if "clean" in stem else "$\\sigma$=0.5") + ", " +
                      ("ensemble" if mode == "ensemble" else "replan"))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("episodes (3 seeds x 100 initial conditions)")
    ax.set_title("Outcome categories (mutually exclusive)")
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.grid(axis="y", alpha=0)
    ax = axes[1]
    d, suc, nv = traces["dist"], traces["success"], traces["n_valid"]
    for lbl, sel, col in (("success", suc, C_ACC), ("failure", ~suc, C_ALT)):
        if sel.sum() == 0:
            continue
        D = d[sel].astype(float).copy()
        for i, n in enumerate(nv[sel]):
            D[i, n:] = np.nan
        D[:, 0] = np.nan          # reset state: positions not yet computed
        m = np.nanmedian(D, 0)
        ax.plot(m, color=col, label=f"{lbl} median (n={int(sel.sum())})")
        ax.fill_between(np.arange(D.shape[1]), np.nanpercentile(D, 25, 0),
                        np.nanpercentile(D, 75, 0), color=col, alpha=0.18, lw=0)
    ax.axhline(0.05, color="k", ls="--", lw=1)
    ax.set_xlabel("control step"); ax.set_ylabel("box$\\rightarrow$target distance (m)")
    ax.set_title("Baseline (clean, replan), valid states only"); ax.legend(fontsize=6)
    fig.tight_layout()
    return _save(fig, out, sources)
