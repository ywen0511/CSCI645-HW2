"""Failure analysis (protocol v2) and verified student rollout videos.

1. Threshold validation. The physical labels use simulator evidence: finger-pad
   / box contact (MJX contact list), box lift (box height above its initial
   height, LIFT_M) and box->target distance. Before using them we check them on
   the teacher's clean rollouts and on every successful student episode: a
   successful pick-and-place must show contact and lift. The record is saved.
2. Taxonomy. Every episode of the selected models (all training seeds, both
   inference modes, read from the suite-A episode files; no new simulation) is
   classified with hw2.metrics.classify_episode: success first, then mutually
   exclusive failure categories in a fixed order. Counts must sum to the number
   of failures.
3. Videos. For the baseline and the sigma=0.5 student (seed 0, per-step
   replanning): see make_videos -- clips come from one batched rollout,
   are labelled by that rollout and re-scored from the rendered states.

    python scripts/stage2b_failure.py
"""
import argparse
import csv

import numpy as np

import common as CM
from hw2 import envs as E
E.configure_rendering()

MODELS = {"baseline": "act_clean_n100", "noise0.5": "act_noise0.5_n100",
          "flow": "flow_clean_n100", "n25": "act_clean_n25", "n50": "act_clean_n50"}
VIDEO_MODELS = ["act_clean_n100_s0", "act_noise0.5_n100_s0"]


def load_ep(suite, name, mode):
    z = np.load(CM.RESULTS / "episodes" / suite / f"{name}__{mode}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()
    from hw2 import metrics as MT

    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD
    out = {"protocol": MT.PROTOCOL_VERSION, "lift_m": MT.LIFT_M,
           "proximity_m": MT.PROXIMITY_M, "category_order": MT.FAILURE_ORDER}

    # ---------------- 1. threshold validation ----------------
    t = load_ep("teacher", "teacher", "sigma0")
    succ_ev = {"contact": [], "lifted": [], "within_12mm": [], "max_lift": [],
               "min_grip": [], "source": []}

    def add(z, tag):
        s = z["ep_success"].astype(bool)
        succ_ev["contact"] += z["ep_any_contact"][s].tolist()
        succ_ev["lifted"] += z["ep_lifted"][s].tolist()
        succ_ev["within_12mm"] += z["ep_within_12mm"][s].tolist()
        succ_ev["max_lift"] += z["ep_max_lift"][s].tolist()
        succ_ev["min_grip"] += z["ep_min_grip"][s].tolist()
        succ_ev["source"] += [tag] * int(s.sum())
    add(t, "teacher_sigma0")
    for key, stem in MODELS.items():
        for s in (0, 1, 2):
            for mode in ("ensemble", "replan1"):
                add(load_ep("A", f"{stem}_s{s}", mode), f"{stem}_s{s}_{mode}")
    n_s = len(succ_ev["contact"])
    tgt_z = t["raw_box_z"][:, 0]
    out["threshold_validation"] = {
        "n_successful_episodes_checked": n_s,
        "frac_with_contact": float(np.mean(succ_ev["contact"])),
        "frac_lifted": float(np.mean(succ_ev["lifted"])),
        "frac_within_12mm": float(np.mean(succ_ev["within_12mm"])),
        "max_lift_m_percentiles": {f"p{q}": float(np.percentile(succ_ev["max_lift"], q))
                                   for q in (0, 5, 50)},
        "min_grip_m_percentiles": {f"p{q}": float(np.percentile(succ_ev["min_grip"], q))
                                   for q in (50, 95, 100)},
        "initial_box_z_m": [float(tgt_z.min()), float(tgt_z.max())],
        "teacher_sigma0": {"n": int(t["ep_success"].size),
                           "success": int(t["ep_success"].sum()),
                           "contact": int(t["ep_any_contact"].sum()),
                           "lifted": int(t["ep_lifted"].sum()),
                           "within_12mm": int(t["ep_within_12mm"].sum())},
    }
    v = out["threshold_validation"]
    print(f"[validate] {n_s} successful episodes: contact {v['frac_with_contact']:.3f}, "
          f"lifted {v['frac_lifted']:.3f}, within 12 mm {v['frac_within_12mm']:.3f}; "
          f"min lift p0 {v['max_lift_m_percentiles']['p0']:.3f} m", flush=True)

    # ---------------- 2. taxonomy ----------------
    tax_dir = CM.RESULTS / "failure"
    tax_dir.mkdir(parents=True, exist_ok=True)
    out["taxonomy"] = {}
    for key, stem in MODELS.items():
        for mode in ("ensemble", "replan1"):
            pooled = {c: 0 for c in MT.FAILURE_ORDER}
            per_seed, n_ep, n_succ = {}, 0, 0
            for s in (0, 1, 2):
                name = f"{stem}_s{s}"
                z = load_ep("A", name, mode)
                lab = z["ep_label"].astype(str)
                suc = z["ep_success"].astype(bool)
                assert np.all((lab == "success") == suc), f"{name}: success/label mismatch"
                cnt = {c: int((lab == c).sum()) for c in MT.FAILURE_ORDER}
                assert sum(cnt.values()) == int((~suc).sum())
                per_seed[s] = {"n_success": int(suc.sum()), **cnt}
                for c in cnt:
                    pooled[c] += cnt[c]
                n_ep += len(suc); n_succ += int(suc.sum())
                with open(tax_dir / f"episodes_{name}__{mode}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["episode", "outcome", "success", "term_cause",
                                "n_valid_states", "longest_hold_steps", "min_box_target_m",
                                "final_box_target_m", "any_finger_contact",
                                "contact_steps", "max_box_lift_m", "min_gripper_box_m",
                                "within_12mm", "left_after_success"])
                    for i in range(len(suc)):
                        w.writerow([i, lab[i], int(suc[i]), z["ep_term_cause"][i],
                                    int(z["ep_n_valid_states"][i]),
                                    int(z["ep_longest_hold"][i]),
                                    f"{z['ep_min_dist'][i]:.4f}", f"{z['ep_final_dist'][i]:.4f}",
                                    int(z["ep_any_contact"][i]), int(z["ep_contact_steps"][i]),
                                    f"{z['ep_max_lift'][i]:.4f}", f"{z['ep_min_grip'][i]:.4f}",
                                    int(z["ep_within_12mm"][i]),
                                    int(z["ep_left_after_success"][i])])
            assert sum(pooled.values()) == n_ep - n_succ
            out["taxonomy"][f"{stem}__{mode}"] = {
                "n_episodes": n_ep, "n_success": n_succ, "n_failures": n_ep - n_succ,
                "pooled": pooled, "per_seed": per_seed}
            print(f"[taxonomy] {stem:18s} {mode:9s} success {n_succ}/{n_ep}  "
                  + "  ".join(f"{c[:22]}={pooled[c]}" for c in MT.FAILURE_ORDER), flush=True)

    # historical v1 classifier on the baseline (from the archived record)
    hist = CM.RESULTS / "historical_v1" / "failure_analysis.json"
    if hist.exists():
        h = CM.read_json(hist)
        lab = np.asarray(h["labels"])
        sr = h["success_rate"]
        n = len(lab)
        out["historical_v1_check"] = {
            "condition": h["condition"], "v1_successes": int(round(sr * n)),
            "v1_episodes_labelled_success": int((lab == "success").sum()),
            "v1_successes_with_failure_labels": int(round(sr * n)) - int((lab == "success").sum()),
            "v1_reached_box_12mm": int(round(h["reached_box_frac"] * n)),
            "v1_taxonomy": h["failure_taxonomy"],
            "explanation": "v1 tested out-of-bounds and the 12 mm proximity BEFORE "
                           "success, so successful episodes received failure labels",
        }

    # ---------------- 3. verified videos ----------------
    if not a.no_video:
        out["videos"] = make_videos(radius, hold)
    CM.write_json(CM.RESULTS / "failure_analysis.json", out)
    print("FAILURE_DONE", flush=True)


def make_videos(radius, hold, n_keep=30):
    """Render clips from ONE batched rollout and label them by that rollout.

    GPU evaluation is not bit-deterministic (batched MJX + torch kernels), so a
    re-run of the suite-A rollout reproduces most but not all outcomes. We
    therefore never name a clip after the suite-A record: we run the same
    100-episode batched evaluation again, keep the states of the first
    `n_keep` episodes, choose the clips by THIS rollout's own outcomes, re-score
    the kept (rendered) trajectory, and record how many of the 100 outcomes
    agree with the suite-A record.
    """
    import jax
    import torch
    from hw2 import collect as C
    from hw2.evaluate import evaluate, load_policy
    from hw2.envs import state_diagnostics_fn

    env = E.make_env(E.TASK, impl="jax")
    spec = E.describe_env(env)
    keys = C.eval_seeds(100)
    diag = jax.jit(state_diagnostics_fn(env, spec))
    made = []
    for name in VIDEO_MODELS:
        mode = "replan1"
        rec = load_ep("A", name, mode)
        model, _, norm = load_policy(str(CM.RUNS / "conditions" / name / "policy.pt"))
        torch.manual_seed(0)
        raw, ep, s = evaluate(env, spec, model, norm, keys, radius, hold, mode,
                              keep_states_for=list(range(n_keep)))
        agree = int((ep["success"] == rec["ep_success"].astype(bool)).sum())
        lab = ep["label"].astype(str)[:n_keep]
        picks = []
        if (lab == "success").any():
            picks.append((int(np.flatnonzero(lab == "success")[0]), "success"))
        fl = [l for l in lab if l != "success"]
        if fl:
            modal = max(set(fl), key=fl.count)
            picks.append((int(np.flatnonzero(lab == modal)[0]), modal))
        print(f"[video] {name}: video rollout {s['n_success']}/100 successes; "
              f"{agree}/100 outcomes agree with the suite-A record", flush=True)
        for idx, tag in picks:
            states = raw["kept_states"][idx]
            dg = [jax.device_get(diag(st)) for st in states]
            dist = np.array([float(d["dist"]) for d in dg])[None]
            done = np.array([bool(st.done) for st in states[1:]])[None]
            nonf = np.array([bool(d["nonfinite"]) for d in dg])[None]
            m = MT_episode(dist, done, radius, hold, nonf)
            trace_dev = float(np.nanmax(np.abs(dist[0] - raw["dist"][idx])))
            ok = (bool(m["success"][0]) == (tag == "success")) and trace_dev < 1e-6
            entry = {"model": name, "mode": mode, "episode": idx, "label": str(tag),
                     "video_rollout_successes": int(s["n_success"]),
                     "video_rollout_agreement_with_suiteA": agree,
                     "suiteA_label_same_episode": str(rec["ep_label"][idx]),
                     "rendered_vs_scored_trace_maxdev_m": trace_dev,
                     "rendered_trajectory_success": bool(m["success"][0]),
                     "rendered_min_dist_m": float(m["min_dist"][0]),
                     "rendered_longest_hold": int(m["longest_hold"][0]),
                     "max_lift_m": float(ep["max_lift"][idx]),
                     "any_contact": bool(ep["any_contact"][idx])}
            if ok:
                frames = env.render(states[::2], height=360, width=480)
                slug = "success" if tag == "success" else "failure_" + "-".join(
                    tag.replace("(", "").replace(")", "").replace("/", "").replace(",", "")
                    .replace("<", "lt").split())
                p = CM.VIDEOS / f"student_{name}_{mode}_ep{idx}_{slug}.mp4"
                w = E.write_video(str(p), frames, fps=round(1.0 / spec.ctrl_dt / 2))
                entry["path"] = CM.rel(w)
                print(f"[video] wrote {CM.rel(w)}", flush=True)
            else:
                entry["path"] = None
                print(f"[video] NOT written {entry}", flush=True)
            made.append(entry)
        del model
        torch.cuda.empty_cache()
    return made


def MT_episode(dist, done, radius, hold, nonf):
    from hw2 import metrics as MT
    return MT.episode_metrics(dist, done, radius, hold, nonfinite=nonf)


if __name__ == "__main__":
    main()
