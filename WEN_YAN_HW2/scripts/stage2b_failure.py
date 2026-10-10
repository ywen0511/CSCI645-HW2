"""Failure analysis for the baseline student + success/failure rollout videos."""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))
from hw2 import envs as E
E.use_osmesa()

import jax
from hw2 import collect as C
from hw2.evaluate import load_policy, rollout_students, rollout_student_states
from hw2.envs import success_from_distances

TASK = "PandaPickCube"


def classify(min_dist, min_grip, held, oob, radius):
    """Mutually exclusive failure taxonomy, computed per episode."""
    if oob:
        return "box left workspace"
    if min_grip > 0.012:
        return "never reached the box"
    if min_dist > 0.15:
        return "reached box, never lifted"
    if min_dist > radius:
        return "lifted, missed target"
    if not held:
        return "entered target, did not hold"
    return "success"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="act_clean_n100_s0")
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--exec_mode", default="ensemble")
    a = ap.parse_args()

    PROJ = os.environ["PROJ"]
    ck_path = os.path.join(PROJ, "runs", "conditions", a.condition, "policy.pt")
    if not os.path.exists(ck_path):
        print(f"ERROR: no checkpoint at {ck_path}"); sys.exit(1)

    env = E.make_env(TASK, impl="jax")
    spec = E.describe_env(env)
    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD
    model, ck, norm = load_policy(ck_path)

    keys = C.eval_seeds(a.n_eval)
    res = rollout_students(env, model, norm, keys, spec.episode_length,
                           exec_mode=a.exec_mode,
                           target_rel_slice=tuple(spec.target_rel_slice),
                           box_rel_slice=tuple(spec.box_rel_slice))
    suc = success_from_distances(res["dist"], radius, hold)
    min_dist = res["dist"].min(1)
    min_grip = res["grip_dist"].min(1)
    oob = res["done"][:, -1]

    tax = {}
    labels = []
    for i in range(a.n_eval):
        lab = classify(float(min_dist[i]), float(min_grip[i]), bool(suc[i]),
                       bool(oob[i]), radius)
        labels.append(lab)
        if not suc[i]:
            tax[lab] = tax.get(lab, 0) + 1

    out = {
        "condition": a.condition, "n_eval": int(a.n_eval),
        "success_radius": radius, "success_hold": hold,
        "success_rate": float(suc.mean()),
        "reached_box_frac": float((min_grip < 0.012).mean()),
        "dist_success": res["dist"][suc].tolist(),
        "dist_failure": res["dist"][~suc].tolist(),
        "min_dist_all": min_dist.tolist(),
        "min_grip_all": min_grip.tolist(),
        "failure_taxonomy": dict(sorted(tax.items(), key=lambda kv: -kv[1])),
        "labels": labels,
        "success_by_radius": {f"{r:g}": float(np.mean(success_from_distances(res["dist"], r, hold)))
                              for r in (0.03, 0.05, 0.08, 0.10, 0.15)},
    }
    p = os.path.join(PROJ, "results", "failure_analysis.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(f"[failure] success={suc.mean():.3f} reached_box={out['reached_box_frac']:.3f}")
    print(f"[failure] taxonomy={out['failure_taxonomy']}")
    print(f"[failure] success_by_radius={out['success_by_radius']}")

    # ---- one success and one failure video ----
    vid = os.path.join(PROJ, "videos"); os.makedirs(vid, exist_ok=True)
    picks = []
    si = np.flatnonzero(suc)
    fi = np.flatnonzero(~suc)
    if si.size:
        picks.append((int(si[0]), "success"))
    if fi.size:
        # pick the most representative failure: the modal taxonomy class
        if out["failure_taxonomy"]:
            modal = next(iter(out["failure_taxonomy"]))
            cand = [i for i in fi if labels[i] == modal]
            picks.append((int(cand[0] if cand else fi[0]), f"failure_{modal.replace(' ', '-')}"))
        else:
            picks.append((int(fi[0]), "failure"))
    made = []
    for idx, tag in picks:
        states, d, g = rollout_student_states(
            env, model, norm, keys[idx], spec.episode_length,
            exec_mode=a.exec_mode,
            target_rel_slice=tuple(spec.target_rel_slice),
            box_rel_slice=tuple(spec.box_rel_slice))
        frames = env.render(states[::2], height=360, width=480)
        p2 = os.path.join(vid, f"student_{a.condition}_ep{idx}_{tag}.mp4")
        w = E.write_video(p2, frames, fps=round(1.0 / spec.ctrl_dt / 2))
        E.write_frames_npz(os.path.join(vid, f"student_{tag}_frames.npz"), frames, 8)
        made.append({"path": w, "episode": idx, "tag": tag,
                     "min_dist": float(d.min()), "min_grip": float(g.min())})
        print(f"[video] {w} min_dist={d.min():.4f} min_grip={g.min():.4f}")
    out["videos"] = made
    json.dump(out, open(p, "w"), indent=2)
    print("FAILURE_DONE")


if __name__ == "__main__":
    main()
