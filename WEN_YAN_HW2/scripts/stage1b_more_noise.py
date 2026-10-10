"""Stage 1b: collect demonstration pools at LARGER teacher action noise.

Stage 1 showed that sigma <= 0.3 barely degrades this teacher (100/100/98%
success), so the Option B axis was uninformative at those levels. A quality
ablation only says something where the degraded arm is actually degraded, so we
extend the sweep until teacher success really falls.

This job reconstructs the teacher from the saved parameter pickle, and FIRST
verifies that the reconstruction reproduces stage 1's clean success rate --
rebuilding a brax inference function is exactly where observation normalisation
silently goes wrong, and a mis-normalised teacher would quietly poison every
dataset collected here.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))
from hw2 import envs as E
E.use_osmesa()

import jax
from hw2 import collect as C
from hw2.teacher import load_teacher_policy
from hw2.evaluate import summarize

TASK = "PandaPickCube"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default=None)
    ap.add_argument("--sigmas", default="0.5,1.0,2.0")
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--n_demo_target", type=int, default=100)
    ap.add_argument("--collect_batch", type=int, default=512)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()

    PROJ = os.environ["PROJ"]
    tdir = a.teacher_dir or os.path.join(PROJ, "runs", "teacher")
    rpt_path = os.path.join(tdir, "stage1_report.json")
    report = json.load(open(rpt_path))
    radius, hold = report["success_radius"], report["success_hold"]
    ref = report["teacher_eval_sigma0.0"]["success_rate"]

    env, policy_fn = load_teacher_policy(TASK, os.path.join(tdir, "teacher_params.pkl"))
    spec = E.describe_env(env)
    data_dir = os.path.join(PROJ, "data"); os.makedirs(data_dir, exist_ok=True)

    # ---- reconstruction check on the identical held-out seeds ----
    res = C.rollout(env, policy_fn, C.eval_seeds(a.n_eval), spec.episode_length,
                    action_noise=0.0,
                    target_rel_slice=tuple(spec.target_rel_slice),
                    box_rel_slice=tuple(spec.box_rel_slice))
    s0 = summarize(res, radius, hold)
    print(f"[check] reloaded teacher clean success={s0['success_rate']:.3f} "
          f"(stage1 reported {ref:.3f})", flush=True)
    if abs(s0["success_rate"] - ref) > a.tol:
        print(f"*** ABORT: reloaded teacher does not reproduce stage 1 "
              f"({s0['success_rate']:.3f} vs {ref:.3f}). The saved policy is not "
              f"being rebuilt faithfully; refusing to collect data with it.")
        sys.exit(1)
    print("[check] reconstruction OK", flush=True)

    sigmas = [float(x) for x in a.sigmas.split(",")]
    for sigma in sigmas:
        tag = f"sigma{sigma:g}"
        # teacher success at this noise level, on the shared held-out seeds
        r = C.rollout(env, policy_fn, C.eval_seeds(a.n_eval), spec.episode_length,
                      action_noise=sigma,
                      target_rel_slice=tuple(spec.target_rel_slice),
                      box_rel_slice=tuple(spec.box_rel_slice))
        s = summarize(r, radius, hold)
        s.pop("per_episode_success", None)
        s["success_by_radius"] = {
            f"{q:g}": float(np.mean(E.success_from_distances(r["dist"], q, hold)))
            for q in (0.03, 0.05, 0.08, 0.10, 0.15)}
        md = r["dist"].min(1)
        s["min_dist_percentiles"] = {f"p{q}": float(np.percentile(md, q))
                                     for q in (10, 25, 50, 75, 90)}
        report[f"teacher_eval_sigma{sigma}"] = s
        print(f"[teacher-eval] sigma={sigma} success={s['success_rate']:.3f} "
              f"CI={s['success_ci95']} return={s['return_mean']:.1f} "
              f"reached_box={s['reached_box_frac']}", flush=True)

        # dataset: fixed 100 episodes, NOT filtered for success
        keys = jax.random.split(jax.random.PRNGKey(1000 + int(sigma * 1000)),
                                a.collect_batch)
        traj = C.rollout(env, policy_fn, keys, spec.episode_length,
                         action_noise=sigma,
                         target_rel_slice=tuple(spec.target_rel_slice),
                         box_rel_slice=tuple(spec.box_rel_slice))
        ds = C.to_demoset(traj, radius, hold,
                          meta={"name": tag, "action_noise": sigma,
                                "collect_key_base": 1000 + int(sigma * 1000)})
        pool_rate = float(ds.success.mean())
        final = ds.subset(list(range(min(a.n_demo_target, ds.n_episodes))))
        final.meta.update({"name": tag, "action_noise": sigma,
                           "filtered_to_success": False,
                           "raw_pool_episodes": int(ds.n_episodes),
                           "raw_pool_success_rate": pool_rate})
        p = os.path.join(data_dir, f"demos_{tag}.npz")
        final.save(p)
        report["datasets"][tag] = {"path": p, "summary": final.summary(),
                                   "meta": final.meta}
        print(f"[collect] sigma={sigma} saved {final.n_episodes} eps "
              f"(pool success {pool_rate:.3f}) -> {p}", flush=True)
        print(f"          {json.dumps(final.summary())}", flush=True)

    def _j(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(str(type(o)))
    json.dump(report, open(rpt_path, "w"), indent=2, default=_j)
    print("STAGE1B_DONE", flush=True)


if __name__ == "__main__":
    main()
