"""Re-generate demonstration datasets from a teacher checkpoint (optional).

The committed datasets in data/ were collected by the ORIGINAL teacher, whose
checkpoint is lost; the reported study reuses them and never re-collects. This
script reproduces the collection PROCEDURE with any teacher (e.g. the retrained
one) and writes to a separate directory, so committed data is never replaced:

  * key base 1000 + floor(1000 sigma); jax.random.split(PRNGKey(base), 512);
  * action noise N(0, sigma^2) added to the deterministic teacher action, the
    sum clipped to [-1, 1] and STORED (the executed action), noise stream
    PRNGKey(0);
  * episodes truncated at the first terminating transition;
  * sigma = 0: keep the first 100 SUCCESSFUL episodes; sigma > 0: keep the
    first 100 episodes, failures included.

    python scripts/stage1_collect.py --teacher $HW2_RUNS/teacher/teacher_params.pkl \
        --sigmas 0,0.1,0.3,0.5,1,2 --out_dir $HW2_RUNS/data_regenerated
"""
import argparse
import os

import numpy as np

import common as CM
from hw2 import envs as E
E.configure_rendering()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=str(CM.RUNS / "teacher" / "teacher_params.pkl"))
    ap.add_argument("--sigmas", default="0,0.1,0.3,0.5,1,2")
    ap.add_argument("--out_dir", default=str(CM.RUNS / "data_regenerated"))
    ap.add_argument("--n_demos", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    if os.path.abspath(a.out_dir) == os.path.abspath(CM.DATA):
        raise SystemExit("refusing to overwrite the committed datasets in data/")

    import jax
    from hw2 import collect as C
    from hw2.teacher import load_teacher_policy

    env, policy = load_teacher_policy(E.TASK, a.teacher)
    spec = E.describe_env(env)
    for sig in [float(s) for s in a.sigmas.split(",")]:
        base = 1000 + int(sig * 1000)
        keys = jax.random.split(jax.random.PRNGKey(base), a.batch)
        traj = C.rollout(env, policy, keys, spec.episode_length, action_noise=sig,
                         target_rel_slice=tuple(spec.target_rel_slice),
                         box_rel_slice=tuple(spec.box_rel_slice))
        pool = C.to_demoset(traj, E.SUCCESS_RADIUS, E.SUCCESS_HOLD,
                            meta={"name": f"sigma{sig:g}", "action_noise": sig,
                                  "collect_key_base": base})
        keep = (np.flatnonzero(pool.success)[: a.n_demos] if sig == 0.0
                else np.arange(min(a.n_demos, pool.n_episodes)))
        ds = pool.subset(keep.tolist())
        ds.meta.update({"filtered_to_success": sig == 0.0,
                        "raw_pool_episodes": int(pool.n_episodes),
                        "raw_pool_success_rate": float(pool.success.mean()),
                        "teacher_sha256": CM.sha256_file(a.teacher)})
        p = os.path.join(a.out_dir, f"demos_sigma{sig:g}.npz")
        ds.save(p)
        print(f"[collect] sigma={sig:g}: {ds.n_episodes} episodes, pool success "
              f"{pool.success.mean():.3f} -> {p}", flush=True)
    print("COLLECT_DONE", flush=True)


if __name__ == "__main__":
    main()
