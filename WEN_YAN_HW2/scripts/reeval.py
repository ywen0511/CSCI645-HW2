"""Re-evaluate every trained condition under BOTH inference modes.

The diagnostic showed that re-planning every control step beats ACT's temporal
ensembling on this task. Rather than pick the winner after seeing the test
numbers (which would be selection on the evaluation set), we evaluate every
condition under both modes on the same held-out initial conditions and report
both: the headline uses the pre-specified ACT default (temporal ensembling),
with the receding-horizon variant reported alongside.

No retraining happens here -- only extra rollouts of existing checkpoints.
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))
from hw2 import envs as E
E.use_osmesa()

import torch
from hw2 import collect as C
from hw2.evaluate import load_policy, rollout_students, summarize

TASK = "PandaPickCube"
MODES = {"ensemble": dict(exec_mode="ensemble"),
         "chunk1": dict(exec_mode="chunk", exec_horizon=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    PROJ = os.environ["PROJ"]
    res_dir = os.path.join(PROJ, "results", "conditions")
    files = sorted(glob.glob(os.path.join(res_dir, "*.json")))
    mine = [f for i, f in enumerate(files) if i % a.n_shards == a.shard]
    print(f"[shard {a.shard}/{a.n_shards}] {len(mine)} conditions", flush=True)

    env = E.make_env(TASK, impl="jax")
    spec = E.describe_env(env)
    keys = C.eval_seeds(a.n_eval)
    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD
    tr, br = tuple(spec.target_rel_slice), tuple(spec.box_rel_slice)

    for f in mine:
        rec = json.load(open(f))
        name = rec["name"]
        if "eval_chunk1" in rec and not a.force:
            print(f"[skip] {name}", flush=True); continue
        ck_path = os.path.join(PROJ, "runs", "conditions", name, "policy.pt")
        if not os.path.exists(ck_path):
            print(f"[ERROR] missing checkpoint for {name}", flush=True); continue
        t0 = time.time()
        model, ck, norm = load_policy(ck_path)
        line = [name]
        for tag, kw in MODES.items():
            r = rollout_students(env, model, norm, keys, spec.episode_length,
                                 target_rel_slice=tr, box_rel_slice=br, **kw)
            s = summarize(r, radius, hold)
            rec["eval" if tag == "ensemble" else f"eval_{tag}"] = s
            line.append(f"{tag}={s['success_rate']:.3f}(box={s['reached_box_frac']:.2f})")
        rec["eval_modes"] = list(MODES)
        rec["n_eval"] = a.n_eval
        json.dump(rec, open(f, "w"), indent=2)
        print("[REEVAL] " + "  ".join(line) + f"  ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    print("REEVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
