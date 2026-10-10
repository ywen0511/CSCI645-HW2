"""Stage 2: train + evaluate one shard of the condition grid on a single GPU.

A "condition" is one (algorithm, dataset, n_demos, seed) cell. Every condition
is evaluated on the SAME held-out evaluation seeds, which are disjoint from the
demonstration-collection seeds, so success rates are directly comparable and
condition-vs-condition tests can be paired.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))
from hw2 import envs as E
E.use_osmesa()

import jax
import torch
from hw2 import collect as C
from hw2.train_il import train
from hw2.evaluate import load_policy, rollout_students, summarize

TASK = "PandaPickCube"


def build_grid(data_dir: str):
    """The full condition grid. Shard assignment is by position in this list."""
    g = []
    # --- Part 3 Option A: demonstration QUANTITY (nested subsets, clean data) ---
    for n in (25, 50, 100):
        for seed in (0, 1, 2):
            g.append({"name": f"act_clean_n{n}_s{seed}", "algo": "act",
                      "demos": os.path.join(data_dir, "demos_sigma0.npz"),
                      "n_demos": n, "seed": seed, "group": "quantity",
                      "successful_only": True, "noise": 0.0})
    # --- Part 3 Option B: teacher action NOISE (dataset size held at 100) ---
    # sigma <= 0.3 left this teacher at 98-100% success (stage 1), so the
    # quality axis only becomes informative at larger noise.
    for sig in ("0.5", "1", "2"):
        for seed in (0, 1, 2):
            g.append({"name": f"act_noise{sig}_n100_s{seed}", "algo": "act",
                      "demos": os.path.join(data_dir, f"demos_sigma{sig}.npz"),
                      "n_demos": 100, "seed": seed, "group": "noise",
                      "successful_only": False, "noise": float(sig)})
    # --- extra modern baseline: flow matching on the clean 100-demo set ---
    for seed in (0, 1, 2):
        g.append({"name": f"flow_clean_n100_s{seed}", "algo": "flow",
                  "demos": os.path.join(data_dir, "demos_sigma0.npz"),
                  "n_demos": 100, "seed": seed, "group": "algorithm",
                  "successful_only": True, "noise": 0.0})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--exec_mode", default="ensemble", choices=["ensemble", "chunk"])
    ap.add_argument("--only", default=None, help="comma-separated condition names")
    # Capacity is scaled down from the image-based ACT reference: our input is a
    # 66-d state vector and the largest dataset is ~15k transitions, so the
    # reference width (512 / ff 3200) would be badly over-parameterised here.
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dim_feedforward", type=int, default=1024)
    ap.add_argument("--flow_hidden", type=int, default=256)
    ap.add_argument("--flow_layers", type=int, default=4)
    # hidden_dim must be divisible by n_heads (nn.MultiheadAttention asserts it);
    # the FlowConfig default of 6 heads does not divide 256.
    ap.add_argument("--flow_heads", type=int, default=4)
    a = ap.parse_args()

    PROJ = os.environ["PROJ"]
    data_dir = os.path.join(PROJ, "data")
    runs_dir = os.path.join(PROJ, "runs", "conditions"); os.makedirs(runs_dir, exist_ok=True)
    res_dir = os.path.join(PROJ, "results", "conditions"); os.makedirs(res_dir, exist_ok=True)

    grid = build_grid(data_dir)
    if a.only:
        want = set(a.only.split(","))
        grid = [c for c in grid if c["name"] in want]
    mine = [c for i, c in enumerate(grid) if i % a.n_shards == a.shard]
    print(f"[shard {a.shard}/{a.n_shards}] {len(mine)} conditions: "
          f"{[c['name'] for c in mine]}", flush=True)

    env = E.make_env(TASK, impl="jax")
    spec = E.describe_env(env)
    eval_keys = C.eval_seeds(a.n_eval)
    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD

    for cond in mine:
        t0 = time.time()
        out_dir = os.path.join(runs_dir, cond["name"])
        res_path = os.path.join(res_dir, cond["name"] + ".json")
        if os.path.exists(res_path):
            print(f"[skip] {cond['name']} already has results", flush=True)
            continue
        print(f"\n===== {cond['name']} =====", flush=True)
        if not os.path.exists(cond["demos"]):
            print(f"[ERROR] missing dataset {cond['demos']}", flush=True)
            continue

        ov = ({"hidden_dim": a.hidden_dim, "dim_feedforward": a.dim_feedforward}
              if cond["algo"] == "act"
              else {"hidden_dim": a.flow_hidden, "n_layers": a.flow_layers,
                    "n_heads": a.flow_heads})
        train(cond["demos"], out_dir, algo=cond["algo"], n_demos=cond["n_demos"],
              chunk=a.chunk, steps=a.steps, batch_size=a.batch_size, lr=a.lr,
              seed=cond["seed"], num_workers=2,
              successful_only=cond["successful_only"], policy_overrides=ov)

        model, ck, norm = load_policy(os.path.join(out_dir, "policy.pt"))
        res = rollout_students(env, model, norm, eval_keys, spec.episode_length,
                               exec_mode=a.exec_mode,
                               target_rel_slice=tuple(spec.target_rel_slice),
                               box_rel_slice=tuple(spec.box_rel_slice))
        s = summarize(res, radius, hold)
        rec = {**cond, "exec_mode": a.exec_mode, "chunk": a.chunk,
               "policy_overrides": ov,
               "train_steps": a.steps, "batch_size": a.batch_size, "lr": a.lr,
               "n_params": ck["n_params"], "demo_summary": ck["demo_summary"],
               "eval": s, "wall_s": time.time() - t0,
               "eval_seed_base": 7777, "n_eval": a.n_eval}
        json.dump(rec, open(res_path, "w"), indent=2)
        print(f"[RESULT] {cond['name']}: success={s['success_rate']:.3f} "
              f"CI95={s['success_ci95']} min_dist={s['min_dist_mean']:.4f} "
              f"reached_box={s['reached_box_frac']} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    print("SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
