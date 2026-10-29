"""Stage 2b: evaluate trained students (protocol v2), resumable.

Suite A  every model of the grid x {ensemble, replan1} on the ORIGINAL 100
         held-out initial conditions, jax.random.split(PRNGKey(7777), 100).
Suite B  the frozen fresh-initialization confirmation in
         configs/fresh_protocol.json (6 models x replan1 x 100 new conditions).

    python scripts/stage2_eval.py --suite A [--shard i --n_shards k]
    python scripts/stage2_eval.py --suite B

Writes results/eval/<suite>/<model>__<mode>.json (summary + identity) and
results/episodes/<suite>/<model>__<mode>.npz (reset keys, validity, terminal
causes, ground-truth traces, per-episode metrics and labels).
"""
import argparse
import time

import numpy as np

import common as CM
from hw2 import envs as E
E.configure_rendering()

ORIG_BASE, N_EVAL = 7777, 100


def suite_jobs(suite):
    from hw2.grid import build_grid
    grid = build_grid()
    if suite == "A":
        return [(c, m) for c in grid for m in ("ensemble", "replan1")], ORIG_BASE, None
    proto = CM.read_json(CM.CONFIGS / "fresh_protocol.json")
    names = set(proto["models"])
    jobs = [(c, proto["mode"]) for c in grid if c["name"] in names]
    assert len(jobs) == len(names) == 6, "fresh protocol must name 6 grid models"
    return jobs, int(proto["eval_base_seed"]), proto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["A", "B"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma-separated model names")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import torch
    from hw2 import collect as C
    from hw2 import metrics as MT
    from hw2.evaluate import evaluate, key_data, load_policy, save_episodes

    jobs, base, proto = suite_jobs(a.suite)
    if a.only:
        want = set(a.only.split(","))
        jobs = [j for j in jobs if j[0]["name"] in want]
    mine = [j for i, j in enumerate(jobs) if i % a.n_shards == a.shard]
    print(f"[eval {a.suite} shard {a.shard}/{a.n_shards}] {len(mine)} jobs", flush=True)

    if proto is not None:
        # the frozen protocol pins the key hash; refuse to run on anything else
        keys = C.eval_seeds(proto["n_episodes"], base=base)
        got = CM.sha256_json(key_data(keys).tolist())
        assert got == proto["reset_keys_sha256"], \
            f"fresh keys hash {got} != frozen {proto['reset_keys_sha256']}"
    else:
        keys = C.eval_seeds(N_EVAL, base=base)
    keys_sha = CM.sha256_json(key_data(keys).tolist())

    env = E.make_env(E.TASK, impl="jax")
    spec = E.describe_env(env)
    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD
    code_sha = CM.code_hash(CM.EVAL_CODE)
    out_json = CM.RESULTS / "eval" / a.suite
    out_npz = CM.RESULTS / "episodes" / a.suite

    for cond, mode in mine:
        name = cond["name"]
        ck_dir = CM.RUNS / "conditions" / name
        ck = ck_dir / "policy.pt"
        if not ck.exists():
            raise SystemExit(f"[ERROR] missing checkpoint {CM.rel(ck)}; run stage2_train.py")
        train_id = CM.read_json(ck_dir / "identity.json")
        ident = {"policy_sha256": CM.sha256_file(ck),
                 "train_identity_sha256": train_id["identity_sha256"],
                 "reset_keys_sha256": keys_sha, "eval_base_seed": base,
                 "n_episodes": int(keys.shape[0]), "mode": mode,
                 "protocol": MT.PROTOCOL_VERSION, "eval_code_sha256": code_sha,
                 "radius": radius, "hold": hold}
        jpath = out_json / f"{name}__{mode}.json"
        if not a.force and jpath.exists() and CM.read_json(jpath).get("identity") == ident:
            print(f"[skip] {name} {mode}: identity matches", flush=True)
            continue
        assert train_id["policy_sha256"] == ident["policy_sha256"], \
            f"{name}: policy.pt changed after training"
        t0 = time.time()
        model, ckd, norm = load_policy(str(ck))
        torch.manual_seed(0)       # flow matching samples noise at inference
        raw, ep, s = evaluate(env, spec, model, norm, keys, radius, hold, mode)
        meta = {"model": name, "suite": a.suite, **ident}
        save_episodes(str(out_npz / f"{name}__{mode}.npz"), keys, raw, ep, meta)
        rec = {"model": name, "suite": a.suite, "condition": cond, "mode": mode,
               "identity": ident, "identity_sha256": CM.sha256_json(ident),
               "eval": s, "n_params": int(ckd["n_params"]),
               "demo_summary": ckd["demo_summary"],
               "episodes_file": CM.rel(out_npz / f"{name}__{mode}.npz"),
               "wall_s": time.time() - t0, "git": CM.git_describe()}
        CM.write_json(jpath, rec)
        print(f"[RESULT {a.suite}] {name:22s} {mode:9s} "
              f"success={s['n_success']}/{s['n_episodes']} "
              f"term={s['terminated_frac']:.2f} changed_vs_v1={s['n_changed_vs_v1_same_rollout']} "
              f"obs_dev={s['obs_vs_sim_dist_maxdev']:.1e} ({time.time()-t0:.0f}s)",
              flush=True)
        del model
        torch.cuda.empty_cache()
    print("EVAL_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
