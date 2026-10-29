"""Stage 2a: train student models of the grid (resumable, identity-checked).

    python scripts/stage2_train.py [--shard i --n_shards k] [--only a,b] [--force]

A model is skipped only if runs/conditions/<name>/policy.pt exists AND its
stored identity (dataset hash, resolved config, training-code hash) equals the
identity this run would produce AND the checkpoint hash still matches.
"""
import argparse
import os
import time

import common as CM                       # sets sys.path and PROJ
from hw2.grid import TRAIN, build_grid


def identity(cond):
    ds = CM.DATA / cond["dataset"]
    return {
        "dataset": cond["dataset"], "dataset_sha256": CM.sha256_file(ds),
        "config": {**{k: cond[k] for k in ("algo", "n_demos", "seed",
                                            "successful_only", "policy_overrides")},
                   **TRAIN},
        "train_code_sha256": CM.code_hash(CM.TRAIN_CODE),
    }


def is_current(out_dir, ident):
    p_id, p_ck = out_dir / "identity.json", out_dir / "policy.pt"
    if not (p_id.exists() and p_ck.exists()):
        return False
    rec = CM.read_json(p_id)
    return (rec.get("identity") == ident
            and rec.get("policy_sha256") == CM.sha256_file(p_ck))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    a = ap.parse_args()

    import torch
    from hw2.train_il import train

    grid = build_grid()
    if a.only:
        want = set(a.only.split(","))
        grid = [c for c in grid if c["name"] in want]
    mine = [c for i, c in enumerate(grid) if i % a.n_shards == a.shard]
    print(f"[train shard {a.shard}/{a.n_shards}] {[c['name'] for c in mine]}", flush=True)
    for cond in mine:
        out_dir = CM.RUNS / "conditions" / cond["name"]
        ident = identity(cond)
        if not a.force and is_current(out_dir, ident):
            print(f"[skip] {cond['name']}: checkpoint identity matches", flush=True)
            continue
        t0 = time.time()
        train(str(CM.DATA / cond["dataset"]), str(out_dir), algo=cond["algo"],
              n_demos=cond["n_demos"], chunk=TRAIN["chunk"], steps=TRAIN["steps"],
              batch_size=TRAIN["batch_size"], lr=TRAIN["lr"],
              weight_decay=TRAIN["weight_decay"], warmup=TRAIN["warmup"],
              seed=cond["seed"], ema_decay=TRAIN["ema_decay"],
              num_workers=a.num_workers,
              successful_only=cond["successful_only"],
              policy_overrides=cond["policy_overrides"])
        CM.write_json(out_dir / "identity.json", {
            "name": cond["name"], "identity": ident,
            "identity_sha256": CM.sha256_json(ident),
            "policy_sha256": CM.sha256_file(out_dir / "policy.pt"),
            "provenance": "retrained (historical checkpoints were not recoverable)",
            "git": CM.git_describe(), "host": os.uname().nodename,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "wall_s": time.time() - t0,
        })
        print(f"[trained] {cond['name']} in {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    print("TRAIN_SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
