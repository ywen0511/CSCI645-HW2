"""Freeze the fresh-initialization confirmation protocol (suite B).

Run ONCE, before any suite-B outcome exists; commit configs/fresh_protocol.json
before launching `stage2_eval.py --suite B`. It refuses to overwrite an
existing protocol. The script derives the fresh reset keys and checks that they
collide neither with the original evaluation keys nor with any demonstration
collection key, both as raw key data and as initial box/target configurations.

    python scripts/freeze_fresh_protocol.py [--base 17777]
"""
import argparse
import datetime as dt

import numpy as np

import common as CM
from hw2 import envs as E
E.configure_rendering()

MODELS = [f"act_clean_n100_s{s}" for s in (0, 1, 2)] + \
         [f"act_noise0.5_n100_s{s}" for s in (0, 1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=17777)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    out = CM.CONFIGS / "fresh_protocol.json"
    if out.exists():
        raise SystemExit(f"{CM.rel(out)} already exists; the protocol is frozen")

    import jax
    from hw2 import collect as C
    from hw2.data import DemoSet
    from hw2.evaluate import key_data

    fresh = C.eval_seeds(a.n, base=a.base)
    orig = C.eval_seeds(100, base=7777)
    others = {"original_eval_7777": orig}
    for f in sorted(CM.DATA.glob("demos_sigma*.npz")):
        b = int(DemoSet.load(str(f)).meta["collect_key_base"])
        others[f"demo_pool_{f.stem}_base{b}"] = jax.random.split(jax.random.PRNGKey(b), 512)
    fk = {tuple(r) for r in key_data(fresh).tolist()}
    assert len(fk) == a.n, "duplicate fresh keys"
    collisions = {name: len(fk & {tuple(r) for r in key_data(k).tolist()})
                  for name, k in others.items()}

    # initial configurations (box pose + target) of every key set
    env = E.make_env(E.TASK, impl="jax")
    vreset = jax.jit(jax.vmap(env.reset))

    def config(keys):
        st = vreset(keys)
        box = np.asarray(st.data.xpos[:, env._obj_body])
        tgt = np.asarray(st.info["target_pos"])
        return np.concatenate([box, tgt], 1)
    cf = config(fresh)
    nearest = {}
    for name, k in others.items():
        co = config(k)
        dmin = np.sqrt(((cf[:, None] - co[None]) ** 2).sum(-1)).min()
        nearest[name] = float(dmin)
    ok = all(v == 0 for v in collisions.values()) and min(nearest.values()) > 1e-6
    assert ok, f"fresh keys overlap: {collisions} {nearest}"

    proto = {
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "confirm (or not) the moderate-noise finding on unseen initial conditions",
        "models": MODELS,
        "checkpoints": "the same suite-A checkpoints (runs/conditions/<model>/policy.pt), "
                       "verified by policy_sha256 in each evaluation identity",
        "mode": "replan1",
        "mode_definition": "re-plan every control step, execute only the first predicted action",
        "eval_base_seed": a.base,
        "n_episodes": a.n,
        "reset_keys": "jax.random.split(jax.random.PRNGKey(eval_base_seed), n_episodes)",
        "reset_keys_sha256": CM.sha256_json(key_data(fresh).tolist()),
        "key_collisions": collisions,
        "nearest_initial_config_distance_m": nearest,
        "success": "box-target distance (simulator) < 0.05 m for 10 consecutive valid "
                   "control steps, protocol v2 (termination-aware)",
        "estimand": "mean over the 3 training seeds of the success rate on the 100 fresh "
                    "initial conditions; contrast = noisy(sigma=0.5) - clean",
        "analysis": [
            "per-model success counts and Wilson 95% intervals",
            "per-condition mean of the 3 per-seed success rates",
            "paired bootstrap over the 100 shared initial conditions (20,000 resamples, "
            "numpy seed 0), models held fixed: uncertainty conditional on these 3+3 models",
            "two-way bootstrap resampling training seeds within condition AND initial "
            "conditions: crude (3 seeds) but includes seed-to-seed variation",
            "exact sign-flip test on per-condition-differences of seed-averaged success "
            "per initial condition (10^5 random flips)",
            "per-seed-index differences (seed s noisy - seed s clean) listed, not tested",
        ],
        "decision_rule": "report 'confirmed' only if the paired initial-condition bootstrap "
                         "95% CI of the contrast lies entirely above 0; otherwise report "
                         "'not confirmed' with the estimate. The two-way interval is reported "
                         "alongside as the more conservative statement.",
        "commitments": [
            "evaluate once; no re-runs with other evaluation seeds",
            "no tuning, checkpoint or seed replacement based on this set",
            "all 600 episode outcomes are saved and reported",
        ],
    }
    CM.write_json(out, proto)
    print(f"froze {CM.rel(out)}: collisions={collisions}")
    print(f"nearest initial-config distances (m): {nearest}")
    print("FREEZE_DONE")


if __name__ == "__main__":
    main()
