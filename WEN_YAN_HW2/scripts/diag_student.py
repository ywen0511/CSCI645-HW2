"""Why does a student that fits its demonstrations fail in closed loop?

Separate DIAGNOSTIC runs (not the headline evaluation), per training seed of
the baseline (ACT, 100 clean demos), protocol v2:

  D2  training vs held-out initial conditions: the 100 initial conditions the
      demonstrations were collected from vs. the original 100 held-out ones.
  D3  one-step action error of the student (first action of its chunk) against
      a reference action, on (a) the stored training states vs. the stored
      actions, (b) states visited by the RETRAINED teacher and (c) states the
      student itself visits, both vs. the retrained teacher's action. The
      retrained teacher is not the one that generated the data, so (b)/(c) are
      read together with the teacher-vs-data disagreement on the training
      states (reported as `d3_new_teacher_vs_stored_mae`).
  D4  inference variants on the held-out conditions: the two headline modes
      plus ACT-reference ensemble weighting (oldest prediction weighted most),
      open-loop full chunks and 5-step chunks.

    python scripts/diag_student.py [--models act_clean_n100_s0,...]
"""
import argparse

import numpy as np

import common as CM
from hw2 import envs as E
E.configure_rendering()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="act_clean_n100_s0,act_clean_n100_s1,act_clean_n100_s2")
    ap.add_argument("--n_traj", type=int, default=16)
    a = ap.parse_args()

    import jax
    import jax.numpy as jp
    import torch
    from hw2 import collect as C
    from hw2.data import DemoSet
    from hw2.evaluate import evaluate, load_policy, make_predictor
    from hw2.teacher import load_teacher_policy

    env = E.make_env(E.TASK, impl="jax")
    spec = E.describe_env(env)
    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD
    d = DemoSet.load(str(CM.DATA / "demos_sigma0.npz"))
    idx = np.asarray(d.meta["subset_episode_indices"])
    train_keys = jax.random.split(jax.random.PRNGKey(int(d.meta["collect_key_base"])), 512)[idx]
    test_keys = C.eval_seeds(100)
    _, tpol = load_teacher_policy(E.TASK, str(CM.RUNS / "teacher" / "teacher_params.pkl"))
    vreset, vstep = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    astd = d.actions.std(0) + 1e-9

    def teacher_act(obs):
        return np.clip(np.asarray(tpol(jp.asarray(obs), jax.random.PRNGKey(0))[0]), -1, 1)

    out = {"protocol": "v2", "note": "diagnostic runs, separate from the headline evaluation"}
    floor = np.abs(teacher_act(d.obs) - d.actions)
    out["d3_new_teacher_vs_stored_mae"] = float(floor.mean())
    for name in a.models.split(","):
        rec = {}
        model, ck, norm = load_policy(str(CM.RUNS / "conditions" / name / "policy.pt"))
        predict = make_predictor(model, norm)
        # D2
        for label, keys in (("train_inits", train_keys), ("test_inits", test_keys)):
            for mode in ("ensemble", "replan1"):
                torch.manual_seed(0)
                _, _, s = evaluate(env, spec, model, norm, keys, radius, hold, mode)
                rec[f"d2_{label}_{mode}"] = {k: s[k] for k in
                                             ("n_success", "n_episodes", "success_rate",
                                              "contact_frac", "lifted_frac", "min_dist_mean")}
        # D3
        first = np.concatenate([predict(d.obs[i:i + 2048])[:, 0, :]
                                for i in range(0, len(d.obs), 2048)])
        e = np.abs(first - d.actions)
        rec["d3_train_states_mae"] = float(e.mean())
        rec["d3_train_states_mae_in_action_std"] = float((e / astd).mean())
        for follow in ("teacher", "student"):
            st = vreset(test_keys[:a.n_traj])
            errs = []
            for t in range(spec.episode_length):
                o = np.asarray(st.obs, np.float32)
                at = teacher_act(o)
                as_ = np.clip(predict(o)[:, 0, :], -1, 1)
                ok = np.isfinite(o).all(1)
                errs.append(np.abs(as_ - at)[ok].mean() if ok.any() else np.nan)
                st = vstep(st, jp.asarray(at if follow == "teacher" else as_))
            rec[f"d3_{follow}_visited_mae"] = float(np.nanmean(errs))
        rec["d3_ratio_student_over_teacher_visited"] = (
            rec["d3_student_visited_mae"] / max(rec["d3_teacher_visited_mae"], 1e-9))
        # D4
        for mode in ("ensemble", "replan1", "ensemble_oldest", "chunk_full", "chunk5"):
            torch.manual_seed(0)
            _, _, s = evaluate(env, spec, model, norm, test_keys, radius, hold, mode)
            rec[f"d4_{mode}"] = {k: s[k] for k in ("n_success", "n_episodes", "success_rate",
                                                   "contact_frac", "lifted_frac")}
        out[name] = rec
        print(f"[diag] {name}: " + "  ".join(
            f"{k}={v['n_success']}" if isinstance(v, dict) else f"{k}={v:.3f}"
            for k, v in rec.items()), flush=True)
        del model
        torch.cuda.empty_cache()
    CM.write_json(CM.RESULTS / "diag" / "diag_student.json", out)
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
