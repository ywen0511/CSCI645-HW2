"""Why does a student that fits the demonstrations fail in closed loop?

Four diagnostics, in increasing order of "this would be a bug":
  D1 data integrity -- replay each stored action sequence from the SAME initial
     condition it was collected from; the simulator must reproduce the stored
     observations. If not, the (obs, action) pairing or the seeding is wrong.
  D2 train vs test initial conditions -- evaluate the student on the initial
     conditions its demonstrations came from. High there and low on held-out
     seeds == generalisation gap. Low on both == underfitting or a bug.
  D3 action accuracy -- the student's one-step action error on training states
     (should be tiny) versus its error at the states it actually visits
     (compounding error / covariate shift).
  D4 inference mode -- temporal ensembling vs open-loop chunking vs re-planning
     every step.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))
from hw2 import envs as E
E.use_osmesa()

import jax, jax.numpy as jp
import torch
from hw2 import collect as C
from hw2.data import DemoSet
from hw2.evaluate import load_policy, rollout_students, summarize
from hw2.teacher import load_teacher_policy
from hw2.envs import success_from_distances

TASK = "PandaPickCube"
out = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="act_clean_n100_s0")
    ap.add_argument("--demos", default=None)
    ap.add_argument("--n_eval", type=int, default=100)
    a = ap.parse_args()
    PROJ = os.environ["PROJ"]
    demos_path = a.demos or os.path.join(PROJ, "data", "demos_sigma0.npz")

    env = E.make_env(TASK, impl="jax")
    spec = E.describe_env(env)
    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD
    tr, br = tuple(spec.target_rel_slice), tuple(spec.box_rel_slice)
    d = DemoSet.load(demos_path)
    print(f"[demos] {d.n_episodes} eps  meta={ {k:v for k,v in d.meta.items() if k!='subset_episode_indices'} }")

    idxs = d.meta.get("subset_episode_indices")
    base = d.meta.get("collect_key_base", 1000)
    assert idxs is not None, "dataset does not record which collection keys it came from"
    keys_all = jax.random.split(jax.random.PRNGKey(base), 512)
    train_keys = jp.stack([keys_all[int(i)] for i in idxs])

    # ---------------- D1: replay the stored actions ----------------
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    devs, rep_succ = [], []
    for j in range(min(5, d.n_episodes)):
        sl = d.episode_slice(j)
        acts = d.actions[sl]
        st = reset(keys_all[int(idxs[j])])
        obs_rep = [np.asarray(st.obs)]
        for t in range(acts.shape[0]):
            st = step(st, jp.asarray(acts[t]))
            obs_rep.append(np.asarray(st.obs))
        obs_rep = np.stack(obs_rep[:-1])          # align with stored pre-step obs
        devs.append(float(np.abs(obs_rep - d.obs[sl]).max()))
        dist = np.linalg.norm(np.stack(
            [o[tr[0]:tr[1]] for o in obs_rep]), axis=-1)
        rep_succ.append(bool(success_from_distances(dist[None], radius, hold)[0]))
    out["D1_replay_max_obs_deviation"] = devs
    out["D1_replay_success"] = rep_succ
    out["D1_stored_success"] = [bool(x) for x in d.success[:len(devs)]]
    print(f"[D1] max |obs_replay - obs_stored| per episode: "
          f"{[f'{x:.2e}' for x in devs]}")
    print(f"[D1] replayed success={rep_succ}  stored success={out['D1_stored_success']}")

    # ---------------- load the student ----------------
    ck_path = os.path.join(PROJ, "runs", "conditions", a.condition, "policy.pt")
    model, ck, norm = load_policy(ck_path)
    print(f"[student] {a.condition} params={ck['n_params']/1e6:.2f}M "
          f"chunk={ck['chunk']} n_demos={ck['n_demos']}")

    # ---------------- D2: train vs test initial conditions ----------------
    for label, keys in (("train_inits", train_keys),
                        ("test_inits", C.eval_seeds(a.n_eval))):
        r = rollout_students(env, model, norm, keys, spec.episode_length,
                             exec_mode="ensemble", target_rel_slice=tr,
                             box_rel_slice=br)
        s = summarize(r, radius, hold); s.pop("per_episode_success", None)
        out[f"D2_{label}"] = s
        print(f"[D2] {label:11s} success={s['success_rate']:.3f} "
              f"reached_box={s['reached_box_frac']:.3f} "
              f"min_dist={s['min_dist_mean']:.4f}")

    # ---------------- D3: action accuracy ----------------
    om = torch.as_tensor(norm["obs_mean"], device="cuda")
    os_ = torch.as_tensor(norm["obs_std"], device="cuda")
    am = torch.as_tensor(norm["act_mean"], device="cuda")
    as_ = torch.as_tensor(norm["act_std"], device="cuda")

    def predict(obs_np):
        o = torch.as_tensor(np.asarray(obs_np, np.float32), device="cuda")
        o = (o - om) / os_
        return (model.predict_chunk(o) * as_ + am).float().cpu().numpy()

    n = min(2000, d.obs.shape[0])
    pred = predict(d.obs[:n])[:, 0, :]
    err = np.abs(pred - d.actions[:n])
    astd = d.actions.std(0) + 1e-9
    out["D3_train_state_action_mae"] = float(err.mean())
    out["D3_train_state_action_mae_in_std"] = float((err / astd).mean())
    print(f"[D3] action MAE on TRAINING states: {err.mean():.4f} raw, "
          f"{(err/astd).mean():.3f} action-std")

    # error at states the student itself reaches, against the teacher's action
    tpath = os.path.join(PROJ, "runs", "teacher", "teacher_params.pkl")
    if os.path.exists(tpath):
        tenv, tpol = load_teacher_policy(TASK, tpath)
        keys = C.eval_seeds(16)
        st = jax.vmap(tenv.reset)(keys)
        vstep = jax.jit(jax.vmap(tenv.step))
        errs, acts_t, acts_s = [], [], []
        for t in range(spec.episode_length):
            obs_np = np.asarray(st.obs, np.float32)
            a_t = np.asarray(tpol(st.obs, jax.random.PRNGKey(0))[0])
            a_s = predict(obs_np)[:, 0, :]
            errs.append(np.abs(a_s - a_t).mean())
            acts_t.append(a_t); acts_s.append(a_s)
            st = vstep(st, jp.asarray(a_t))        # follow the TEACHER
        out["D3_teacher_state_action_mae"] = float(np.mean(errs))
        print(f"[D3] action MAE on TEACHER-visited states: {np.mean(errs):.4f} raw "
              f"({np.mean(errs)/astd.mean():.3f} action-std)")

        # now follow the STUDENT and measure the same thing
        st = jax.vmap(tenv.reset)(keys)
        errs2 = []
        for t in range(spec.episode_length):
            obs_np = np.asarray(st.obs, np.float32)
            a_t = np.asarray(tpol(st.obs, jax.random.PRNGKey(0))[0])
            a_s = np.clip(predict(obs_np)[:, 0, :], -1, 1)
            errs2.append(np.abs(a_s - a_t).mean())
            st = vstep(st, jp.asarray(a_s))        # follow the STUDENT
        out["D3_student_state_action_mae"] = float(np.mean(errs2))
        print(f"[D3] action MAE on STUDENT-visited states: {np.mean(errs2):.4f} raw "
              f"({np.mean(errs2)/astd.mean():.3f} action-std)")
        out["D3_mae_ratio_student_over_teacher"] = float(
            np.mean(errs2) / max(np.mean(errs), 1e-9))

    # ---------------- D4: inference mode ----------------
    ekeys = C.eval_seeds(a.n_eval)
    for label, kw in (("ensemble", dict(exec_mode="ensemble")),
                      ("chunk_full", dict(exec_mode="chunk")),
                      ("chunk_1", dict(exec_mode="chunk", exec_horizon=1)),
                      ("chunk_5", dict(exec_mode="chunk", exec_horizon=5))):
        r = rollout_students(env, model, norm, ekeys, spec.episode_length,
                             target_rel_slice=tr, box_rel_slice=br, **kw)
        s = summarize(r, radius, hold); s.pop("per_episode_success", None)
        out[f"D4_{label}"] = {"success_rate": s["success_rate"],
                              "reached_box_frac": s["reached_box_frac"],
                              "min_dist_mean": s["min_dist_mean"]}
        print(f"[D4] {label:11s} success={s['success_rate']:.3f} "
              f"reached_box={s['reached_box_frac']:.3f} "
              f"min_dist={s['min_dist_mean']:.4f}")

    p = os.path.join(PROJ, "results", f"diag_{a.condition}.json")
    json.dump(out, open(p, "w"), indent=2)
    print("wrote", p)
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
