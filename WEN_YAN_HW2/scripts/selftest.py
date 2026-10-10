"""CPU-only self-test of the dataset / training / eval plumbing and the
success criterion. No GPU, no simulator."""
import os, sys, json, tempfile
import numpy as np
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))

from hw2.data import DemoSet, Normalizer, ActionChunkDataset
from hw2.envs import success_from_distances

fails = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    if not cond:
        fails.append(name)

print("== success_from_distances ==")
# exactly 10 consecutive inside -> success with hold=10
d = np.full((1, 50), 1.0); d[0, 10:20] = 0.01
check("exactly hold steps inside -> success", success_from_distances(d, 0.05, 10)[0] == True)
# 9 consecutive -> failure
d = np.full((1, 50), 1.0); d[0, 10:19] = 0.01
check("hold-1 steps inside -> failure", success_from_distances(d, 0.05, 10)[0] == False)
# 9 + gap + 9 -> failure (not consecutive)
d = np.full((1, 50), 1.0); d[0, 5:14] = 0.01; d[0, 20:29] = 0.01
check("two sub-threshold runs -> failure", success_from_distances(d, 0.05, 10)[0] == False)
# all inside -> success
check("all inside -> success", success_from_distances(np.zeros((1, 50)), 0.05, 10)[0] == True)
# shorter than hold -> failure
check("T < hold -> failure", success_from_distances(np.zeros((1, 5)), 0.05, 10)[0] == False)
# boundary: exactly at radius is NOT inside (strict <)
check("distance == radius -> failure", success_from_distances(np.full((1, 50), 0.05), 0.05, 10)[0] == False)
# batching
d = np.full((3, 50), 1.0); d[1, 0:12] = 0.0
r = success_from_distances(d, 0.05, 10)
check("batched rows independent", r.tolist() == [False, True, False], str(r.tolist()))

print("== DemoSet / chunking ==")
rng = np.random.default_rng(0)
n_ep, T, od, ad = 6, 20, 7, 3
obs = rng.normal(size=(n_ep * T, od)).astype(np.float32)
act = rng.normal(size=(n_ep * T, ad)).astype(np.float32)
ds = DemoSet(obs=obs, actions=act, rewards=np.zeros(n_ep * T, np.float32),
             episode_ends=np.arange(1, n_ep + 1) * T,
             success=np.ones(n_ep, bool), meta={"name": "synthetic"})
check("n_episodes", ds.n_episodes == n_ep)
check("episode_lengths", (ds.episode_lengths == T).all())
check("summary transitions", ds.summary()["n_transitions"] == n_ep * T)
sub = ds.subset([0, 2])
check("subset episodes", sub.n_episodes == 2 and sub.obs.shape[0] == 2 * T)
check("subset is nested prefix-consistent",
      np.allclose(ds.subset([0, 1, 2]).obs[:T], ds.subset([0]).obs))
nz = Normalizer.fit(sub)
check("normalizer no zero std", (nz.obs_std > 0).all() and (nz.act_std > 0).all())
# constant dimension must not explode
c = DemoSet(obs=np.ones((10, 2), np.float32), actions=np.ones((10, 2), np.float32),
            rewards=np.zeros(10, np.float32), episode_ends=np.array([10]),
            success=np.ones(1, bool), meta={})
nc = Normalizer.fit(c)
check("constant dim std -> 1", np.allclose(nc.obs_std, 1.0))
check("normalize constant dim finite", np.isfinite(nc.norm_obs(c.obs)).all())

H = 8
cd = ActionChunkDataset(ds, H, nz)
check("dataset length == transitions", len(cd) == n_ep * T)
o, ch, pad = cd[0]
check("chunk shapes", o.shape == (od,) and ch.shape == (H, ad) and pad.shape == (H,))
check("no padding mid-episode", pad.sum() == 0)
o, ch, pad = cd[T - 1]                      # last step of episode 0
check("padding at episode end", pad.sum() == H - 1, f"pad={pad.tolist()}")
check("pad repeats final action", np.allclose(ch[1:], ch[0]))
# chunks must never cross an episode boundary
o, ch, pad = cd[T - 2]
ep0_last = ds.actions[T - 1]
check("chunk does not cross boundary",
      np.allclose(nz.denorm_act(ch[2]), ep0_last, atol=1e-5))

print("== ACT + flow forward/backward ==")
import torch
from hw2.train_il import train, build_policy
for algo, ov in (("act", {"hidden_dim": 32, "dim_feedforward": 64, "n_heads": 2,
                          "n_enc_layers": 1, "n_dec_layers": 1,
                          "n_cvae_enc_layers": 1, "latent_dim": 4}),
                 ("flow", {"hidden_dim": 32, "n_layers": 2, "n_heads": 2})):
    m, cfg = build_policy(algo, od, ad, H, ov)
    ob = torch.randn(4, od); ac = torch.randn(4, H, ad); pd = torch.zeros(4, H)
    loss, logs = m.loss(ob, ac, pd)
    loss.backward()
    gnorm = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    check(f"{algo}: loss finite", np.isfinite(logs["loss"]), str(logs))
    check(f"{algo}: gradients flow", gnorm > 0)
    ch = m.predict_chunk(ob)
    check(f"{algo}: predict_chunk shape", tuple(ch.shape) == (4, H, ad))
    # padded steps must not affect the loss
    pd2 = torch.zeros(4, H); pd2[:, H // 2:] = 1.0
    ac2 = ac.clone(); ac2[:, H // 2:] = 1e3
    torch.manual_seed(0); l_a, _ = m.loss(ob, ac, pd2)
    torch.manual_seed(0); l_b, _ = m.loss(ob, ac2, pd2)
    check(f"{algo}: padded steps excluded from loss",
          abs(float(l_a) - float(l_b)) < 1e-3, f"{float(l_a):.5f} vs {float(l_b):.5f}")

print("== end-to-end train + checkpoint reload ==")
with tempfile.TemporaryDirectory() as td:
    dp = os.path.join(td, "demos.npz"); ds.save(dp)
    od_ = os.path.join(td, "run")
    train(dp, od_, algo="act", n_demos=4, chunk=H, steps=12, batch_size=8,
          lr=1e-3, warmup=2, seed=0, num_workers=0, successful_only=True,
          policy_overrides={"hidden_dim": 32, "dim_feedforward": 64, "n_heads": 2,
                            "n_enc_layers": 1, "n_dec_layers": 1,
                            "n_cvae_enc_layers": 1, "latent_dim": 4},
          log_every=6)
    check("policy.pt written", os.path.exists(os.path.join(od_, "policy.pt")))
    from hw2.evaluate import load_policy
    m2, ck, nm = load_policy(os.path.join(od_, "policy.pt"), device="cpu")
    check("checkpoint reloads", ck["algo"] == "act" and ck["n_demos"] == 4)
    check("normalizer round-trips", set(nm) == {"obs_mean", "obs_std", "act_mean", "act_std"})
    out = m2.predict_chunk(torch.randn(2, od))
    check("reloaded policy predicts", tuple(out.shape) == (2, H, ad))
    tl = json.load(open(os.path.join(od_, "train_log.json")))
    check("train log has history", len(tl["history"]) >= 2)
    check("episode indices recorded", ck["episode_indices"] == [0, 1, 2, 3])

print("== wilson / paired stats ==")
from hw2.evaluate import wilson_ci
p, lo, hi = wilson_ci(50, 100)
check("wilson centered", abs(p - 0.5) < 1e-9 and lo < 0.5 < hi)
check("wilson returns plain floats",
      all(type(x) is float for x in (p, lo, hi)), str([type(x).__name__ for x in (p, lo, hi)]))
check("wilson json-serialisable", json.dumps({"ci": [lo, hi]}) is not None)
p0, lo0, hi0 = wilson_ci(0, 50)
check("wilson at zero", p0 == 0.0 and lo0 == 0.0 and hi0 > 0)

print()
if fails:
    print(f"SELFTEST FAILED ({len(fails)}): {fails}")
    sys.exit(1)
print("SELFTEST_OK")
