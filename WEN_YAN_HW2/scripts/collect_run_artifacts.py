"""Copy the small run records out of $HW2_RUNS into the repository, write the
checkpoint manifest, and build the checkpoint bundles for download.

After this step the report can be regenerated on a CPU-only machine from the
committed files alone (scripts/stage3_report.py never reads $HW2_RUNS).

    python scripts/collect_run_artifacts.py [--bundle]
"""
import argparse
import shutil
import tarfile

import common as CM
from hw2.grid import build_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="store_true",
                    help="also write checkpoint tarballs to $HW2_RUNS/release")
    a = ap.parse_args()
    tl, idd, td = CM.CONFIGS / "trainlogs", CM.CONFIGS / "identities", CM.CONFIGS / "teacher"
    for d in (tl, idd, td):
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in build_grid():
        r = CM.RUNS / "conditions" / c["name"]
        shutil.copy(r / "train_log.json", tl / f"trainlog_{c['name']}.json")
        shutil.copy(r / "identity.json", idd / f"{c['name']}.json")
        ident = CM.read_json(r / "identity.json")
        sha = CM.sha256_file(r / "policy.pt")
        assert sha == ident["policy_sha256"], f"{c['name']}: checkpoint changed"
        rows.append({"model": c["name"], "file": f"conditions/{c['name']}/policy.pt",
                     "sha256": sha, "bytes": (r / "policy.pt").stat().st_size,
                     "train_identity_sha256": ident["identity_sha256"],
                     "provenance": ident["provenance"], "dataset": c["dataset"],
                     "dataset_sha256": ident["identity"]["dataset_sha256"]})
    t = CM.RUNS / "teacher"
    for f in ("ppo_progress.json", "teacher_info.json"):
        shutil.copy(t / f, td / f)
    rows.append({"model": "ppo_teacher", "file": "teacher/teacher_params.pkl",
                 "sha256": CM.sha256_file(t / "teacher_params.pkl"),
                 "bytes": (t / "teacher_params.pkl").stat().st_size,
                 "provenance": "retrained; NOT the policy that generated data/*.npz"})
    bundles = {}
    if a.bundle:
        rel = CM.RUNS / "release"
        rel.mkdir(exist_ok=True)
        groups = {"hw2_ckpt_act_quantity.tar": lambda n: n.startswith("act_clean"),
                  "hw2_ckpt_act_noise.tar": lambda n: n.startswith("act_noise"),
                  "hw2_ckpt_flow_and_teacher.tar": lambda n: n.startswith("flow") or n == "ppo_teacher"}
        for tar, sel in groups.items():
            p = rel / tar
            with tarfile.open(p, "w") as tf:
                for r in rows:
                    if sel(r["model"]):
                        tf.add(CM.RUNS / r["file"], arcname=r["file"])
                        if r["model"] != "ppo_teacher":
                            d = (CM.RUNS / r["file"]).parent
                            for extra in ("normalizer.npz", "identity.json", "train_log.json"):
                                tf.add(d / extra, arcname=f"conditions/{d.name}/{extra}")
                        else:
                            for extra in ("teacher_info.json", "ppo_progress.json"):
                                tf.add(t / extra, arcname=f"teacher/{extra}")
            bundles[tar] = {"sha256": CM.sha256_file(p), "bytes": p.stat().st_size,
                            "models": [r["model"] for r in rows if sel(r["model"])]}
            for r in rows:
                if sel(r["model"]):
                    r["bundle"] = tar
            print(f"[bundle] {tar}: {p.stat().st_size/1e6:.0f} MB")
    CM.write_json(CM.RESULTS / "checkpoints_manifest.json", {
        "note": "checkpoints are not stored in git; download the bundles from the "
                "GitHub release named in README.md and extract them into $HW2_RUNS",
        "checkpoints": rows, "bundles": bundles})
    print(f"[collect] {len(rows)} checkpoints recorded")
    print("COLLECT_ARTIFACTS_DONE")


if __name__ == "__main__":
    main()
