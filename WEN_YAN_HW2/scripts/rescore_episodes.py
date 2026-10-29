"""Re-score every saved rollout from its raw traces (CPU, no simulation).

Recomputes the per-episode metrics/labels (ep_*) of each
results/episodes/<suite>/*.npz with the current hw2.scoring code and updates
the summary in the matching results/eval/<suite>/*.json (and, for the teacher,
results/teacher/teacher_report.json). The raw traces and reset keys are not
touched. Each updated record notes the scoring-code hash it was produced with.

    python scripts/rescore_episodes.py
"""
import json

import numpy as np

import common as CM
from hw2 import envs as E
from hw2.scoring import score, summarize


def main():
    radius, hold = E.SUCCESS_RADIUS, E.SUCCESS_HOLD
    code_sha = CM.code_hash(CM.EVAL_CODE)
    teacher_path = CM.RESULTS / "teacher" / "teacher_report.json"
    teacher = CM.read_json(teacher_path)
    n = 0
    for p in sorted((CM.RESULTS / "episodes").glob("*/*.npz")):
        z = dict(np.load(p, allow_pickle=False))
        raw = {k[4:]: v for k, v in z.items() if k.startswith("raw_")}
        ep = score(raw, radius, hold)
        ep.update({"_dist": raw["dist"], "_done": raw["done"], "_nonfinite": raw["nonfinite"]})
        s = summarize(ep, radius, hold)
        old_succ = z["ep_success"].astype(bool)
        assert np.array_equal(old_succ, ep["success"]), f"{p.name}: success changed"
        for k, v in ep.items():
            if not k.startswith("_"):
                z[f"ep_{k}"] = np.asarray(v).astype(str) if k in ("label", "term_cause") \
                    else np.asarray(v)
        meta = json.loads(str(z["meta"]))
        meta["scoring_code_sha256"] = code_sha
        z["meta"] = json.dumps(meta)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **z)
        tmp.replace(p)
        suite = p.parent.name
        if suite == "teacher":
            sig = p.stem.split("sigma")[1]
            s.pop("per_episode_label")
            teacher["eval"][sig] = s
        else:
            jp = CM.RESULTS / "eval" / suite / (p.stem + ".json")
            rec = CM.read_json(jp)
            rec["eval"] = {**s, "mode": rec["mode"]}
            rec["identity"]["eval_code_sha256"] = code_sha
            rec["identity_sha256"] = CM.sha256_json(rec["identity"])
            rec["rescored_from_saved_rollout"] = True
            CM.write_json(jp, rec)
        n += 1
    teacher["rescored_from_saved_rollout"] = True
    CM.write_json(teacher_path, teacher)
    print(f"[rescore] {n} episode files re-scored with scoring code {code_sha[:12]}")
    print("RESCORE_DONE")


if __name__ == "__main__":
    main()
