"""Stage 3 (CPU only): statistics, figures, tables, macros and report.pdf.

Reads ONLY committed files (results/, configs/, data/, videos/); it never
touches $HW2_RUNS or a GPU, so it reproduces the report from a fresh checkout.

    python scripts/stage3_report.py [--no_pdf]
Environment: HW2_LATEX = pdflatex (default, from PATH or $HW2_TEX_BIN) | tectonic
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

import common as CM
from hw2 import figures as FG
from hw2 import report as RP
from hw2 import stats as ST
from hw2.grid import TRAIN, build_grid

RES, CFG = CM.RESULTS, CM.CONFIGS
FIGS, TABS, REPORT = RES / "figures", RES / "tables", CM.PROJ / "report"
MODES = ("ensemble", "replan1")
MODE_NAME = {"ensemble": "temporal ensembling", "replan1": "replan every step"}
CELLS = {  # cell -> (group, algo, n_demos, sigma, display label)
    "act_clean_n25": ("quantity", "act", 25, 0.0, "ACT, 25 clean"),
    "act_clean_n50": ("quantity", "act", 50, 0.0, "ACT, 50 clean"),
    "act_clean_n100": ("quantity", "act", 100, 0.0, "ACT, 100 clean (baseline)"),
    "act_noise0.5_n100": ("noise", "act", 100, 0.5, r"ACT, 100 at $\sigma{=}0.5$"),
    "act_noise1_n100": ("noise", "act", 100, 1.0, r"ACT, 100 at $\sigma{=}1$"),
    "act_noise2_n100": ("noise", "act", 100, 2.0, r"ACT, 100 at $\sigma{=}2$"),
    "flow_clean_n100": ("algorithm", "flow", 100, 0.0, "Flow matching, 100 clean (suppl.)"),
}
CONTRASTS = [  # key, label, A, B
    ("QtyTwentyFiveHundred", r"25 $\rightarrow$ 100 clean demos", "act_clean_n25", "act_clean_n100"),
    ("QtyFiftyHundred", r"50 $\rightarrow$ 100 clean demos", "act_clean_n50", "act_clean_n100"),
    ("QtyTwentyFiveFifty", r"25 $\rightarrow$ 50 clean demos", "act_clean_n25", "act_clean_n50"),
    ("NoiseHalf", r"clean $\rightarrow$ $\sigma{=}0.5$", "act_clean_n100", "act_noise0.5_n100"),
    ("NoiseOne", r"clean $\rightarrow$ $\sigma{=}1$", "act_clean_n100", "act_noise1_n100"),
    ("NoiseTwo", r"clean $\rightarrow$ $\sigma{=}2$", "act_clean_n100", "act_noise2_n100"),
    ("ActFlow", r"ACT $\rightarrow$ flow matching", "act_clean_n100", "flow_clean_n100"),
]


def cell_of(name):
    return name.rsplit("_s", 1)[0]


def load_suite(suite):
    recs = {}
    for p in sorted(glob.glob(str(RES / "eval" / suite / "*.json"))):
        r = CM.read_json(p)
        recs[(r["model"], r["mode"])] = r
    mats = {}
    for (name, mode), r in recs.items():
        mats.setdefault(mode, {}).setdefault(cell_of(name), {})[r["condition"]["seed"]] = \
            np.asarray(r["eval"]["per_episode_success"], float)
    out = {m: {c: np.stack([d[s] for s in sorted(d)]) for c, d in cs.items()}
           for m, cs in mats.items()}
    return recs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_pdf", action="store_true")
    a = ap.parse_args()
    for d in (FIGS, TABS, REPORT):
        d.mkdir(parents=True, exist_ok=True)
    # start from empty output folders so no stale figure/table survives
    for d, pat in ((FIGS, "*.png"), (TABS, "*.tex"), (REPORT / "figures", "*.png"),
                   (REPORT / "tables", "*.tex")):
        for f in d.glob(pat):
            f.unlink()
    M = RP.Macros()
    prov = []

    # ------------------------------------------------------------ inputs
    env_info = CM.read_json(RES / "environment.json")
    teacher = CM.read_json(RES / "teacher" / "teacher_report.json")
    t_info = teacher["teacher_info"]
    hist_t = CM.read_json(RES / "historical_v1" / "teacher_noise_eval_v1.json")
    fa = CM.read_json(RES / "failure_analysis.json")
    diag = CM.read_json(RES / "diag" / "diag_student.json")
    replay = CM.read_json(RES / "diag" / "diag_replay.json")
    proto = CM.read_json(CFG / "fresh_protocol.json")
    recA, A = load_suite("A")
    recB, B = load_suite("B")
    assert len(recA) == 42, f"suite A has {len(recA)} of 42 evaluations"
    assert len(recB) == 6, f"suite B has {len(recB)} of 6 evaluations"

    # ------------------------------------------------------------ statistics
    cells = {m: {c: ST.cell_summary(A[m][c]) for c in CELLS} for m in MODES}
    contrasts = {m: {k: {**ST.paired_contrast(A[m][ca], A[m][cb]), "label": lbl,
                         "a": ca, "b": cb}
                     for k, lbl, ca, cb in CONTRASTS} for m in MODES}
    mode_contrast = {c: ST.paired_contrast(A["ensemble"][c], A["replan1"][c]) for c in CELLS}
    fresh = {c: ST.cell_summary(B["replan1"][c]) for c in ("act_clean_n100", "act_noise0.5_n100")}
    fresh_ct = ST.paired_contrast(B["replan1"]["act_clean_n100"], B["replan1"]["act_noise0.5_n100"])
    confirmed = fresh_ct["ci_ic"][0] > 0
    # historical v1 numbers (different, unrecoverable checkpoints)
    hist = {m: {} for m in MODES}
    for p in sorted(glob.glob(str(RES / "historical_v1" / "conditions" / "*.json"))):
        r = CM.read_json(p)
        for m, key in (("ensemble", "eval"), ("replan1", "eval_chunk1")):
            hist[m].setdefault(cell_of(r["name"]), []).append(
                float(np.mean(r[key]["per_episode_success"])))
    changed = {m: {c: sum(recA[(f"{c}_s{s}", m)]["eval"]["n_changed_vs_v1_same_rollout"]
                          for s in (0, 1, 2)) for c in CELLS} for m in MODES}
    term = {m: {c: float(np.mean([recA[(f"{c}_s{s}", m)]["eval"]["terminated_frac"]
                                  for s in (0, 1, 2)])) for c in CELLS} for m in MODES}
    obs_dev = max(r["eval"]["obs_vs_sim_dist_maxdev"] for r in recA.values())
    summary = {"suite_A": {"cells": cells, "contrasts": contrasts,
                           "replan_minus_ensemble": mode_contrast,
                           "changed_vs_v1_same_rollout": changed,
                           "terminated_frac": term, "historical_v1_per_seed": hist},
               "suite_B": {"cells": fresh, "contrast_noisy_minus_clean": fresh_ct,
                           "confirmed_by_frozen_rule": bool(confirmed)},
               "notes": ST.__doc__}
    CM.write_json(RES / "summary.json", summary)
    with open(RES / "summary.csv", "w") as f:
        f.write("suite,mode,condition,seed0,seed1,seed2,mean,ci_ic_lo,ci_ic_hi,"
                "ci_2way_lo,ci_2way_hi,n_success,n_episodes\n")
        for m in MODES:
            for c in CELLS:
                s = cells[m][c]
                f.write(f"A,{m},{c}," + ",".join(f"{v:.2f}" for v in s["per_seed"])
                        + f",{s['mean']:.4f},{s['ci_ic'][0]:.4f},{s['ci_ic'][1]:.4f},"
                        f"{s['ci_2way'][0]:.4f},{s['ci_2way'][1]:.4f},"
                        f"{s['n_success_total']},{s['n_episodes_total']}\n")
        for c, s in fresh.items():
            f.write(f"B,replan1,{c}," + ",".join(f"{v:.2f}" for v in s["per_seed"])
                    + f",{s['mean']:.4f},{s['ci_ic'][0]:.4f},{s['ci_ic'][1]:.4f},"
                    f"{s['ci_2way'][0]:.4f},{s['ci_2way'][1]:.4f},"
                    f"{s['n_success_total']},{s['n_episodes_total']}\n")
    prov += [("results/summary.json, results/summary.csv", "results/eval/A/*.json, results/eval/B/*.json")]

    # ------------------------------------------------------------ figures
    src_A = sorted(glob.glob(str(RES / "eval" / "A" / "*.json")))
    tsig = {float(k): v["success_rate"] for k, v in teacher["eval"].items()}
    figs = {
        "ppo_curve": FG.fig_ppo_curve(str(CFG / "teacher" / "ppo_progress.json"),
                                      str(FIGS / "ppo_curve.png"),
                                      str(RES / "historical_v1" / "ppo_progress.json")),
        "demos": FG.fig_demo_stats(str(CM.DATA / "demos_sigma0.npz"),
                                   teacher_spec(), str(FIGS / "demos.png")),
        "loss": FG.fig_loss_curves(sorted(glob.glob(str(CFG / "trainlogs" / "*.json"))),
                                   str(FIGS / "loss.png")),
        "quantity": FG.fig_quantity(cells, tsig[0.0], str(FIGS / "quantity.png"), src_A),
        "noise": FG.fig_noise(cells, tsig, {float(k): v for k, v in hist_t["success_rate"].items()},
                              str(FIGS / "noise.png"), src_A),
        "algorithms": FG.fig_algorithms(cells, str(FIGS / "algorithms.png"), src_A),
        "fresh": FG.fig_fresh(fresh, str(FIGS / "fresh.png"),
                              sorted(glob.glob(str(RES / "eval" / "B" / "*.json")))),
    }
    tr = {"dist": [], "success": [], "n_valid": []}
    for s in (0, 1, 2):
        z = np.load(RES / "episodes" / "A" / f"act_clean_n100_s{s}__replan1.npz")
        tr["dist"].append(z["raw_dist"]); tr["success"].append(z["ep_success"])
        tr["n_valid"].append(z["ep_n_valid_states"])
    tr = {k: np.concatenate(v) for k, v in tr.items()}
    tr["success"] = tr["success"].astype(bool)
    figs["failure"] = FG.fig_failure(fa, tr, str(FIGS / "failure.png"),
                                     [str(RES / "failure_analysis.json")])
    vids = [str(CM.VIDEOS / "expert_demo0_success.mp4")] + \
        [str(CM.PROJ / v["path"]) for v in fa.get("videos", []) if v.get("path")]
    figs["frames"] = FG.fig_demo_frames(vids, str(FIGS / "frames.png"))
    prov += [(f"results/figures/{os.path.basename(v)}", src)
             for v, src in ((figs["ppo_curve"], "configs/teacher/ppo_progress.json, results/historical_v1/ppo_progress.json"),
                            (figs["demos"], "data/demos_sigma0.npz"),
                            (figs["loss"], "configs/trainlogs/*.json"),
                            (figs["quantity"], "results/eval/A/*.json"),
                            (figs["noise"], "results/eval/A/*.json, results/teacher/teacher_report.json"),
                            (figs["algorithms"], "results/eval/A/*.json"),
                            (figs["fresh"], "results/eval/B/*.json"),
                            (figs["failure"], "results/failure_analysis.json, results/episodes/A/act_clean_n100_s*__replan1.npz"),
                            (figs["frames"], "videos/*.mp4")) if v]

    # ------------------------------------------------------------ tables
    rows, mids = [], []
    for i, c in enumerate(CELLS):
        if i in (3, 6):
            mids.append(len(rows))
        r = [CELLS[c][4]]
        for m in MODES:
            s = cells[m][c]
            r += [" / ".join(f"{100*v:.0f}" for v in s["per_seed"]),
                  rf"\textbf{{{RP.pct(s['mean'])}}}", RP.ci(s["ci_ic"]), RP.ci(s["ci_2way"])]
        rows.append(r)
    RP.write_table(str(TABS / "main.tex"), "l" + "rrrr" * 2,
                   ["Condition", r"\multicolumn{4}{c}{Temporal ensembling}",
                    r"\multicolumn{4}{c}{Replan every step}"][:1] +
                   ["Seeds 0/1/2", "Mean", "IC 95\\% CI", "2-way CI"] * 2,
                   rows,
                   r"Suite A: task success (\%) of every trained student on the original "
                   r"100 held-out initial conditions (ICs), corrected protocol v2. Left block: "
                   r"temporal ensembling (historical headline mode); right block: replanning "
                   r"every step. `Mean' averages the three training seeds. `IC 95\% CI' resamples "
                   r"the shared ICs with the trained models fixed; `2-way CI' also resamples the "
                   r"training seeds (crude with three seeds).",
                   "tab:main", size=r"\scriptsize", midrules=mids, star=True, fit=True)
    rows = []
    for c, (grp, algo, nd, sig, lbl) in CELLS.items():
        k = f"{sig:g}"
        e = teacher["eval"][k]
        ho = hist_t["success_rate"][k]
        rows.append([lbl, str(nd), f"{RP.pct(e['success_rate'], 0)} ({RP.pct(ho, 0)})",
                     rf"{RP.pct(cells['ensemble'][c]['mean'])} {RP.ci(cells['ensemble'][c]['ci_ic'])}",
                     rf"{RP.pct(cells['replan1'][c]['mean'])} {RP.ci(cells['replan1'][c]['ci_ic'])}",
                     "3 $\\times$ 100"])
    RP.write_table(str(TABS / "required.tex"), "lrrrrr",
                   ["Condition", "\\#Demos", "Teacher succ.\\ (orig.)",
                    "Student, ensembling", "Student, replan", "Eval.\\ episodes"], rows,
                   r"The table required by Part 4 (success in \%, suite A, protocol v2). "
                   r"Teacher: the retrained teacher at the action-noise level of that "
                   r"condition's data on the same 100 ICs; in parentheses the original "
                   r"(data-collecting) teacher's archived v1 result. Student: mean of 3 "
                   r"training seeds with the IC-bootstrap 95\% CI; each seed is evaluated on "
                   r"the same 100 held-out ICs (300 episodes per condition, not 300 "
                   r"independent ICs).", "tab:required", size=r"\scriptsize", midrules=[3, 6], fit=True)
    rows = []
    for k, lbl, ca, cb in CONTRASTS:
        r = [lbl]
        for m in MODES:
            c = contrasts[m][k]
            r += [f"{100*c['delta']:+.1f}", RP.ci(c["ci_ic"]), RP.ci(c["ci_2way"]),
                  ptab(c["p_signflip"])]
        rows.append(r)
    RP.write_table(str(TABS / "contrasts.tex"), "l" + "rrrr" * 2,
                   ["Contrast (B$-$A)"] + [r"$\Delta$ (pp)", "IC CI", "2-way CI", "$p$"] * 2,
                   rows,
                   r"Suite A paired contrasts in success (percentage points), first block "
                   r"temporal ensembling, second block replanning every step. Both arms are "
                   r"evaluated on the same 100 ICs, so the IC bootstrap resamples ICs jointly; "
                   r"$p$ is a sign-flip test of the per-IC difference of seed-averaged success "
                   r"(conditional on the trained models). Multiple contrasts are reported "
                   r"without correction.", "tab:contrasts", size=r"\scriptsize", star=True, fit=True)
    rows = []
    for c in ("act_clean_n100", "act_noise0.5_n100"):
        s = fresh[c]
        rows.append([CELLS[c][4]] + [f"{int(round(100*v))}/100" for v in s["per_seed"]]
                    + [rf"\textbf{{{RP.pct(s['mean'])}}}", RP.ci(s["ci_ic"]), RP.ci(s["ci_2way"])])
    rows.append([r"$\sigma{=}0.5$ $-$ clean", *[f"{100*v:+.0f}" for v in fresh_ct["per_seed_index_delta"]],
                 rf"\textbf{{{100*fresh_ct['delta']:+.1f}}}",
                 f"[{100*fresh_ct['ci_ic'][0]:+.1f}, {100*fresh_ct['ci_ic'][1]:+.1f}]",
                 f"[{100*fresh_ct['ci_2way'][0]:+.1f}, {100*fresh_ct['ci_2way'][1]:+.1f}]"])
    RP.write_table(str(TABS / "fresh.tex"), "lrrrrrr",
                   ["Suite B (replan)", "Seed 0", "Seed 1", "Seed 2", "Mean", "IC 95\\% CI",
                    "2-way CI"], rows,
                   r"Suite B, the frozen confirmation: the same six checkpoints on 100 fresh "
                   r"initial conditions (\texttt{PRNGKey(\freshBase)}), replanning every step, "
                   r"600 episodes. Last row: noisy minus clean, per seed index and for the mean "
                   r"(percentage points); sign-flip $p\freshP$.", "tab:fresh",
                   size=r"\scriptsize", midrules=[2])
    rows = []
    for c in CELLS:
        r = [CELLS[c][4]]
        for m in MODES:
            hv = hist[m].get(c, [])
            r += [RP.pct(np.mean(hv)) if hv else "--", RP.pct(cells[m][c]["mean"]),
                  str(changed[m][c])]
        rows.append(r)
    RP.write_table(str(TABS / "oldnew.tex"), "l" + "rrr" * 2,
                   ["Condition"] + ["v1 (orig.)", "v2 (new)", "flips"] * 2, rows,
                   r"Historical (protocol v1, original checkpoints) versus corrected (v2, "
                   r"retrained checkpoints) mean success (\%), ensembling then replanning. The "
                   r"checkpoints differ, so the difference is NOT attributable to the evaluation "
                   r"fix alone. `flips' counts, on the new rollouts, the episodes (of 300) whose "
                   r"outcome changes between the v1 and v2 rules -- the part that IS due to the fix.",
                   "tab:oldnew", size=r"\scriptsize")
    rows = []
    for k in ("0", "0.1", "0.3", "0.5", "1", "2"):
        e = teacher["eval"][k]
        dset = teacher["datasets"][k]
        hv = hist_t["success_rate"].get(f"{float(k):g}", hist_t["success_rate"].get(k))
        rows.append([k, RP.pct(hv, 0) if hv is not None else "--",
                     f"{e['n_success']}/{e['n_episodes']}", RP.pct(e["terminated_frac"], 0),
                     f"{dset['stored_success']}/{dset['n_episodes']}",
                     str(dset["episodes_terminated_early"]),
                     f"{dset['new_teacher_action_residual_mean_abs']:.3f}"])
    RP.write_table(str(TABS / "teacher.tex"), "rrrrrrr",
                   [r"$\sigma$", r"Orig.\ teacher", "Retrained", r"Term.\,\%",
                    "Dataset succ.", r"Data term.", r"$|a_{\mathrm{data}}-a_{\mathrm{new}}|$"],
                   rows,
                   r"Teacher provenance. `Orig.\ teacher': success (\%) of the ORIGINAL teacher "
                   r"that collected the datasets (archived, protocol v1). `Retrained': the "
                   r"retrained teacher on the same 100 held-out ICs (v2) and its share of "
                   r"terminated episodes. `Dataset succ.': stored success labels of the committed "
                   r"100-episode datasets (identical under v2). `Data term.': dataset episodes "
                   r"truncated at termination. Last column: mean absolute difference between "
                   r"the stored actions and the retrained teacher's clipped action at the same "
                   r"states.", "tab:teacher", size=r"\scriptsize")
    rows = []
    for k in ("0", "0.1", "0.3", "0.5", "1", "2"):
        dset = teacher["datasets"][k]
        z = np.load(CM.DATA / f"demos_sigma{k}.npz")
        ends = z["episode_ends"]; L = np.diff(np.r_[0, ends])
        used = {"0": "quantity + flow", "0.5": "noise", "1": "noise", "2": "noise"}.get(k, "not used (preliminary)")
        rows.append([k, str(dset["n_episodes"]), f"{dset['n_transitions']:,}",
                     f"{L.mean():.1f}", str(dset["episodes_terminated_early"]),
                     f"{dset['stored_success']}", "success only" if dset["filtered_to_success"] else "none",
                     used])
    RP.write_table(str(TABS / "data.tex"), "rrrrrrll",
                   [r"$\sigma$", "Eps.", "Transitions", "Mean len.", "Term.", "Succ.",
                    "Filter", "Used by"], rows,
                   r"Committed demonstration datasets (collected by the original teacher). The "
                   r"clean set keeps successful episodes only (the first 100 of 512 collected "
                   r"all succeeded, so the filter removed none); noisy sets keep the first 100 "
                   r"episodes including failures. $\sigma{=}0.1, 0.3$ were a preliminary sweep "
                   r"that barely degraded the teacher and are not used by any student.",
                   "tab:data", size=r"\scriptsize")
    rows = []
    for c in ("act_clean_n100", "act_noise0.5_n100"):
        for m in MODES:
            t = fa["taxonomy"][f"{c}__{m}"]
            rows.append([CELLS[c][4], MODE_NAME[m], str(t["n_success"])] +
                        [str(t["pooled"][k]) for k in fa["category_order"]])
    RP.write_table(str(TABS / "taxonomy.tex"), "llrrrrrr",
                   ["Model", "Mode", "Succ.", "Term.", r"No touch", r"No lift",
                    r"Lifted, far", r"Short hold"], rows,
                   r"Mutually exclusive episode outcomes, pooled over 3 seeds $\times$ 100 ICs "
                   r"(suite A). Decision order: success; terminated early; never touched the box "
                   r"(no finger-pad contact); touched but never lifted it $\geq$2\,cm; lifted "
                   r"but never within 5\,cm of the target; within 5\,cm but held $<$10 steps.",
                   "tab:taxonomy", size=r"\scriptsize")
    rows = []
    for s in (0, 1, 2):
        d = diag[f"act_clean_n100_s{s}"]
        rows.append([f"seed {s}",
                     f"{d['d2_train_inits_ensemble']['n_success']} / {d['d2_test_inits_ensemble']['n_success']}",
                     f"{d['d2_train_inits_replan1']['n_success']} / {d['d2_test_inits_replan1']['n_success']}",
                     f"{d['d3_train_states_mae']:.3f}", f"{d['d3_teacher_visited_mae']:.3f}",
                     f"{d['d3_student_visited_mae']:.3f}",
                     f"{d['d4_ensemble_oldest']['n_success']}", f"{d['d4_chunk_full']['n_success']}",
                     f"{d['d4_chunk5']['n_success']}"])
    RP.write_table(str(TABS / "diag.tex"), "lrrrrrrrr",
                   ["Baseline", "Ens.\\ train/test", "Replan train/test", "MAE data",
                    "MAE teacher", "MAE student", "Ens.\\ (ACT order)", "Open loop", "Chunk 5"],
                   rows,
                   r"Separate diagnostic runs on the three baseline models (ACT, 100 clean). "
                   r"Columns 2--3: successes out of 100 on the ICs the demonstrations came from "
                   r"vs.\ the held-out ICs. MAE: mean absolute error of the first predicted "
                   r"action on the stored training states (vs.\ stored actions) and on states "
                   r"visited by the retrained teacher or by the student (vs.\ the retrained "
                   r"teacher; its own disagreement with the stored actions on training states "
                   r"is \diagTeacherFloor). Last three: successes of other inference variants.",
                   "tab:diag", size=r"\scriptsize", star=True, fit=True)
    rows = [[CM_esc(s), CM_esc(src)] for s, src in prov] + [
        [CM_esc("results/tables/*.tex"), CM_esc("results/summary.json, results/teacher/teacher_report.json, results/failure_analysis.json, results/diag/*.json")],
        [CM_esc("report/numbers.tex"), "all of the above, " + CM_esc("results/environment.json, configs/*")]]
    RP.write_table(str(TABS / "prov.tex"), r"p{0.36\linewidth}p{0.56\linewidth}",
                   ["Artifact", "Produced from"], rows,
                   r"Provenance of every figure and table (all built by \texttt{scripts/stage3\_report.py}).",
                   "tab:prov", size=r"\scriptsize")

    # ------------------------------------------------------------ macros
    sp = teacher_spec()
    M.set("obsDim", sp["obs_dim"]); M.set("actDim", sp["act_dim"])
    M.set("epLen", sp["episode_length"]); M.set("ctrlDt", sp["ctrl_dt"])
    M.set("chunkSize", TRAIN["chunk"]); M.set("trainSteps", f"{TRAIN['steps']:,}")
    M.set("batchSize", TRAIN["batch_size"]); M.set("learnRate", "$10^{-4}$")
    for m, tag in (("ensemble", "Ens"), ("replan1", "Rep")):
        for c in CELLS:
            s = cells[m][c]
            base = f"{c.replace('_', '')}{tag}"
            M.pct(base + "Mean", s["mean"]); M.pct(base + "Lo", s["ci_ic"][0])
            M.pct(base + "Hi", s["ci_ic"][1]); M.pct(base + "TwoLo", s["ci_2way"][0])
            M.pct(base + "TwoHi", s["ci_2way"][1])
            for i, v in enumerate(s["per_seed"]):
                M.pct(base + f"Seed{i}", v, nd=0)
            M.pct(base + "Term", term[m][c], nd=1)
        for k, _, _, _ in CONTRASTS:
            c = contrasts[m][k]
            base = f"cmp{k}{tag}"
            M.set(base + "Delta", f"{100*c['delta']:+.1f}")
            M.set(base + "Lo", f"{100*c['ci_ic'][0]:+.1f}")
            M.set(base + "Hi", f"{100*c['ci_ic'][1]:+.1f}")
            M.set(base + "TwoLo", f"{100*c['ci_2way'][0]:+.1f}")
            M.set(base + "TwoHi", f"{100*c['ci_2way'][1]:+.1f}")
            M.set(base + "P", prel(c["p_signflip"]))
    for c in CELLS:
        mc = mode_contrast[c]
        base = f"mode{c.replace('_', '')}"
        M.set(base + "Delta", f"{100*mc['delta']:+.1f}")
        M.set(base + "Lo", f"{100*mc['ci_ic'][0]:+.1f}")
        M.set(base + "Hi", f"{100*mc['ci_ic'][1]:+.1f}")
    for c, s in fresh.items():
        base = f"fresh{c.replace('_', '')}"
        M.pct(base + "Mean", s["mean"]); M.pct(base + "Lo", s["ci_ic"][0])
        M.pct(base + "Hi", s["ci_ic"][1]); M.pct(base + "TwoLo", s["ci_2way"][0])
        M.pct(base + "TwoHi", s["ci_2way"][1])
        for i, v in enumerate(s["per_seed"]):
            M.pct(base + f"Seed{i}", v, nd=0)
    M.set("freshDelta", f"{100*fresh_ct['delta']:+.1f}")
    M.set("freshLo", f"{100*fresh_ct['ci_ic'][0]:+.1f}")
    M.set("freshHi", f"{100*fresh_ct['ci_ic'][1]:+.1f}")
    M.set("freshTwoLo", f"{100*fresh_ct['ci_2way'][0]:+.1f}")
    M.set("freshTwoHi", f"{100*fresh_ct['ci_2way'][1]:+.1f}")
    M.set("freshP", prel(fresh_ct["p_signflip"]))
    M.set("freshMcB", fresh_ct["mcnemar_majority_b01"]); M.set("freshMcC", fresh_ct["mcnemar_majority_b10"])
    M.set("freshMcP", prel(fresh_ct["mcnemar_majority_p"]))
    for i, v in enumerate(fresh_ct["per_seed_index_delta"]):
        M.set(f"freshSeedDelta{i}", f"{100*v:+.0f}")
    M.set("freshVerdict", "confirmed" if confirmed else "not confirmed")
    M.set("freshBase", proto["eval_base_seed"]); M.set("freshSha", proto["reset_keys_sha256"][:12])
    M.set("freshFrozen", proto["frozen_utc"][:10])
    M.set("freshNearest", f"{1000*min(proto['nearest_initial_config_distance_m'].values()):.1f}")
    # teacher
    for k, e in teacher["eval"].items():
        M.set(f"tNewSucc{k}", e["n_success"]); M.pct(f"tNewTerm{k}", e["terminated_frac"], nd=0)
    for k, v in hist_t["success_rate"].items():
        M.pct(f"tOldSucc{k}", v, nd=0)
    M.set("teacherSteps", f"{int(t_info['num_timesteps_requested']):,}")
    M.set("teacherStepsReached", f"{int(t_info['num_timesteps_reached']):,}")
    M.set("teacherWallMin", f"{t_info['wall_s']/60:.0f}")
    M.set("teacherReloadDev", sci(t_info.get("reload_vs_memory_action_maxdev", 0)))
    d0 = teacher["datasets"]["0"]
    M.set("dataResidClean", f"{d0['new_teacher_action_residual_mean_abs']:.3f}")
    M.set("dataStepZeroDev", f"{d0['new_teacher_step0_action_maxdev']:.2f}")
    M.set("dataResetDev", sci(max(v["reset_obs_vs_recorded_keys_maxdev"] for v in teacher["datasets"].values())))
    for k, v in teacher["datasets"].items():
        M.set(f"dataTrans{k}", f"{v['n_transitions']:,}"); M.set(f"dataTerm{k}", v["episodes_terminated_early"])
        M.set(f"dataSucc{k}", v["stored_success"])
        M.pct(f"dataPool{k}", v["raw_pool_success_rate_v1"] or 0, nd=1)
    # failure / taxonomy
    tv = fa["threshold_validation"]
    M.set("valN", tv["n_successful_episodes_checked"])
    M.pct("valContact", tv["frac_with_contact"]); M.pct("valLift", tv["frac_lifted"])
    M.pct("valTwelve", tv["frac_within_12mm"], nd=0)
    M.set("valMinLiftCm", f"{100*tv['max_lift_m_percentiles']['p0']:.1f}")
    M.set("valGripMedMm", f"{1000*tv['min_grip_m_percentiles']['p50']:.0f}")
    M.set("valGripMaxMm", f"{1000*tv['min_grip_m_percentiles']['p100']:.0f}")
    hv = fa.get("historical_v1_check", {})
    if hv:
        M.set("histSucc", hv["v1_successes"]); M.set("histLabelled", hv["v1_episodes_labelled_success"])
        M.set("histMislabelled", hv["v1_successes_with_failure_labels"])
        M.set("histReached", hv["v1_reached_box_12mm"])
        M.set("histNever", hv["v1_taxonomy"].get("never reached the box", 0))
    names = ["Term", "NoTouch", "NoLift", "LiftedFar", "ShortHold"]
    for c in ("act_clean_n100", "act_noise0.5_n100"):
        for m, tag in (("ensemble", "Ens"), ("replan1", "Rep")):
            t = fa["taxonomy"][f"{c}__{m}"]
            base = f"tax{c.replace('_', '')}{tag}"
            M.set(base + "Succ", t["n_success"]); M.set(base + "Fail", t["n_failures"])
            for nm, cat in zip(names, fa["category_order"]):
                M.set(base + nm, t["pooled"][cat])
    vids = [v for v in fa.get("videos", []) if v.get("path")]
    M.set("nVideos", len(vids))
    ag = {v["model"]: v["video_rollout_agreement_with_suiteA"] for v in fa.get("videos", [])}
    M.set("videoAgreeMin", min(ag.values())); M.set("videoAgreeMax", max(ag.values()))
    # diagnostics
    M.set("diagTeacherFloor", f"{diag['d3_new_teacher_vs_stored_mae']:.3f}")
    for s in (0, 1, 2):
        d = diag[f"act_clean_n100_s{s}"]
        M.set(f"diagTrainEns{s}", d["d2_train_inits_ensemble"]["n_success"])
        M.set(f"diagTestEns{s}", d["d2_test_inits_ensemble"]["n_success"])
        M.set(f"diagTrainRep{s}", d["d2_train_inits_replan1"]["n_success"])
        M.set(f"diagTestRep{s}", d["d2_test_inits_replan1"]["n_success"])
        M.set(f"diagMaeTrain{s}", f"{d['d3_train_states_mae']:.3f}")
        M.set(f"diagMaeTeacher{s}", f"{d['d3_teacher_visited_mae']:.3f}")
        M.set(f"diagMaeStudent{s}", f"{d['d3_student_visited_mae']:.3f}")
        M.set(f"diagMaeRatio{s}", f"{d['d3_ratio_student_over_teacher_visited']:.1f}")
        M.set(f"diagOldest{s}", d["d4_ensemble_oldest"]["n_success"])
        M.set(f"diagEnsZ{s}", d["d4_ensemble"]["n_success"])
    M.set("replayResetDev", sci(replay["reset_vs_stored_maxdev"]))
    M.set("replayFirstBig", ", ".join(str(x) for x in replay["first_exceed_1e3"]))
    M.set("obsSimDev", sci(obs_dev))
    # runtimes / environment
    ids = [CM.read_json(p) for p in sorted(glob.glob(str(CFG / "identities" / "*.json")))]
    act_w = [i["wall_s"] for i in ids if i["name"].startswith("act")]
    flow_w = [i["wall_s"] for i in ids if i["name"].startswith("flow")]
    M.set("trainMinAct", f"{np.mean(act_w)/60:.1f}"); M.set("trainMinFlow", f"{np.mean(flow_w)/60:.1f}")
    M.set("trainConcurrency", "eight")
    ev_w = [r["wall_s"] for r in recA.values()]
    M.set("evalSecMean", f"{np.mean(ev_w):.0f}")
    M.set("gpuName", env_info.get("gpu", "?")); M.set("hostName", env_info.get("host", "?"))
    for k, v in env_info["versions"].items():
        M.set("ver" + k.replace("-", ""), v)
    M.set("playgroundSha", env_info.get("playground_commit", "?")[:12])
    M.set("menagerieSha", env_info.get("menagerie_commit", "?")[:12])
    M.set("nParamsAct", f"{recA[('act_clean_n100_s0', 'ensemble')]['n_params']/1e6:.1f}")
    M.set("nParamsFlow", f"{recA[('flow_clean_n100_s0', 'ensemble')]['n_params']/1e6:.1f}")
    bad = {k: v for k, v in M._m.items()
           if re.search(r"(?i)(?<![a-z])(nan|inf)(?![a-z])|None|\?", v)}
    if bad:
        print("ERROR: non-numeric macro values:", bad)
        sys.exit(3)
    (REPORT / "numbers.tex").write_text(M.tex())
    print(f"[report] wrote {len(M._m)} macros")

    # copy figures/tables next to the tex
    for sub, src in (("figures", FIGS), ("tables", TABS)):
        dst = REPORT / sub
        dst.mkdir(exist_ok=True)
        for f in src.iterdir():
            shutil.copy(f, dst / f.name)
    body = (REPORT / "report_template.tex").read_text()
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    for t in TABS.glob("*.tex"):
        used |= set(re.findall(r"\\([A-Za-z]+)", t.read_text()))
    missing = sorted(u for u in used if u not in M._m and u not in KNOWN_TEX)
    if missing:
        print("ERROR: undefined macros:", missing)
        sys.exit(2)
    for t in sorted(TABS.glob("*.tex")):
        txt = t.read_text()
        for pat in (r"\(tbd\)", r"(?<![A-Za-z])nan(?![A-Za-z])", r"\bNone\b",
                    r"/rightarrow", r"/sigma", r"(?<![A-Za-z\\])inf(?![A-Za-z])"):
            if re.search(pat, txt):
                print(f"ERROR: {t.name} contains {pat}")
                sys.exit(3)
    (REPORT / "report.tex").write_text(body)
    if not a.no_pdf:
        if compile_pdf() != 0:
            sys.exit(4)
        shutil.copy(REPORT / "report.pdf", CM.PROJ / "report.pdf")
        print("[report] copied report/report.pdf -> report.pdf (top level)")
    print("STAGE3_DONE")


def prel(p: float) -> str:
    """'=<p>' or '<10^{-4}' for use as $p\\macro$ in math mode."""
    if p < 1e-4:
        return "<10^{-4}"
    if p < 0.001:
        e = int(np.floor(np.log10(p)))
        return f"={p / 10**e:.1f}\\times10^{{{e}}}"
    return f"={p:.2g}"


def ptab(p: float) -> str:
    return "$" + prel(p).lstrip("=") + "$"


def sci(x: float) -> str:
    """LaTeX for a small non-negative number: '0' or 'a \\times 10^{b}'."""
    x = float(x)
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x / 10**e:.1f}\\times10^{{{e}}}$"


def CM_esc(s):
    """Comma-separated file paths as breakable, unescaped \\nolinkurl items."""
    return ", ".join(r"\nolinkurl{" + p.strip() + "}" for p in s.split(","))


def teacher_spec():
    return {"obs_dim": 66, "act_dim": 8, "episode_length": 150, "ctrl_dt": 0.02,
            "nq": 16, "nv": 15, "target_rel_slice": [49, 52],
            "gripper_slice": [31, 34], "box_rel_slice": [46, 49]}


def compile_pdf() -> int:
    eng = os.environ.get("HW2_LATEX", "pdflatex")
    tbin = os.environ.get("HW2_TEX_BIN", "")
    env = dict(os.environ)
    if tbin:
        env["PATH"] = tbin + os.pathsep + env.get("PATH", "")
    pdf = REPORT / "report.pdf"
    if pdf.exists():
        pdf.unlink()
    if eng == "tectonic":
        r = subprocess.run(["tectonic", "report.tex"], cwd=REPORT, env=env,
                           capture_output=True, text=True)
        log = r.stdout + r.stderr
    else:
        log = ""
        for _ in range(3):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                "report.tex"], cwd=REPORT, env=env, capture_output=True, text=True)
            log = r.stdout + r.stderr
            if r.returncode != 0:
                break
    (REPORT / "report.build.log").write_text(log)
    if r.returncode != 0 or not pdf.exists():
        print("ERROR: LaTeX failed; see report/report.build.log")
        print("\n".join(log.splitlines()[-25:]))
        return 1
    if "undefined references" in log or "Reference `" in log:
        print("ERROR: undefined references; see report/report.build.log")
        return 1
    txt = pdf_text(pdf, env)
    if txt is None:
        print("ERROR: could not extract text from report.pdf for verification")
        return 1
    (REPORT / "report.txt").write_text(txt)
    for pat in (r"\?\?", r"\(tbd\)", r"/rightarrow", r"/sigma"):
        if re.search(pat, txt):
            print(f"ERROR: report.pdf text contains {pat!r}")
            return 1
    print(f"[report] report.pdf OK ({pdf.stat().st_size/1024:.0f} KB, {len(txt):,} chars)")
    return 0


def pdf_text(pdf, env):
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages)
    except Exception:
        pass
    exe = shutil.which("pdftotext", path=env.get("PATH"))
    if exe:
        out = pdf.with_suffix(".txt")
        subprocess.run([exe, "-layout", str(pdf), str(out)], capture_output=True)
        if out.exists():
            return out.read_text(errors="ignore")
    return None


KNOWN_TEX = set("""documentclass usepackage begin end title author date maketitle section
subsection subsubsection paragraph textbf textit texttt emph item label ref cite caption
centering includegraphics toprule midrule bottomrule hline footnotesize small scriptsize
large Large newcommand input clearpage newpage noindent quad qquad times approx pm sigma
alpha beta rightarrow leftarrow le ge neq cdot sim mathrm mathbf text frac sqrt sum
linewidth textwidth hfill vspace hspace url href tabular table figure itemize enumerate
description footnote percent left right langle rangle textcolor geometry hypersetup
setlength renewcommand arraystretch multicolumn raggedright par nolinkurl pi mu epsilon
mathcal mathbb leq geq ll gg infty dots ldots cdots in Delta lambda gamma delta theta phi
tau rho eta kappa varepsilon smallskip medskip bigskip textsc underline to propto equiv
min max log exp mathit boldsymbol textendash textemdash thanks columnwidth baselineskip
parskip parindent textbackslash textasciitilde textasciicircum S bibitem thebibliography
newline linebreak cr tiny normalsize textsuperscript ensuremath mid colon vert lvert
rvert cline addlinespace makecell tabularnewline bf it rm tt sf emph footnotesize
captionsetup subcaption subfloat hyphenation sloppy allowbreak nobreak checkmark
textgreater textless textquotedbl leftmargin itemsep topsep labelsep nolinebreak
renewenvironment newenvironment fbox mbox makebox raisebox resizebox scalebox lfloor rfloor
subset subseteq adjustbox emergencystretch resizebox IfFileExists
""".split()) | {",", "%", "&", "#", "$", "_", "{", "}", "\\"}


if __name__ == "__main__":
    main()
