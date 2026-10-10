"""Stage 3 (CPU-only): aggregate results, build figures/tables, emit report.pdf."""
import argparse, glob, json, os, re, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.join(os.environ["PROJ"], "code"))
from hw2 import aggregate as AG
from hw2 import figures as FG
from hw2 import report as RP

PROJ = os.environ["PROJ"]
RES = os.path.join(PROJ, "results")
FIGS = os.path.join(RES, "figures")
TABS = os.path.join(RES, "tables")
REPORT_DIR = os.path.join(PROJ, "report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", default=os.path.join(PROJ, "runs", "teacher"))
    ap.add_argument("--no_pdf", action="store_true")
    a = ap.parse_args()

    for d in (FIGS, TABS, REPORT_DIR):
        os.makedirs(d, exist_ok=True)

    s1 = json.load(open(os.path.join(a.teacher_dir, "stage1_report.json")))
    spec = s1["spec"]
    teacher = {}
    for k, v in s1.items():
        if k.startswith("teacher_eval_sigma"):
            teacher[float(k.replace("teacher_eval_sigma", ""))] = v

    records = AG.load_conditions(os.path.join(RES, "conditions"))
    if not records:
        print("ERROR: no condition results found"); sys.exit(1)
    rows = AG.aggregate(records, "eval")
    rows_c1 = AG.aggregate(records, "eval_chunk1")
    AG.write_csv(rows, os.path.join(RES, "summary.csv"))
    if rows_c1:
        AG.write_csv(rows_c1, os.path.join(RES, "summary_chunk1.csv"))
    json.dump({"ensemble": rows, "chunk1": rows_c1},
              open(os.path.join(RES, "summary.json"), "w"), indent=2)

    # ---- paired comparisons (same held-out initial conditions) ----
    # Explicit short keys so the LaTeX macro names stay readable and stable.
    cmps, cmps_c1, labels = [], [], {}
    def add(key, lbl, ka, kb):
        c = AG.paired_compare(records, ka, kb, eval_key="eval")
        c["key"] = key; c["label"] = lbl; labels[key] = lbl
        cmps.append(c)
        if rows_c1:
            c2 = AG.paired_compare(records, ka, kb, eval_key="eval_chunk1")
            c2["key"] = key; c2["label"] = lbl
            cmps_c1.append(c2)
    add("QtyTwentyFiveToHundred", "25 demos $\\rightarrow$ 100 demos",
        ("quantity", "act", 25, 0.0), ("quantity", "act", 100, 0.0))
    add("QtyFiftyToHundred", "50 demos $\\rightarrow$ 100 demos",
        ("quantity", "act", 50, 0.0), ("quantity", "act", 100, 0.0))
    add("NoiseLow", "clean $\\rightarrow$ noise $\\sigma{=}0.5$",
        ("quantity", "act", 100, 0.0), ("noise", "act", 100, 0.5))
    add("NoiseMid", "clean $\\rightarrow$ noise $\\sigma{=}1$",
        ("quantity", "act", 100, 0.0), ("noise", "act", 100, 1.0))
    add("NoiseHigh", "clean $\\rightarrow$ noise $\\sigma{=}2$",
        ("quantity", "act", 100, 0.0), ("noise", "act", 100, 2.0))
    add("ActToFlow", "ACT $\\rightarrow$ flow matching",
        ("quantity", "act", 100, 0.0), ("algorithm", "flow", 100, 0.0))
    json.dump({"ensemble": cmps, "chunk1": cmps_c1},
              open(os.path.join(RES, "paired_comparisons.json"), "w"), indent=2)

    # ---- figures ----
    made = {}
    made["ppo"] = FG.fig_ppo_curve(os.path.join(a.teacher_dir, "ppo_progress.json"),
                                   os.path.join(FIGS, "ppo_curve.png"))
    clean = os.path.join(PROJ, "data", "demos_sigma0.npz")
    if os.path.exists(clean):
        made["demos"] = FG.fig_demo_stats(clean, spec, os.path.join(FIGS, "demos.png"))
    made["frames"] = FG.fig_demo_frames(os.path.join(PROJ, "videos"),
                                        os.path.join(FIGS, "demo_frames.png"))
    made["loss"] = FG.fig_loss_curves(os.path.join(PROJ, "runs", "conditions"),
                                      os.path.join(FIGS, "loss.png"))
    srcs = sorted(glob.glob(os.path.join(RES, "conditions", "*.json")))
    made["quantity"] = FG.fig_quantity(rows, teacher, os.path.join(FIGS, "quantity.png"), srcs)
    made["noise"] = FG.fig_noise(rows, teacher, os.path.join(FIGS, "noise.png"), srcs)
    made["algo"] = FG.fig_algorithms(rows, os.path.join(FIGS, "algorithms.png"), srcs)
    made["failure"] = FG.fig_failure(os.path.join(RES, "failure_analysis.json"),
                                     os.path.join(FIGS, "failure.png"))

    # ---- tables ----
    AG.main_table_tex(rows, teacher, os.path.join(TABS, "main.tex"))
    RP.dataset_table(s1["datasets"], os.path.join(TABS, "data.tex"))
    RP.comparison_table(cmps, {c["key"]: c["label"] for c in cmps if "key" in c},
                        os.path.join(TABS, "paired.tex"),
                        cmps_c1=cmps_c1 or None)
    if rows_c1:
        RP.mode_table(rows, rows_c1, os.path.join(TABS, "modes.tex"))

    _srcmap = {
        "ppo": "runs/teacher/ppo_progress.json",
        "demos": "data/demos_sigma0.npz",
        "frames": "videos/expert_demo*_frames.npz",
        "loss": "runs/conditions/*/train_log.json",
        "quantity": "results/conditions/*.json",
        "noise": "results/conditions/*.json",
        "algo": "results/conditions/*.json",
        "failure": "results/failure_analysis.json",
    }
    prov = [{"artifact": os.path.relpath(v, PROJ),
             "source": _srcmap.get(k, "results/conditions/*.json")}
            for k, v in made.items() if v]
    prov += [
        {"artifact": "results/tables/main.tex", "source": "results/conditions/*.json"},
        {"artifact": "results/tables/data.tex", "source": "runs/teacher/stage1_report.json"},
        {"artifact": "results/tables/paired.tex", "source": "results/paired_comparisons.json"},
        {"artifact": "results/tables/modes.tex", "source": "results/conditions/*.json"},
        {"artifact": "results/summary.csv", "source": "results/conditions/*.json"},
    ]
    RP.provenance_table(prov, os.path.join(TABS, "prov.tex"))

    # ---- macros ----
    M = RP.Macros()
    env_info = json.load(open(os.path.join(PROJ, "results", "environment.json"))) \
        if os.path.exists(os.path.join(PROJ, "results", "environment.json")) else {}
    M.set("obsDim", spec["obs_dim"]); M.set("actDim", spec["act_dim"])
    M.set("epLen", spec["episode_length"]); M.set("ctrlDt", spec["ctrl_dt"])
    M.set("successRadiusCm", f"{100*s1['success_radius']:.0f}")
    M.set("successHold", s1["success_hold"])
    M.set("successHoldSec", f"{s1['success_hold']*spec['ctrl_dt']:.2f}")
    M.set("teacherSteps", f"{int(float(s1['ppo_params']['num_timesteps'])):,}")
    M.set("teacherWallMin", f"{s1['teacher_wall_s']/60:.0f}")
    M.set("teacherNumEnvs", s1["ppo_params"]["num_envs"])
    for sig, v in teacher.items():
        tag = f"teacherSuccSigma{sig}"
        M.pct(tag, v["success_rate"])
        M.set(f"teacherRetSigma{sig}", f"{v['return_mean']:.0f}")
    for tag, d in s1["datasets"].items():
        s = d["summary"]
        M.set(f"data{tag}Eps", s["n_episodes"])
        M.set(f"data{tag}Trans", f"{s['n_transitions']:,}")
        M.set(f"data{tag}LenMean", f"{s['ep_len_mean']:.1f}")
        M.set(f"data{tag}LenStd", f"{s['ep_len_std']:.1f}")
        M.pct(f"data{tag}Succ", s["success_fraction"])
        M.pct(f"data{tag}PoolSucc", d["meta"].get("raw_pool_success_rate", 0.0))
    for r in rows:
        tag = f"res{r['group']}{r['algo']}N{r['n_demos']}S{r['noise']}"
        M.pct(tag + "Mean", r["success_pooled"])
        M.pct(tag + "Lo", r["ci95_lo"]); M.pct(tag + "Hi", r["ci95_hi"])
        M.set(tag + "NEval", r["n_eval_total"])
        M.set(tag + "NSeeds", r["n_seeds"])
        M.set(tag + "Params", f"{r['n_params']/1e6:.1f}")
    for c in cmps_c1:
        if "error" in c: continue
        tag = "cmpCOne" + c["key"]
        M.pct(tag + "Delta", c["delta"]); M.pct(tag + "Lo", c["ci95"][0])
        M.pct(tag + "Hi", c["ci95"][1]); M.set(tag + "P", f"{c['mcnemar_p']:.3g}")
        M.set(tag + "Sig", "yes" if c["significant_at_05"] else "no")
    for c in cmps:
        if "error" in c: continue
        tag = "cmp" + c["key"]
        M.pct(tag + "Delta", c["delta"]); M.pct(tag + "Lo", c["ci95"][0])
        M.pct(tag + "Hi", c["ci95"][1]); M.set(tag + "P", f"{c['mcnemar_p']:.3g}")
        M.set(tag + "Sig", "yes" if c["significant_at_05"] else "no")
    # ---- diagnostics (covariate shift, inference mode, determinism) ----
    dpath = os.path.join(RES, "diag_act_clean_n100_s0.json")
    if os.path.exists(dpath):
        dg = json.load(open(dpath))
        astd_tr = dg.get("D3_train_state_action_mae_in_std")
        if astd_tr is not None:
            M.set("diagMaeTrainStd", f"{astd_tr:.2f}")
        for k, mk in (("D3_train_state_action_mae", "diagMaeTrainRaw"),
                      ("D3_teacher_state_action_mae", "diagMaeTeacherRaw"),
                      ("D3_student_state_action_mae", "diagMaeStudentRaw")):
            if k in dg:
                M.set(mk, f"{dg[k]:.3f}")
        if "D3_mae_ratio_student_over_teacher" in dg:
            M.set("diagMaeRatio", f"{dg['D3_mae_ratio_student_over_teacher']:.1f}")
        for k, mk in (("D2_train_inits", "diagTrainInit"),
                      ("D2_test_inits", "diagTestInit")):
            if k in dg:
                M.pct(mk + "Succ", dg[k]["success_rate"])
                M.pct(mk + "Box", dg[k]["reached_box_frac"] or 0.0)
        for k, mk in (("D4_ensemble", "diagModeEns"),
                      ("D4_chunk_full", "diagModeChunkFull"),
                      ("D4_chunk_1", "diagModeChunkOne"),
                      ("D4_chunk_5", "diagModeChunkFive")):
            if k in dg:
                M.pct(mk + "Succ", dg[k]["success_rate"])
                M.pct(mk + "Box", dg[k]["reached_box_frac"] or 0.0)
    fpath = os.path.join(RES, "failure_analysis.json")
    if os.path.exists(fpath):
        fa = json.load(open(fpath))
        tax = fa.get("failure_taxonomy") or {}
        n_ev = int(fa.get("n_eval", 0))
        n_fail = int(sum(tax.values()))
        M.set("failNEval", n_ev)
        M.set("failNFail", n_fail)
        M.pct("failReachedBox", fa.get("reached_box_frac") or 0.0, nd=0)
        if tax:
            top = max(tax.items(), key=lambda kv: kv[1])
            M.set("failTopLabel", top[0])
            M.set("failTopCount", int(top[1]))
            M.pct("failTopShare", top[1] / max(n_fail, 1), nd=0)
            others = sorted(tax.items(), key=lambda kv: -kv[1])[1:]
            M.set("failOtherList", ", ".join(
                f"{k} ({v})" for k, v in others) or "none")
        sbr = fa.get("success_by_radius") or {}
        for k, v in sbr.items():
            # macro_name maps "." -> "Pt", so "0.05" becomes ZeroPtZeroFive
            M.pct("failSuccRadius" + k, v, nd=0)
    rpath = os.path.join(RES, "diag_replay.json")
    if os.path.exists(rpath):
        rp = json.load(open(rpath))
        M.set("diagResetDev", f"{rp.get('reset_vs_stored_maxdev', 0.0):.0e}")
        c0 = (rp.get("per_step_deviation") or [[0, 0]])[0]
        M.set("diagReplayStepOneDev", f"{(c0[1] if len(c0) > 1 else 0.0):.0e}")

    # chunk1 (re-planning) headline numbers, for the inference-mode discussion
    for r in rows_c1:
        tag = f"cone{r['group']}{r['algo']}N{r['n_demos']}S{r['noise']}"
        M.pct(tag + "Mean", r["success_pooled"])
        M.pct(tag + "Box", r["reached_box_frac"])

    r0 = next((r for r in rows if r["group"] == "quantity" and r["n_demos"] == 100), None)
    if r0:
        M.set("execMode", r0["exec_mode"])
        M.set("chunkSize", next((x["chunk"] for x in records), 30))
        M.set("trainSteps", f"{next((x['train_steps'] for x in records), 0):,}")
        M.set("batchSize", next((x["batch_size"] for x in records), 0))
        M.set("learnRate", next((x["lr"] for x in records), 0))
    for k, v in (env_info.get("versions") or {}).items():
        M.set("ver" + k.replace("-", ""), v)
    M.set("gpuName", env_info.get("gpu", "NVIDIA A100-SXM4-80GB"))
    M.set("hostName", env_info.get("host", "gc216"))
    M.set("playgroundSha", env_info.get("playground_commit", "unknown")[:12])
    M.set("nEvalPerSeed", rows[0]["n_eval_per_seed"])
    M.set("nSeeds", max(r["n_seeds"] for r in rows))
    M.set("successRadiusM", s1["success_radius"])
    open(os.path.join(REPORT_DIR, "numbers.tex"), "w").write(M.tex())
    print(f"[report] wrote {len(M._m)} macros")

    # copy figures/tables next to the tex so the pdf is buildable standalone
    subprocess.run(["bash", "-c",
        f"mkdir -p {REPORT_DIR}/figures {REPORT_DIR}/tables && "
        f"cp -f {FIGS}/*.png {REPORT_DIR}/figures/ 2>/dev/null; "
        f"cp -f {TABS}/*.tex {REPORT_DIR}/tables/ 2>/dev/null; true"], check=False)
    # ---- structural validation, then compile ----
    tmpl = os.path.join(REPORT_DIR, "report_template.tex")
    if not os.path.exists(tmpl):
        print(f"ERROR: missing {tmpl}"); sys.exit(1)
    body = open(tmpl).read()
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    defined = set(M._m)
    known_tex = _KNOWN_TEX
    missing = sorted(u for u in used if u not in defined and u not in known_tex)
    if missing:
        print("ERROR: report_template.tex uses macros that were not generated:")
        for u in missing:
            print("   \\" + u)
        print("\nGenerated macros available:")
        for u in sorted(defined):
            print("   \\" + u)
        sys.exit(2)
    print(f"[report] macro check OK ({len(used & defined)} generated macros used)")

    # sanity: no placeholder or NaN text leaked into generated tables
    # Word-boundary matching: a naive substring test flags "nan" inside
    # "Provenance" and "inf" inside "information".
    bad = []
    _markers = [r"\(tbd\)", r"\bTBD\b", r"(?<![A-Za-z])nan(?![A-Za-z])",
                r"\bNaN\b", r"\bNone\b", r"(?<![A-Za-z])inf(?![A-Za-z])",
                r"(?<![A-Za-z])-?inf(?![A-Za-z])"]
    for t in sorted(glob.glob(os.path.join(TABS, "*.tex"))):
        txt = open(t).read()
        for marker in _markers:
            m = re.search(marker, txt)
            if m:
                bad.append((os.path.basename(t), m.group(0)))
    if bad:
        print("ERROR: placeholder/NaN markers found in generated tables:", bad)
        sys.exit(3)
    print("[report] table marker check OK")

    open(os.path.join(REPORT_DIR, "report.tex"), "w").write(body)
    if not a.no_pdf:
        rc = _compile(REPORT_DIR)
        if rc != 0:
            sys.exit(4)
    print("STAGE3_DONE")


_KNOWN_TEX = {
    "documentclass","usepackage","begin","end","title","author","date","maketitle",
    "section","subsection","subsubsection","paragraph","textbf","textit","texttt",
    "emph","item","label","ref","cite","caption","centering","includegraphics",
    "toprule","midrule","bottomrule","hline","footnotesize","small","scriptsize",
    "large","Large","newcommand","input","clearpage","newpage","noindent","quad",
    "qquad","times","approx","pm","sigma","alpha","beta","rightarrow","leftarrow",
    "le","ge","neq","cdot","sim","mathrm","mathbf","text","frac","sqrt","sum",
    "linewidth","textwidth","hfill","vspace","hspace","bibliography","url","href",
    "tabular","table","figure","itemize","enumerate","description","verbatim",
    "subfigure","textsuperscript","and","today","footnote","percent","it","bf",
    "left","right","big","Big","langle","rangle","top","bf","tt","rm","sf",
    "textcolor","colorbox","definecolor","geometry","hypersetup","setlength",
    "renewcommand","arraystretch","multicolumn","multirow","raggedright","par",
    "lstset","lstinputlisting","lstlisting","href","nolinkurl","pi","mu","epsilon",
    "mathcal","mathbb","operatorname","leq","geq","ll","gg","infty","partial",
    "bigl","bigr","Bigl","Bigr","mid","colon","dots","ldots","cdots","in",
    "providecommand","ensuremath",
    "Delta","Gamma","Lambda","Omega","Sigma","Phi","Theta","lambda",
    "gamma","delta","theta","phi","psi","omega","tau","rho","eta","zeta",
    "kappa","nu","xi","chi","varepsilon","varphi","vartheta","lfloor",
    "rfloor","lceil","rceil","parskip","parindent","baselineskip",
    "smallskip","medskip","bigskip","centerline","textsc","underline",
    "to","mapsto","implies","iff","forall","exists","subset","subseteq",
    "cup","cap","setminus","emptyset","propto","equiv","simeq","cong",
    "prod","int","max","min","argmax","argmin","log","exp","sin","cos",
    "tanh","mathit","mathsf","mathtt","boldsymbol","textendash",
    "textemdash","st","nd","rd","th",
    ",","%","&","#","$","_","{","}","\\",
}


def _compile(d: str) -> int:
    """Compile with pdflatex twice (refs), then verify the PDF really updated."""
    tl = "/data/rtxw01/nbak/robot_intel/kztrgg/texlive2025/bin/x86_64-linux"
    env = dict(os.environ)
    env["PATH"] = tl + os.pathsep + env.get("PATH", "")
    pdf = os.path.join(d, "report.pdf")
    before = os.path.getmtime(pdf) if os.path.exists(pdf) else 0
    log = ""
    for i in range(2):
        r = subprocess.run([os.path.join(tl, "pdflatex"), "-interaction=nonstopmode",
                            "-halt-on-error", "report.tex"],
                           cwd=d, env=env, capture_output=True, text=True)
        log = r.stdout + r.stderr
    logpath = os.path.join(d, "report.pdflatex.log")
    open(logpath, "w").write(log)
    if "Undefined control sequence" in log:
        print("ERROR: pdflatex reported an undefined control sequence; see", logpath)
        for ln in log.splitlines():
            if "Undefined control sequence" in ln:
                print("   ", ln)
        return 1
    # Verify the artifact, not the absence of error strings.
    if not os.path.exists(pdf):
        print("ERROR: report.pdf was not produced; see", logpath); return 1
    if os.path.getmtime(pdf) <= before:
        print("ERROR: report.pdf did not update (stale build); see", logpath); return 1
    try:
        head = open(pdf, "rb").read(5)
        assert head == b"%PDF-", head
    except Exception as e:
        print("ERROR: report.pdf is not a valid PDF:", e); return 1
    npages = _pdf_pages(pdf)
    if npages <= 0:
        print("ERROR: could not determine the page count from the LaTeX log")
        return 1
    if "There were undefined references" in log:
        print("ERROR: pdflatex reports undefined references; see", logpath)
        return 1
    if not _verify_pdf_text(d, pdf):
        return 1
    print(f"[report] report.pdf OK ({os.path.getsize(pdf)/1024:.0f} KB, "
          f"{npages} pages)")
    return 0


def _pdf_pages(pdf: str) -> int:
    """Page count from pdflatex's own log (object streams are compressed, so
    grepping the PDF bytes for /Type/Page undercounts to zero)."""
    log = os.path.join(os.path.dirname(pdf), "report.log")
    if os.path.exists(log):
        m = re.search(r"Output written on .*?\((\d+) pages?",
                      open(log, errors="ignore").read())
        if m:
            return int(m.group(1))
    return -1


def _verify_pdf_text(d: str, pdf: str) -> bool:
    """Read the finished PDF back and check it actually contains the report.

    Verifying the artifact, not merely the absence of error strings in a log.
    """
    txt = None
    # poppler's pdftotext is not part of TeX Live; prefer the pure-python reader.
    try:
        from pypdf import PdfReader
        txt = "\n".join((pg.extract_text() or "") for pg in PdfReader(pdf).pages)
    except Exception as e:
        tl = "/data/rtxw01/nbak/robot_intel/kztrgg/texlive2025/bin/x86_64-linux"
        exe = os.path.join(tl, "pdftotext")
        if os.path.exists(exe):
            txt_path = os.path.join(d, "report.txt")
            subprocess.run([exe, pdf, txt_path], capture_output=True)
            if os.path.exists(txt_path):
                txt = open(txt_path, errors="ignore").read()
    if txt is None:
        print("ERROR: could not extract text from report.pdf for verification")
        return False
    open(os.path.join(d, "report.txt"), "w").write(txt)
    required = ["Task and demonstration source", "Demonstration quality",
                "Policy and training", "Baseline results", "Data ablation",
                "Failure analysis", "Reproducibility and reflection",
                "Citations and use of tools", "PandaPickCube", "ACT"]
    miss = [r for r in required if r not in txt]
    if miss:
        print("ERROR: report.pdf text is missing required sections:", miss)
        return False
    # unresolved cross-references render as "??"
    if "??" in txt:
        print("ERROR: report.pdf contains unresolved references (??)")
        return False
    import re as _re
    for pat in (r"\(tbd\)", r"(?<![A-Za-z])nan(?![A-Za-z])", r"\bNaN\b"):
        m = _re.search(pat, txt)
        if m:
            print(f"ERROR: report.pdf contains placeholder/NaN text: {m.group(0)!r}")
            return False
    print(f"[report] text verification OK ({len(txt):,} chars, "
          f"all {len(required)} required sections present)")
    return True


if __name__ == "__main__":
    main()
