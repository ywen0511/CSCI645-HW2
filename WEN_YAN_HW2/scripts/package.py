"""Build and verify the submission archive. No `zip` binary on these nodes, so
the archive is written and re-opened with python's zipfile."""
import argparse, os, re, shutil, sys, zipfile

PROJ = os.environ["PROJ"]

SECRET = re.compile(rb"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
                    rb"ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                    rb"BEGIN [A-Z ]*PRIVATE KEY)")


def copytree(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("last"); ap.add_argument("first")
    a = ap.parse_args()
    name = f"{a.last}_{a.first}_HW2"
    dist = os.path.join(PROJ, "dist")
    stage = os.path.join(dist, name)
    shutil.rmtree(dist, ignore_errors=True)
    os.makedirs(stage)

    copytree(os.path.join(PROJ, "code"), os.path.join(stage, "code"))
    copytree(os.path.join(PROJ, "scripts"), os.path.join(stage, "scripts"))
    shutil.copy(os.path.join(PROJ, "README.md"), os.path.join(stage, "README.md"))

    for d in ("results", "configs", "videos", "report", "data"):
        os.makedirs(os.path.join(stage, d), exist_ok=True)
    for sub in ("conditions", "figures", "tables"):
        copytree(os.path.join(PROJ, "results", sub),
                 os.path.join(stage, "results", sub))
    for f in ("summary.csv", "summary_chunk1.csv", "summary.json",
              "paired_comparisons.json", "failure_analysis.json",
              "environment.json", "diag_act_clean_n100_s0.json",
              "diag_replay.json"):
        p = os.path.join(PROJ, "results", f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(stage, "results", f))
    for src, dstname in (
            ("results/environment.json", "environment.json"),
            ("runs/teacher/stage1_report.json", "teacher_stage1_report.json"),
            ("runs/teacher/ppo_progress.json", "ppo_progress.json")):
        p = os.path.join(PROJ, src)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(stage, "configs", dstname))
    cond_dir = os.path.join(PROJ, "runs", "conditions")
    if os.path.isdir(cond_dir):
        for c in sorted(os.listdir(cond_dir)):
            tl = os.path.join(cond_dir, c, "train_log.json")
            if os.path.exists(tl):
                shutil.copy(tl, os.path.join(stage, "configs", f"trainlog_{c}.json"))
    vd = os.path.join(PROJ, "videos")
    if os.path.isdir(vd):
        for f in sorted(os.listdir(vd)):
            if f.endswith((".mp4", ".gif")):
                shutil.copy(os.path.join(vd, f), os.path.join(stage, "videos", f))
    dd = os.path.join(PROJ, "data")
    if os.path.isdir(dd):
        for f in sorted(os.listdir(dd)):
            if f.startswith("demos_") and f.endswith(".npz"):
                shutil.copy(os.path.join(dd, f), os.path.join(stage, "data", f))
    rp = os.path.join(PROJ, "report", "report.pdf")
    if os.path.exists(rp):
        shutil.copy(rp, os.path.join(stage, "report.pdf"))
    for f in ("report_template.tex", "numbers.tex"):
        p = os.path.join(PROJ, "report", f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(stage, "report", f))
    for sub in ("figures", "tables"):
        copytree(os.path.join(PROJ, "report", sub), os.path.join(stage, "report", sub))

    # hygiene
    for root, dirs, files in os.walk(stage, topdown=True):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".pyc"):
                os.remove(os.path.join(root, f))

    print("=== scanning for credential-shaped strings ===")
    hits = []
    for root, _, files in os.walk(stage):
        for f in files:
            p = os.path.join(root, f)
            if os.path.getsize(p) > 20_000_000:
                continue
            try:
                if SECRET.search(open(p, "rb").read()):
                    hits.append(os.path.relpath(p, stage))
            except Exception:
                pass
    if hits:
        print("*** ABORT: possible secrets:", hits); sys.exit(1)
    print("none found")

    zpath = os.path.join(dist, name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, _, files in os.walk(stage):
            for f in sorted(files):
                full = os.path.join(root, f)
                z.write(full, os.path.join(name, os.path.relpath(full, stage)))

    print(f"=== archive: {zpath} ({os.path.getsize(zpath)/1e6:.1f} MB) ===")
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            print("*** ABORT: corrupt entry", bad); sys.exit(1)
        names = z.namelist()
    print(f"entries: {len(names)}  (archive opens and passes CRC check)")
    tops = sorted({"/".join(n.split("/")[1:3]) for n in names if n.count("/") >= 2})
    print("--- contents ---")
    for t in tops[:40]:
        print("   ", t)
    print("--- required deliverables ---")
    ok = True
    for req in ("report.pdf", "README.md", "code/hw2/policies/act.py",
                "code/hw2/policies/flow.py", "results/summary.csv",
                "results/tables/main.tex", "configs/environment.json"):
        present = f"{name}/{req}" in names
        ok &= present
        print(f"   {'OK  ' if present else 'MISS'} {req}")
    nvid = sum(1 for n in names if n.startswith(f"{name}/videos/")
               and n.endswith((".mp4", ".gif")))
    nfig = sum(1 for n in names if n.startswith(f"{name}/results/figures/"))
    ncond = sum(1 for n in names if n.startswith(f"{name}/results/conditions/"))
    print(f"   videos={nvid}  figures={nfig}  condition-results={ncond}")
    if not ok or nvid == 0 or nfig == 0:
        print("PACKAGE_INCOMPLETE"); sys.exit(1)
    print("PACKAGE_DONE")


if __name__ == "__main__":
    main()
