"""Build and verify the submission ZIP from a COMMITTED tree.

    python scripts/package.py [--rev HEAD] [--out dist/WEN_YAN_HW2.zip]

The archive is `git archive <rev> WEN_YAN_HW2`, so it contains exactly the
committed files under a WEN_YAN_HW2/ top-level folder -- never an untracked
intermediate file. It is then re-opened, CRC-checked, scanned for
credential-shaped strings and checked for the required deliverables, and its
SHA-256 is printed.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

SECRET = re.compile(rb"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                    rb"AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)")
REQUIRED = ["report.pdf", "README.md", "code/hw2/policies/act.py", "code/hw2/evaluate.py",
            "code/hw2/metrics.py", "scripts/stage3_report.py", "report/report_template.tex",
            "results/summary.csv", "results/tables/main.tex", "results/environment.json",
            "configs/fresh_protocol.json", "configs/requirements.lock.txt",
            "data/demos_sigma0.npz", "tests/test_metrics.py"]


def main():
    here = Path(__file__).resolve().parents[1]          # .../WEN_YAN_HW2
    repo = here.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", default=str(here / "dist" / f"{here.name}.zip"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rev = subprocess.run(["git", "-C", str(repo), "rev-parse", a.rev], capture_output=True,
                         text=True, check=True).stdout.strip()
    tree = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{a.rev}:{here.name}"],
                          capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "archive", "--format=zip", "-o", str(out),
                    rev, here.name], check=True)
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad:
            sys.exit(f"corrupt entry {bad}")
        names = set(z.namelist())
        hits = [n for n in names if not n.endswith("/") and z.getinfo(n).file_size < 20_000_000
                and SECRET.search(z.read(n))]
    if hits:
        sys.exit(f"possible secrets in {hits}")
    tops = {n.split("/")[0] for n in names}
    assert tops == {here.name}, f"unexpected top-level entries {tops}"
    missing = [r for r in REQUIRED if f"{here.name}/{r}" not in names]
    nvid = sum(n.endswith(".mp4") for n in names)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"archive  {out}  ({out.stat().st_size/1e6:.1f} MB, {len(names)} entries)")
    print(f"commit   {rev}\ntree     {tree}  (git tree of {here.name}/)\nsha256   {sha}")
    print(f"videos   {nvid}")
    if missing or nvid < 4:
        sys.exit(f"PACKAGE_INCOMPLETE missing={missing} videos={nvid}")
    print("PACKAGE_DONE")


if __name__ == "__main__":
    main()
