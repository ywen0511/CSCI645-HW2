#!/bin/bash
# Build the submission ZIP:  package.sh LASTNAME FIRSTNAME
# Includes code, configs, results, figures, tables, videos, report.pdf and the
# demonstration datasets. EXCLUDES policy checkpoints and LSF logs (large and
# regenerable); README.md documents how to regenerate them.
set -uo pipefail
LAST=${1:?usage: package.sh LASTNAME FIRSTNAME}
FIRST=${2:?usage: package.sh LASTNAME FIRSTNAME}
PROJ=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci645-hw2
NAME="${LAST}_${FIRST}_HW2"
STAGE="$PROJ/dist/$NAME"

rm -rf "$PROJ/dist"; mkdir -p "$STAGE"
cp -r "$PROJ/code"      "$STAGE/code"
cp -r "$PROJ/scripts"   "$STAGE/scripts"
cp    "$PROJ/README.md" "$STAGE/README.md"

mkdir -p "$STAGE/results" "$STAGE/configs" "$STAGE/videos" "$STAGE/report" "$STAGE/data"
cp -r "$PROJ/results/conditions"  "$STAGE/results/" 2>/dev/null
cp -r "$PROJ/results/figures"     "$STAGE/results/" 2>/dev/null
cp -r "$PROJ/results/tables"      "$STAGE/results/" 2>/dev/null
for f in summary.csv summary.json paired_comparisons.json failure_analysis.json environment.json; do
  cp "$PROJ/results/$f" "$STAGE/results/" 2>/dev/null
done
cp "$PROJ/results/environment.json"            "$STAGE/configs/" 2>/dev/null
cp "$PROJ/runs/teacher/stage1_report.json"     "$STAGE/configs/teacher_stage1_report.json" 2>/dev/null
cp "$PROJ/runs/teacher/ppo_progress.json"      "$STAGE/configs/ppo_progress.json" 2>/dev/null
for d in "$PROJ"/runs/conditions/*/train_log.json; do
  [ -f "$d" ] || continue
  cp "$d" "$STAGE/configs/trainlog_$(basename "$(dirname "$d")").json"
done
cp "$PROJ"/videos/*.mp4 "$PROJ"/videos/*.gif "$STAGE/videos/" 2>/dev/null
cp "$PROJ"/data/demos_*.npz "$STAGE/data/" 2>/dev/null
cp "$PROJ/report/report.pdf" "$STAGE/report.pdf" 2>/dev/null
cp "$PROJ/report/report_template.tex" "$PROJ/report/numbers.tex" "$STAGE/report/" 2>/dev/null
cp -r "$PROJ/report/figures" "$PROJ/report/tables" "$STAGE/report/" 2>/dev/null

# hygiene: no caches, no secrets
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find "$STAGE" -name '*.pyc' -delete 2>/dev/null
echo "=== scanning for credential-shaped strings ==="
if grep -rIlE '(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)' "$STAGE" 2>/dev/null; then
  echo "*** ABORT: possible secret in the staged tree"; exit 1
fi
echo "none found"

cd "$PROJ/dist" && zip -qr "${NAME}.zip" "$NAME"
echo "=== zip built ==="
ls -lh "$PROJ/dist/${NAME}.zip"
echo "=== verifying the archive opens ==="
unzip -t "$PROJ/dist/${NAME}.zip" | tail -2
echo "=== contents (top levels) ==="
unzip -l "$PROJ/dist/${NAME}.zip" | awk '{print $4}' | sed -n '4,$p' \
  | sed "s|^$NAME/||" | cut -d/ -f1-2 | sort -u | head -40
echo "=== required deliverables present? ==="
for req in report.pdf README.md code/hw2/policies/act.py results/summary.csv; do
  if unzip -l "$PROJ/dist/${NAME}.zip" | grep -q "$NAME/$req"; then echo "  OK   $req"; else echo "  MISSING $req"; fi
done
n_vid=$(unzip -l "$PROJ/dist/${NAME}.zip" | grep -cE "$NAME/videos/.*\.(mp4|gif)" || true)
echo "  videos: $n_vid"
echo "PACKAGE_DONE"
