#!/bin/bash
set -euo pipefail
ROOT="${ROOT:-/data/lab/CareJudge}"
MODEL="${MODEL:-local_hf:Qwen/Qwen2.5-1.5B-Instruct}"
STAMP="${STAMP:-qwen15b_mtbench_human}"
cd "$ROOT"
mkdir -p data/realistic outputs/$STAMP/mtbench_human reports/innovation_validation logs
LOG="logs/${STAMP}.log"
exec > >(tee "$LOG") 2>&1

echo "========================================"
echo "CARE-Judge MT-Bench Human/LMArena-like validation"
echo "Model: $MODEL"
echo "Start: $(date)"
echo "========================================"

python3 scripts/convert_mtbench_human.py --out data/realistic/mtbench_human.jsonl --limit 300
wc -l data/realistic/mtbench_human.jsonl

python3 scripts/collect_uncertainty.py \
  --input data/realistic/mtbench_human.jsonl \
  --out outputs/$STAMP/mtbench_human/features.jsonl \
  --judge "$MODEL" \
  --k-self 3 \
  --sim-annotators 2 \
  --sim-shots 2 \
  --adaptive-k \
  --adaptive-tau 0.85 \
  --temperature 0.7

python3 scripts/fit_calibrator.py \
  --input outputs/$STAMP/mtbench_human/features.jsonl \
  --out outputs/$STAMP/mtbench_human/calibrator.pkl \
  --method logistic \
  --calibration-frac 0.5

python3 scripts/run_selective_eval.py \
  --input outputs/$STAMP/mtbench_human/features.jsonl \
  --calibrator outputs/$STAMP/mtbench_human/calibrator.pkl \
  --out outputs/$STAMP/mtbench_human/selected.jsonl \
  --alpha 0.15 \
  --delta 0.10 \
  --min-keep 20

python3 scripts/run_baselines.py \
  --input outputs/$STAMP/mtbench_human/features.jsonl \
  --out outputs/$STAMP/mtbench_human/baselines.jsonl \
  --alpha 0.15 \
  --delta 0.10 \
  --min-keep 20

python3 scripts/make_plots.py \
  --selected outputs/$STAMP/mtbench_human/selected.jsonl \
  --out-prefix reports/${STAMP}_mtbench_human

python3 scripts/analyze_innovations.py \
  --features outputs/$STAMP/mtbench_human/features.jsonl \
  --dataset mtbench_human \
  --out-dir reports/innovation_validation \
  --alpha 0.15 \
  --delta 0.10 \
  --min-keep 20 \
  --seeds 5

python3 - <<'PY'
import csv, glob, json
rows=[]
for path in glob.glob('reports/innovation_validation/*_innovation_ablation.json'):
    rows.extend(json.load(open(path)))
keys=sorted({k for r in rows for k in r})
with open('reports/innovation_validation/combined_innovation_ablation.csv','w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print('combined rows', len(rows))
PY

echo "Final report:"
cat outputs/$STAMP/mtbench_human/selected.report.json | python3 -m json.tool

git add scripts/convert_mtbench_human.py scripts/run_mtbench_human_validation.sh data/realistic/mtbench_human.jsonl logs/${STAMP}.log

git add -f outputs/$STAMP reports/${STAMP}_* reports/innovation_validation

git commit -m "Add MT-Bench human preference validation for local Qwen 1.5B" || true
git push origin main

echo "Finished: $(date)"
