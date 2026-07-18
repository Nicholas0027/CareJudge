#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-local_hf:Qwen/Qwen2.5-1.5B-Instruct}"
STAMP="${STAMP:-qwen15b_realistic}"
ROOT="${ROOT:-/data/lab/CareJudge}"
cd "$ROOT"

mkdir -p data/realistic outputs/$STAMP reports logs
LOG="logs/${STAMP}.log"

echo "========================================" | tee "$LOG"
echo "CARE-Judge realistic benchmark run" | tee -a "$LOG"
echo "Model: $MODEL" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "[1/6] Downloading realistic benchmarks" | tee -a "$LOG"
python3 scripts/download_realistic_benchmarks.py \
  --out-dir data/realistic \
  --judgebench-limit 620 \
  --lmarena-limit 300 \
  --tldr-limit 300 2>&1 | tee -a "$LOG"

for DATASET in judgebench lmarena tldr; do
  DATA="data/realistic/${DATASET}.jsonl"
  OUT="outputs/${STAMP}/${DATASET}"
  mkdir -p "$OUT"
  N=$(wc -l < "$DATA" || echo 0)
  echo "" | tee -a "$LOG"
  echo "[2/6] Dataset $DATASET has $N rows" | tee -a "$LOG"
  if [ "$N" -eq 0 ]; then
    echo "Skipping $DATASET because no rows were downloaded" | tee -a "$LOG"
    continue
  fi

  echo "[3/6] Collecting uncertainty for $DATASET" | tee -a "$LOG"
  python3 scripts/collect_uncertainty.py \
    --input "$DATA" \
    --out "$OUT/features.jsonl" \
    --judge "$MODEL" \
    --k-self 3 \
    --sim-annotators 2 \
    --sim-shots 2 \
    --adaptive-k \
    --adaptive-tau 0.85 \
    --temperature 0.7 2>&1 | tee -a "$LOG"

  echo "[4/6] Calibration/selective eval for $DATASET" | tee -a "$LOG"
  python3 scripts/fit_calibrator.py \
    --input "$OUT/features.jsonl" \
    --out "$OUT/calibrator.pkl" \
    --method logistic \
    --calibration-frac 0.5 2>&1 | tee -a "$LOG"

  python3 scripts/run_selective_eval.py \
    --input "$OUT/features.jsonl" \
    --calibrator "$OUT/calibrator.pkl" \
    --out "$OUT/selected.jsonl" \
    --alpha 0.15 \
    --delta 0.10 \
    --min-keep 20 2>&1 | tee -a "$LOG"

  echo "[5/6] Baselines/reports for $DATASET" | tee -a "$LOG"
  python3 scripts/run_baselines.py \
    --input "$OUT/features.jsonl" \
    --out "$OUT/baselines.jsonl" \
    --alpha 0.15 \
    --delta 0.10 \
    --min-keep 20 2>&1 | tee -a "$LOG"

  python3 scripts/make_plots.py \
    --selected "$OUT/selected.jsonl" \
    --out-prefix "reports/${STAMP}_${DATASET}" 2>&1 | tee -a "$LOG"
done

echo "[6/6] Creating summary tables" | tee -a "$LOG"
REPORTS=$(find outputs/$STAMP -name selected.report.json | sort | tr '\n' ' ')
if [ -n "$REPORTS" ]; then
  python3 scripts/make_tables.py --reports $REPORTS --out "reports/${STAMP}_table.csv" 2>&1 | tee -a "$LOG"
fi

echo "========================================" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "Git status before commit:" | tee -a "$LOG"
git status --short | tee -a "$LOG"

git add scripts/download_realistic_benchmarks.py scripts/run_realistic_benchmarks.sh data/realistic || true
git add -f outputs/$STAMP reports/${STAMP}_* logs/${STAMP}.log || true
git commit -m "Add realistic benchmark results for local Qwen 1.5B" || true
git push origin main
