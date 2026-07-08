#!/bin/bash
set -e
cd /data/lab/CareJudge

MODEL="local_hf:Qwen/Qwen2.5-1.5B-Instruct"
DATA="examples/test_100.jsonl"
OUTDIR="outputs/qwen15b_main"
LOGFILE="logs/qwen15b_experiment.log"

echo "========================================" | tee "$LOGFILE"
echo "CARE-Judge Local Model Experiment" | tee -a "$LOGFILE"
echo "Model: $MODEL" | tee -a "$LOGFILE"
echo "Dataset: $DATA" | tee -a "$LOGFILE"
echo "Start: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

# Step 1: Collect uncertainty features
echo "" | tee -a "$LOGFILE"
echo "[STEP 1] Collecting uncertainty features..." | tee -a "$LOGFILE"
python3 scripts/collect_uncertainty.py \
  --input "$DATA" \
  --out "$OUTDIR/features.jsonl" \
  --judge "$MODEL" \
  --k-self 3 \
  --sim-annotators 3 \
  --sim-shots 3 \
  --temperature 0.7 \
  2>&1 | tee -a "$LOGFILE"

echo "[STEP 1] Done: $(date)" | tee -a "$LOGFILE"
echo "Feature file size: $(wc -l < $OUTDIR/features.jsonl) rows" | tee -a "$LOGFILE"

# Step 2: Fit calibrator
echo "" | tee -a "$LOGFILE"
echo "[STEP 2] Fitting calibrator..." | tee -a "$LOGFILE"
python3 scripts/fit_calibrator.py \
  --input "$OUTDIR/features.jsonl" \
  --out "$OUTDIR/calibrator.pkl" \
  --method logistic \
  --calibration-frac 0.5 \
  2>&1 | tee -a "$LOGFILE"

echo "[STEP 2] Done: $(date)" | tee -a "$LOGFILE"

# Step 3: Run selective evaluation
echo "" | tee -a "$LOGFILE"
echo "[STEP 3] Running selective evaluation..." | tee -a "$LOGFILE"
python3 scripts/run_selective_eval.py \
  --input "$OUTDIR/features.jsonl" \
  --calibrator "$OUTDIR/calibrator.pkl" \
  --out "$OUTDIR/selected.jsonl" \
  --alpha 0.15 \
  --delta 0.1 \
  --min-keep 10 \
  2>&1 | tee -a "$LOGFILE"

echo "[STEP 3] Done: $(date)" | tee -a "$LOGFILE"

# Step 4: Run all baselines
echo "" | tee -a "$LOGFILE"
echo "[STEP 4] Running all baselines..." | tee -a "$LOGFILE"
python3 scripts/run_baselines.py \
  --input "$OUTDIR/features.jsonl" \
  --out "$OUTDIR/baselines.jsonl" \
  --alpha 0.15 \
  --delta 0.1 \
  --min-keep 10 \
  2>&1 | tee -a "$LOGFILE"

echo "[STEP 4] Done: $(date)" | tee -a "$LOGFILE"

# Step 5: Generate tables and plots
echo "" | tee -a "$LOGFILE"
echo "[STEP 5] Generating tables and plots..." | tee -a "$LOGFILE"
python3 scripts/make_tables.py \
  --reports "$OUTDIR/selected.report.json" \
  --out reports/qwen15b_table.csv \
  2>&1 | tee -a "$LOGFILE"

python3 scripts/make_plots.py \
  --selected "$OUTDIR/selected.jsonl" \
  --out-prefix reports/qwen15b \
  2>&1 | tee -a "$LOGFILE"

echo "[STEP 5] Done: $(date)" | tee -a "$LOGFILE"

# Summary
echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "EXPERIMENT COMPLETE: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Outputs:" | tee -a "$LOGFILE"
ls -la "$OUTDIR/" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "Reports:" | tee -a "$LOGFILE"
ls -la reports/ | tee -a "$LOGFILE"

# Print final report
echo "" | tee -a "$LOGFILE"
echo "=== FINAL SELECTIVE REPORT ===" | tee -a "$LOGFILE"
cat "$OUTDIR/selected.report.json" | python3 -m json.tool 2>/dev/null | tee -a "$LOGFILE"

