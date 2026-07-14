#!/bin/bash
set -euo pipefail
cd /data/lab/CareJudge
set -a; . ./.env; set +a

STAMP=dual_api_full
mkdir -p outputs/$STAMP logs
LOG=logs/${STAMP}.log
exec > >(tee "$LOG") 2>&1

WORKERS=16
K_SELF=2
N_RUBRICS=3
ALPHA=0.15
DELTA=0.10
MIN_KEEP=20

DATASETS="judgebench tldr mtbench_human"

echo "========================================"
echo "Dual-API Full Formal Experiment"
echo "Start: $(date)"
echo "========================================"

# ── Phase 1: Collect features for both models ──
for MODEL_SPEC in "deepseek:deepseek-chat" "openai:gpt-5.5"; do
  MODEL_NAME=$(echo $MODEL_SPEC | cut -d: -f2)
  MODEL_TAG=$(echo $MODEL_NAME | tr '.' '_')
  
  for DS in $DATASETS; do
    INPUT="data/realistic/${DS}.jsonl"
    OUT="outputs/$STAMP/${MODEL_TAG}_${DS}_features.jsonl"
    
    if [ -f "$OUT" ]; then
      N=$(wc -l < "$OUT")
      echo "[SKIP] $MODEL_TAG/$DS already has $N rows"
      continue
    fi
    
    echo "[COLLECT] $MODEL_TAG/$DS $(date)"
    python3 scripts/collect_uncertainty_concurrent.py \
      --input "$INPUT" \
      --out "$OUT" \
      --judge "$MODEL_SPEC" \
      --limit 9999 \
      --k-self $K_SELF \
      --n-rubrics $N_RUBRICS \
      --workers $WORKERS
  done
done

# ── Phase 2: Run full experiment analysis ──
for MODEL_TAG in "deepseek-chat" "gpt-5_5"; do
  for DS in $DATASETS; do
    FEATURES="outputs/$STAMP/${MODEL_TAG}_${DS}_features.jsonl"
    if [ ! -f "$FEATURES" ]; then
      echo "[SKIP] no features for $MODEL_TAG/$DS"
      continue
    fi
    echo "[EXPERIMENT] $MODEL_TAG/$DS $(date)"
    python3 scripts/run_full_qwen_experiment.py \
      --features "$FEATURES" \
      --dataset "${MODEL_TAG}_${DS}" \
      --out-dir outputs/$STAMP/results \
      --seeds 10 \
      --alpha $ALPHA \
      --delta $DELTA \
      --min-keep $MIN_KEEP \
      --bound clopper_pearson
  done
done

# ── Phase 3: Commit and push ──
echo "[PUSH] $(date)"
git add care_judge/judges/deepseek_compat.py care_judge/judges/factory.py scripts/collect_uncertainty_concurrent.py scripts/run_full_qwen_experiment.py run_dual_api_full.sh logs/${STAMP}.log 2>/dev/null || true
git add -f outputs/$STAMP 2>/dev/null || true
git config user.name "CareJudge Lab" 2>/dev/null || true
git config user.email "lab@carejudge.local" 2>/dev/null || true
git commit -m "Dual-API full formal experiment: GPT-5.5 + DeepSeek-chat on all datasets

- Full scale: 620 JudgeBench + 300 TL;DR + 300 MT-Bench per model
- 6 uncertainty calls/example: base + self-consistency + swap + 3 rubrics
- Corrected methodology: 3-way split, exact Clopper-Pearson, 10 seeds
- Alpha sweep (0.05-0.30) for risk-coverage curves
- Best-single-signal gating vs logistic fusion vs raw
- Signal failure rates (rubric/position/self stable vs unstable)" || true
git push origin main || echo "PUSH_FAILED"
echo "[DONE] $(date)"
