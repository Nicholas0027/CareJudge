#!/bin/bash
set -euo pipefail
cd /data/lab/CareJudge
set -a; . ./.env; set +a
STAMP=gpt55_formal
mkdir -p outputs/$STAMP reports/innovation_validation logs
LOG=logs/${STAMP}.log
exec > >(tee "$LOG") 2>&1
echo "=== GPT-5.5 formal experiment start $(date) ==="

LIMIT=90
WORKERS=16
for DS in judgebench tldr mtbench_human; do
  echo "[collect] $DS $(date)"
  python3 scripts/collect_uncertainty_concurrent.py \
    --input data/realistic/${DS}.jsonl \
    --out outputs/$STAMP/${DS}_features.jsonl \
    --judge openai:gpt-5.5 \
    --limit $LIMIT --k-self 2 --n-rubrics 3 --workers $WORKERS
done

for DS in judgebench tldr mtbench_human; do
  echo "[experiment] $DS $(date)"
  python3 scripts/run_main_experiment.py \
    --input data/realistic/${DS}.jsonl \
    --features outputs/$STAMP/${DS}_features.jsonl \
    --out-dir outputs/$STAMP/${DS} \
    --method logistic --bound clopper_pearson \
    --alpha 0.15 --delta 0.10 --min-keep 10 \
    --train-frac 0.4 --cal-frac 0.3 || true
  python3 scripts/analyze_innovations.py \
    --features outputs/$STAMP/${DS}_features.jsonl \
    --dataset ${DS} --out-dir reports/innovation_validation \
    --alpha 0.15 --delta 0.10 --min-keep 8 --seeds 8 \
    --train-frac 0.4 --cal-frac 0.3 || true
done

echo "=== done $(date) ==="
git add scripts/collect_uncertainty_concurrent.py scripts/run_cascade_experiment.py care_judge/judges/openai_compat.py logs/${STAMP}.log 2>/dev/null || true
git add care_judge scripts README.md 2>/dev/null || true
git add -f outputs/$STAMP reports/innovation_validation 2>/dev/null || true
git config user.name "CareJudge Lab" 2>/dev/null || true
git config user.email "lab@carejudge.local" 2>/dev/null || true
git commit -m "Methodology fixes + GPT-5.5 formal experiment (valid held-out selective risk, exact Clopper-Pearson, real calibrators)" || true
git push origin main || echo "PUSH_FAILED"
echo "=== pushed $(date) ==="
