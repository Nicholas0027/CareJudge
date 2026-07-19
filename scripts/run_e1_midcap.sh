#!/usr/bin/env bash
# =============================================================================
# E1 — Second mid-capability judge across ALL FOUR benchmarks
# =============================================================================
# Reviewer request (AAAI-27 super-review, item E1):
#   "The entire 'fusion helps' story rests on a single mid-capability point
#    (DeepSeek-V4). One more judge in the 0.65-0.80 raw-accuracy band ... would
#    turn the Wilcoxon result significant and make the effect 鲜明."
#
# Qwen2.5-7B-Instruct is the natural second mid-capability judge: same family as
# the ladder, raw accuracy 0.57 (JB) / 0.86 (RB) already in hand. It is currently
# MISSING on the two subjective benchmarks. This script collects exactly those
# two, so that Qwen-7B becomes a full 4-benchmark mid-capability judge and enters
# the primary Wilcoxon test (raising it from 12 to 14 primary pairs).
#
# Cost: Qwen-7B collects ~450 items/hour on one 24-32GB GPU, 6 calls/item.
#   tldr_2k (2000) + lmaarena_2k (2000) = 4000 items  ->  ~9 GPU-hours.
# Idempotent: collect_ladder.py resumes from existing IDs, safe to re-run.
#
# Prereqs: run ON the lab machine, inside /data/lab/CareJudge, with the Qwen
# weights already cached (scripts/download_models.py) and the *_2k.jsonl data
# present under data/ (scripts/prep_data.py).
# =============================================================================
set -euo pipefail

REPO="${REPO:-/data/lab/CareJudge}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${TAG:-qwen-7b}"
WORKERS="${WORKERS:-2}"

cd "$REPO"
mkdir -p outputs/scale outputs/logs

echo "=== E1: $TAG on subjective benchmarks (tldr, lmaarena) ==="
for BENCH in tldr lmaarena; do
  echo "--- collecting $TAG / $BENCH ---"
  python scripts/collect_ladder.py \
      --model "$MODEL" \
      --tag   "$TAG" \
      --bench "$BENCH" \
      --workers "$WORKERS" \
      2>&1 | tee "outputs/logs/e1_${TAG}_${BENCH}.log"
done

echo "=== E1: re-running final_analysis to fold Qwen-7B into all tables ==="
python scripts/final_analysis.py \
    --features-dir outputs/scale \
    --out outputs/final_analysis.json \
    --seeds 20

echo "=== E1 DONE. Inspect the updated Wilcoxon (should now cover 14 primary pairs) ==="
python - <<'PY'
import json
d = json.load(open('outputs/final_analysis.json'))
w = d.get('_global_wilcoxon', {})
print("Global Wilcoxon:", w)
q7 = d.get('qwen-7b', {}).get('benches', {})
for b, bv in q7.items():
    c = bv['care']
    print(f"  qwen-7b/{b}: n={bv['n_total']} raw={bv['raw_acc']:.3f} "
          f"care_auroc={c['auroc_mean']:.3f}±{c['auroc_std']:.3f}")
PY
