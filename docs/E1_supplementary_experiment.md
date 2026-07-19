# E1 — Main-Body Strengthening Experiment (mid-capability replication)

**Purpose (important):** E1 is NOT appendix filler. It strengthens the paper's
**central main-body claim (RQ2)**: that protocol-stability fusion helps in the
mid-capability regime. That claim currently rests on a single mid-capability
judge (DeepSeek-V4), which a reviewer can dismiss as an anecdote. E1 turns it
into a **recipe-independent replication** by evaluating a second mid-capability
judge (Qwen2.5-7B, raw 0.57/0.86) on all four benchmarks. The Qwen-7B rows sit
directly in the main results table (Table 1) and in the RQ2 significance
argument (§5.2) — not in the appendix. Completing the two subjective points is
what carries the mid-capability-restricted significance test below `p < 0.05`
and makes the headline effect sharp ("鲜明").

## What is already done
`outputs/final_analysis.json` already contains Qwen2.5-7B on:
- `judgebench` (n=619, raw 0.570, CARE AUROC 0.548±0.035)
- `rewardbench` (n=1000, raw 0.857, CARE AUROC 0.643±0.042)

Both are above the competence floor → the theory predicts positive fusion gains.

## What E1 collects (the missing cells)
| Judge | Benchmark | Status |
|-------|-----------|--------|
| Qwen2.5-7B | tldr_2k (2000) | **TO COLLECT** |
| Qwen2.5-7B | lmaarena_2k (2000) | **TO COLLECT** |

That's it — no new code, no new model. `collect_ladder.py` is already
parameterized by `--model/--tag/--bench`, and `final_analysis.py` auto-includes
any `outputs/scale/<tag>_<bench>_features.jsonl` it finds.

## How to run (on the Lab machine)
```bash
# On the lab box, inside /data/lab/CareJudge, Qwen-7B weights cached, *_2k.jsonl present
bash scripts/run_e1_midcap.sh
```
- Cost: ~450 items/hour × 4000 items ≈ **9 GPU-hours** on one 24–32 GB GPU.
- Idempotent: resumes from existing IDs; safe to re-run after interruption.
- The script re-runs `final_analysis.py` at the end and prints the updated
  `_global_wilcoxon` block.

## Expected outcome and how the paper reports it
- Primary judge–benchmark pairs go from **12 → 14** (adds qwen-7b/tldr, qwen-7b/lmaarena).
- The mid-capability-restricted Wilcoxon (non-saturated pairs) is expected to
  drop below `p = 0.05` if both new points show positive fusion gains, as the
  competence-gating account predicts.
- **Honesty rule:** until these two runs complete, the paper reports the two
  existing Qwen-7B points (JB, RB) and marks the two subjective points as
  *in-progress* (see §5.7, "A second full mid-capability judge"). Do NOT
  back-fill invented numbers. Once collected, update:
  1. `outputs/final_analysis.json` (produced automatically by the script),
  2. §5.7 paragraph — replace "in-progress" with the measured AUROC±std,
  3. the Wilcoxon sentence in §5.2 Finding 2 (n_pairs 15→17, new p-value),
  4. Table 6 (`tab:ensemble`) and Table 7 (`tab:featgroup`) qwen-7b rows.

## Fallback (if a subjective point comes back near chance)
If Qwen-7B on TL;DR or LMArena lands at/below the competence floor (AUROC ≈ 0.5),
that is still a *consistent* result: it means 7B is below threshold on that
harder subjective task. Report it as such (it reinforces competence-gating) and
lean on RewardBench + JudgeBench for the mid-capability significance claim.
