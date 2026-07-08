# CARE-Judge

CARE-Judge is a runnable research codebase for **Calibrated, Abstaining, and Robust LLM-as-a-Judge Evaluation**. It implements the proposed AAAI-style experiment plan: multi-signal judge uncertainty, rubric perturbation, position-swap robustness, self-consistency, learned correctness calibration, fixed-sequence risk-controlled abstention, and cost-aware cascades.

## What Is Implemented

- Pairwise dataset loader for local JSONL and optional HuggingFace datasets.
- Judges:
  - `mock[:accuracy]` for reproducible smoke tests with no API key.
  - `litellm:<model>` for OpenAI/Anthropic/Gemini/Together/local OpenAI-compatible endpoints via LiteLLM.
- Uncertainty features:
  - verbal/base confidence
  - self-consistency
  - A/B position-swap consistency
  - rubric perturbation stability
  - confidence variance
  - length-gap bias proxy
- Calibration:
  - logistic regression
  - isotonic regression
  - gradient boosting
  - fixed-sequence Clopper-Pearson thresholding
- Selective evaluation and cascaded cheap-to-strong judging.

## Install

```bash
python -m pip install -e .
```

For API judges:

```bash
python -m pip install -e '.[api]'
export OPENAI_API_KEY=...
```

For HuggingFace dataset loading:

```bash
python -m pip install -e '.[hf]'
```

## Smoke Test

```bash
bash scripts/run_smoke.sh
```

This runs the full pipeline using the mock judge and writes outputs under `outputs/`.

## Data Format

Use JSONL with these fields:

```json
{"id":"1","prompt":"...","response_a":"...","response_b":"...","label":"A","domain":"math"}
```

Aliases are supported: `question`, `query`, `chosen`, `rejected`, `winner`, `category`, etc. If rows contain `chosen` and `rejected`, `chosen` is treated as response A and label defaults to `A`.

## Main Commands

Collect uncertainty features:

```bash
python scripts/collect_uncertainty.py \
  --input data/lmarena.jsonl \
  --out outputs/lmarena_gptmini_features.jsonl \
  --judge litellm:gpt-4o-mini \
  --k-self 5
```

Use adaptive sampling and simulated annotators:

```bash
python3 scripts/collect_uncertainty.py \
  --input data/lmarena.jsonl \
  --out outputs/lmarena_adaptive_features.jsonl \
  --judge litellm:gpt-4o-mini \
  --k-self 8 \
  --adaptive-k \
  --adaptive-tau 0.85 \
  --sim-annotators 5 \
  --sim-shots 5
```

Collect ensemble-disagreement features across multiple judges:

```bash
python3 scripts/collect_ensemble_features.py \
  --input data/lmarena.jsonl \
  --out outputs/lmarena_ensemble.jsonl \
  --judges litellm:gpt-4o-mini,litellm:gpt-4o,litellm:claude-3-5-sonnet-20240620
```

Fit a calibrator:

```bash
python scripts/fit_calibrator.py \
  --input outputs/lmarena_gptmini_features.jsonl \
  --out outputs/lmarena_logistic.pkl \
  --method logistic
```

Run selective evaluation:

```bash
python scripts/run_selective_eval.py \
  --input outputs/lmarena_gptmini_features.jsonl \
  --calibrator outputs/lmarena_logistic.pkl \
  --out outputs/lmarena_selected.jsonl \
  --alpha 0.10 \
  --delta 0.10
```

Run cascade:

```bash
python scripts/run_cascade.py \
  --input data/lmarena.jsonl \
  --out outputs/lmarena_cascade.jsonl \
  --judges cheap=litellm:gpt-4o-mini,strong=litellm:gpt-4o \
  --calibrators cheap=outputs/cheap.pkl,strong=outputs/strong.pkl \
  --thresholds cheap=0.80,strong=0.90
```

Run all offline baselines on an existing feature file:

```bash
python3 scripts/run_baselines.py \
  --input outputs/lmarena_gptmini_features.jsonl \
  --out outputs/lmarena_baselines.jsonl \
  --alpha 0.10 \
  --delta 0.10
```

Run the complete one-dataset experiment, including feature collection,
calibration, CARE selective evaluation, and all baselines:

```bash
python3 scripts/run_main_experiment.py \
  --input data/lmarena.jsonl \
  --out-dir outputs/main_lmarena \
  --judge litellm:gpt-4o-mini \
  --k-self 5 \
  --sim-annotators 5 \
  --sim-shots 5 \
  --alpha 0.10 \
  --delta 0.10
```

Convert common dataset dumps to CARE JSONL:

```bash
python3 scripts/convert_dataset.py \
  --input raw/judgebench.jsonl \
  --out data/judgebench.jsonl \
  --format judgebench
```

Create tables and reliability data:

```bash
python3 scripts/make_tables.py --reports outputs/main_lmarena/report.json --out reports/main_table.csv
python3 scripts/make_plots.py --selected outputs/main_lmarena/selected.jsonl --out-prefix reports/lmarena
```

Run cross-dataset calibration transfer:

```bash
python3 scripts/run_transfer_experiment.py \
  --source outputs/tldr_features.jsonl \
  --target outputs/judgebench_features.jsonl \
  --out outputs/tldr_to_judgebench_selected.jsonl \
  --alpha 0.10 \
  --delta 0.10
```

## Recommended AAAI 2027 Experiment Matrix

Datasets to convert into this JSONL format:

- TL;DR summarization preference pairs
- LMArena / Chatbot Arena preference subset
- JudgeBench
- RewardBench 2
- ContextualJudgeBench
- IF-RewardBench

Baselines to run:

- raw cheap judge
- raw strong judge
- self-consistency `k=3,5,8`
- position-swap abstention
- rubric-stability abstention
- Trust-or-Escalate / simulated annotators baseline
- criteria + ensemble baseline
- CARE-Single
- CARE-Cascade

Metrics to report:

- raw accuracy / human agreement
- selective risk
- coverage
- ECE
- Brier score
- AUROC/AUPRC for predicting judge correctness
- cost per 1,000 examples
- accepted-by-tier distribution
- rubric flip rate
- position inconsistency rate

## GPU Requirements

- API-only: 0 GPUs.
- Local 7B/8B judge through vLLM: 1 GPU with 24GB VRAM.
- Calibration and thresholding: CPU only.
