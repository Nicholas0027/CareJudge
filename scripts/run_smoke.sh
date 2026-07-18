#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" scripts/collect_uncertainty.py --input examples/mini_pairs.jsonl --out outputs/mini_features.jsonl --judge mock:0.72 --k-self 3
"$PYTHON_BIN" scripts/fit_calibrator.py --input outputs/mini_features.jsonl --out outputs/mini_calibrator.pkl --method logistic --calibration-frac 0.7
"$PYTHON_BIN" scripts/run_selective_eval.py --input outputs/mini_features.jsonl --calibrator outputs/mini_calibrator.pkl --out outputs/mini_selected.jsonl --alpha 0.3 --delta 0.1 --min-keep 2
"$PYTHON_BIN" scripts/run_cascade.py --input examples/mini_pairs.jsonl --out outputs/mini_cascade.jsonl --judges cheap=mock:0.65,strong=mock:0.85 --calibrators cheap=outputs/mini_calibrator.pkl,strong=outputs/mini_calibrator.pkl --thresholds cheap=0.55,strong=0.55 --k-self 2
"$PYTHON_BIN" scripts/run_baselines.py --input outputs/mini_features.jsonl --out outputs/mini_baselines.jsonl --alpha 0.3 --delta 0.1 --min-keep 2
"$PYTHON_BIN" scripts/run_main_experiment.py --input examples/mini_pairs.jsonl --out-dir outputs/mini_main --judge mock:0.72 --alpha 0.3 --delta 0.1 --min-keep 2 --k-self 2 --sim-annotators 2 --sim-shots 2
"$PYTHON_BIN" scripts/make_tables.py --reports outputs/mini_main/report.json --out reports/mini_table.csv
"$PYTHON_BIN" scripts/make_plots.py --selected outputs/mini_main/selected.jsonl --out-prefix reports/mini
"$PYTHON_BIN" scripts/collect_ensemble_features.py --input examples/mini_pairs.jsonl --out outputs/mini_ensemble.jsonl --judges mock:0.65,mock:0.72,mock:0.85
"$PYTHON_BIN" scripts/run_transfer_experiment.py --source outputs/mini_features.jsonl --target outputs/mini_ensemble.jsonl --out outputs/mini_transfer.jsonl --alpha 0.3 --delta 0.1 --min-keep 2
