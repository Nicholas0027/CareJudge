from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from care_judge.baselines import run_all_baselines
from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import fit_calibrator
from care_judge.data.loaders import load_jsonl_pairs
from care_judge.evaluation.metrics import calibration_report
from care_judge.judges.factory import make_judge
from care_judge.selective.evaluate import apply_threshold, summarize_selective
from care_judge.uncertainty.feature_builder import collect_uncertainty_features, record_to_row
from care_judge.utils import write_jsonl


def run_single_dataset_experiment(
    dataset_path: str,
    out_dir: str,
    judge_spec: str,
    method: str = "logistic",
    limit: int | None = None,
    seed: int = 0,
    calibration_frac: float = 0.5,
    alpha: float = 0.1,
    delta: float = 0.1,
    min_keep: int = 20,
    k_self: int = 3,
    sim_annotators: int = 0,
    sim_shots: int = 3,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = load_jsonl_pairs(dataset_path, limit=limit)
    judge = make_judge(judge_spec)
    rows = []
    for item in items:
        rec = collect_uncertainty_features(item, judge, k_self=k_self, sim_examples=items, sim_annotators=sim_annotators, sim_shots=sim_shots)
        row = record_to_row(rec)
        row["domain"] = item.domain
        rows.append(row)
    write_jsonl(out / "features.jsonl", rows)

    labeled = [r for r in rows if r.get("correct") is not None]
    rng = random.Random(seed)
    rng.shuffle(labeled)
    n_cal = max(2, int(len(labeled) * calibration_frac))
    cal_ids = {r["id"] for r in labeled[:n_cal]}
    cal_rows = [r for r in rows if r["id"] in cal_ids]
    test_rows = [r for r in rows if r["id"] not in cal_ids]
    bundle = fit_calibrator(cal_rows, method=method)
    bundle.save(str(out / "calibrator.pkl"))
    p_test = bundle.predict_proba(test_rows)
    labeled_test = [(p, int(r["correct"])) for r, p in zip(test_rows, p_test) if r.get("correct") is not None]
    threshold, threshold_trace = calibrate_threshold([x[0] for x in labeled_test], [x[1] for x in labeled_test], alpha=alpha, delta=delta, min_keep=min_keep)
    selected = apply_threshold(test_rows, p_test, threshold)
    write_jsonl(out / "selected.jsonl", selected)

    baselines = run_all_baselines(test_rows, alpha=alpha, delta=delta, min_keep=min_keep)
    write_jsonl(out / "baselines.jsonl", baselines)
    report = {
        "dataset_path": dataset_path,
        "judge": judge_spec,
        "n_total": len(rows),
        "n_calibration": len(cal_rows),
        "n_test": len(test_rows),
        "alpha": alpha,
        "delta": delta,
        "method": method,
        "care_threshold": threshold,
        "care_threshold_trace": threshold_trace,
        "care_calibration": calibration_report(test_rows, p_test),
        "care_selective": summarize_selective(selected),
        "baselines": baselines,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
