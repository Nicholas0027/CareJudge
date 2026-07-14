from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from care_judge.baselines import run_all_baselines
from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import fit_calibrator
from care_judge.data.loaders import load_jsonl_pairs
from care_judge.evaluation.metrics import calibration_report, bootstrap_ci
from care_judge.judges.factory import make_judge
from care_judge.selective.evaluate import apply_threshold, summarize_selective
from care_judge.uncertainty.feature_builder import collect_uncertainty_features, record_to_row
from care_judge.utils import write_jsonl


def _three_way_split(labeled: List[Dict[str, Any]], seed: int, train_frac: float, cal_frac: float):
    """Split labeled rows into disjoint train / calibration / test sets.

    - train:      used to FIT the calibrator (feature -> P(correct))
    - calibration used to SELECT the risk-controlled threshold
    - test:       held-out, used ONLY to report risk / coverage / calibration

    This disjointness is what makes the selective-risk guarantee valid.
    """
    rng = random.Random(seed)
    rows = list(labeled)
    rng.shuffle(rows)
    n = len(rows)
    n_train = max(2, int(n * train_frac))
    n_cal = max(2, int(n * cal_frac))
    train = rows[:n_train]
    cal = rows[n_train:n_train + n_cal]
    test = rows[n_train + n_cal:]
    if not test:  # tiny datasets: fall back to reusing cal as test but flag it
        test = cal
    return train, cal, test


def _per_domain_breakdown(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dom: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_dom[str(r.get("domain", "general"))].append(r)
    out = {}
    for dom, rows in by_dom.items():
        out[dom] = summarize_selective(rows)
    return out


def run_single_dataset_experiment(
    dataset_path: str,
    out_dir: str,
    judge_spec: str,
    method: str = "logistic",
    limit: int | None = None,
    seed: int = 0,
    train_frac: float = 0.4,
    cal_frac: float = 0.3,
    alpha: float = 0.1,
    delta: float = 0.1,
    min_keep: int = 20,
    k_self: int = 3,
    sim_annotators: int = 0,
    sim_shots: int = 3,
    bound: str = "clopper_pearson",
    features_path: str | None = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if features_path and Path(features_path).exists():
        from care_judge.utils import read_jsonl
        rows = read_jsonl(features_path)
    else:
        items = load_jsonl_pairs(dataset_path, limit=limit)
        judge = make_judge(judge_spec)
        rows = []
        for item in items:
            rec = collect_uncertainty_features(
                item, judge, k_self=k_self, sim_examples=items,
                sim_annotators=sim_annotators, sim_shots=sim_shots,
            )
            row = record_to_row(rec)
            row["domain"] = item.domain
            rows.append(row)
        write_jsonl(out / "features.jsonl", rows)

    labeled = [r for r in rows if r.get("correct") is not None]
    train, cal, test = _three_way_split(labeled, seed, train_frac, cal_frac)

    # 1) Fit calibrator on TRAIN only.
    bundle = fit_calibrator(train, method=method)
    bundle.save(str(out / "calibrator.pkl"))

    # 2) Select risk-controlled threshold on the disjoint CALIBRATION set.
    p_cal = bundle.predict_proba(cal)
    cal_pairs = [(p, int(r["correct"])) for r, p in zip(cal, p_cal)]
    threshold, threshold_trace = calibrate_threshold(
        [x[0] for x in cal_pairs], [x[1] for x in cal_pairs],
        alpha=alpha, delta=delta, min_keep=min_keep, bound=bound,
    )

    # 3) Evaluate ONLY on held-out TEST with the fixed threshold.
    p_test = bundle.predict_proba(test)
    selected = apply_threshold(test, p_test, threshold)
    write_jsonl(out / "selected.jsonl", selected)

    # Baselines are also selected on cal and evaluated on test (fair comparison).
    baselines = run_all_baselines(cal, test, alpha=alpha, delta=delta, min_keep=min_keep, bound=bound)
    write_jsonl(out / "baselines.jsonl", baselines)

    sel = summarize_selective(selected)
    accepted_correct = [int(r["correct"]) for r in selected if r.get("accepted") and r.get("correct") is not None]
    ci = bootstrap_ci([float(x) for x in accepted_correct]) if accepted_correct else (None, None)

    report = {
        "dataset_path": dataset_path,
        "judge": judge_spec,
        "n_total": len(rows),
        "n_train": len(train),
        "n_calibration": len(cal),
        "n_test": len(test),
        "alpha": alpha,
        "delta": delta,
        "method": bundle.method,
        "bound": bound,
        "care_threshold": threshold,
        "care_threshold_trace": threshold_trace,
        "care_calibration_test": calibration_report(test, p_test),
        "care_selective_test": sel,
        "care_accepted_accuracy_ci95": {"low": ci[0], "high": ci[1]},
        "care_per_domain_test": _per_domain_breakdown(selected),
        "baselines_test": baselines,
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
