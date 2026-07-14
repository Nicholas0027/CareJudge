#!/usr/bin/env python3
"""Cascade v2: only charge for tiers that actually judged (not skipped tiers).
Also runs cross-dataset transfer in the same script."""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from typing import Dict, List
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.calibration.fixed_sequence import calibrate_threshold
from care_judge.calibration.models import fit_calibrator
from care_judge.utils import read_jsonl, write_jsonl


def split3(rows, seed, cal_frac):
    rng = random.Random(seed)
    labeled = [r for r in rows if r.get('correct') is not None]
    rng.shuffle(labeled)
    n = len(labeled)
    nc = max(2, int(n * cal_frac))
    cal = labeled[:nc]
    test = labeled[nc:]
    if not test: test = cal
    return cal, test


def safe_float(x, d=0.0):
    try: return float(x)
    except: return d


def run_cascade(tier_features: Dict[str,str], costs: Dict[str,float], alpha, delta, min_keep, cal_frac, seed=0):
    tiers = list(tier_features.keys())
    feats = {name: {r['id']: r for r in read_jsonl(path) if r.get('correct') is not None}
             for name, path in tier_features.items()}
    common_ids = set.intersection(*[set(feats[t].keys()) for t in tiers])
    ids = sorted(common_ids)
    
    # Split
    cal, test = split3([{'id': i, 'correct': feats[tiers[0]][i]['correct']} for i in ids], seed, cal_frac)
    cal_ids = {r['id'] for r in cal}
    test_ids = [r['id'] for r in test]
    
    # Calibrate each tier
    calibrators, thresholds = {}, {}
    for name in tiers:
        cal_rows = [feats[name][i] for i in cal_ids if i in feats[name]]
        bundle = fit_calibrator(cal_rows, method='logistic')
        calibrators[name] = bundle
        p_cal = bundle.predict_proba(cal_rows)
        pairs = [(p, int(r['correct'])) for r, p in zip(cal_rows, p_cal)]
        thr, _ = calibrate_threshold([x[0] for x in pairs], [x[1] for x in pairs],
                                     alpha=alpha, delta=delta, min_keep=min_keep, bound='clopper_pearson')
        thresholds[name] = thr
    
    # Run cascade on test
    routing = {t: 0 for t in tiers}
    routing['abstain'] = 0
    total_cost = 0.0
    results = []
    for i in test_ids:
        decided = False
        spent = 0.0
        for name in tiers:
            row = feats[name][i]
            spent += costs.get(name, 0.0)  # pay for each tier attempted
            p = float(calibrators[name].predict_proba([row])[0])
            if p >= thresholds[name]:
                results.append({'id': i, 'accepted_by': name, 'correct': int(row['correct']), 'cost': spent})
                routing[name] += 1
                decided = True
                break
        if not decided:
            routing['abstain'] += 1
            results.append({'id': i, 'accepted_by': 'abstain', 'correct': None, 'cost': spent})
        total_cost += spent
    
    accepted = [r for r in results if r['accepted_by'] != 'abstain']
    acc = sum(r['correct'] for r in accepted) / len(accepted) if accepted else None
    strong = tiers[-1]
    strong_only = len(test_ids) * costs.get(strong, 0.0)
    
    return {
        'tiers': tiers,
        'thresholds': thresholds,
        'n_test': len(test_ids),
        'coverage': len(accepted) / max(1, len(test_ids)),
        'accuracy_accepted': acc,
        'risk_accepted': (1-acc) if acc is not None else None,
        'routing': routing,
        'total_cost': total_cost,
        'mean_cost_per_example': total_cost / max(1, len(test_ids)),
        'strong_only_cost': strong_only,
        'cost_savings': strong_only - total_cost,
        'cost_savings_pct': (strong_only - total_cost) / max(1e-9, strong_only) * 100,
    }


def run_transfer(source_path, target_path, alpha, delta, min_keep, cal_frac, seed=0):
    """Train calibrator on source, evaluate on target."""
    source = [r for r in read_jsonl(source_path) if r.get('correct') is not None]
    target = [r for r in read_jsonl(target_path) if r.get('correct') is not None]
    
    # Fit on source
    bundle = fit_calibrator(source, method='logistic')
    
    # Split target into cal (threshold) + test
    cal, test = split3(target, seed, cal_frac)
    p_cal = bundle.predict_proba(cal)
    pairs = [(p, int(r['correct'])) for r, p in zip(cal, p_cal)]
    thr, _ = calibrate_threshold([x[0] for x in pairs], [x[1] for x in pairs],
                                 alpha=alpha, delta=delta, min_keep=min_keep, bound='clopper_pearson')
    p_test = bundle.predict_proba(test)
    
    accepted = [(p, int(r['correct'])) for r, p in zip(test, p_test) if p >= thr]
    acc = sum(c for _, c in accepted) / len(accepted) if accepted else None
    
    return {
        'source': source_path,
        'target': target_path,
        'n_test': len(test),
        'coverage': len(accepted) / max(1, len(test)),
        'accuracy_accepted': acc,
        'risk_accepted': (1-acc) if acc is not None else None,
        'threshold': thr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cascade', 'transfer', 'both'], default='both')
    ap.add_argument('--alpha', type=float, default=0.15)
    ap.add_argument('--delta', type=float, default=0.10)
    ap.add_argument('--min-keep', type=int, default=20)
    ap.add_argument('--cal-frac', type=float, default=0.4)
    ap.add_argument('--out-dir', default='outputs/dual_api_full')
    args = ap.parse_args()
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    models = {
        'qwen15b': {
            'judgebench': 'outputs/qwen15b_realistic/judgebench/features.jsonl',
            'tldr': 'outputs/qwen15b_realistic/tldr/features.jsonl',
            'mtbench_human': 'outputs/qwen15b_mtbench_human/mtbench_human/features.jsonl',
        },
        'deepseek': {
            'judgebench': 'outputs/dual_api_full/deepseek-chat_judgebench_features.jsonl',
            'tldr': 'outputs/dual_api_full/deepseek-chat_tldr_features.jsonl',
            'mtbench_human': 'outputs/dual_api_full/deepseek-chat_mtbench_human_features.jsonl',
        },
        'gpt55': {
            'judgebench': 'outputs/dual_api_full/gpt-5_5_judgebench_features.jsonl',
            'tldr': 'outputs/dual_api_full/gpt-5_5_tldr_features.jsonl',
            'mtbench_human': 'outputs/dual_api_full/gpt-5_5_mtbench_human_features.jsonl',
        },
    }
    costs = {'qwen15b': 0.0, 'deepseek': 0.002, 'gpt55': 0.01}
    
    all_results = {}
    
    if args.mode in ['cascade', 'both']:
        print('\n=== CASCADE EXPERIMENTS ===')
        cascade_results = {}
        for ds in ['judgebench', 'tldr', 'mtbench_human']:
            print(f'\n--- {ds} ---')
            for alpha in [0.15, 0.20, 0.25, 0.30]:
                tier_features = {m: models[m][ds] for m in ['qwen15b', 'deepseek', 'gpt55']}
                r = run_cascade(tier_features, costs, alpha, args.delta, args.min_keep, args.cal_frac)
                rt = r['routing']
                print(f'  α={alpha:.2f}  cov={r["coverage"]:.3f}  acc={r["accuracy_accepted"] or 0:.3f}  '
                      f'qwen={rt["qwen15b"]} ds={rt["deepseek"]} gpt55={rt["gpt55"]} abst={rt["abstain"]}  '
                      f'cost={r["total_cost"]:.2f} strong={r["strong_only_cost"]:.2f} save={r["cost_savings"]:.2f} ({r["cost_savings_pct"]:.1f}%)')
                cascade_results[f'{ds}_alpha{alpha}'] = r
        all_results['cascade'] = cascade_results
        with open(out / 'cascade_all_results.json', 'w') as f:
            json.dump(cascade_results, f, indent=2)
    
    if args.mode in ['transfer', 'both']:
        print('\n=== CROSS-DATASET TRANSFER ===')
        transfer_results = {}
        for model in ['deepseek', 'gpt55']:
            print(f'\n--- Model: {model} ---')
            for source in ['judgebench', 'tldr', 'mtbench_human']:
                for target in ['judgebench', 'tldr', 'mtbench_human']:
                    if source == target:
                        continue
                    r = run_transfer(models[model][source], models[model][target],
                                    args.alpha, args.delta, args.min_keep, args.cal_frac)
                    acc_str = f'{r["accuracy_accepted"]:.3f}' if r['accuracy_accepted'] else 'N/A'
                    print(f'  {source:>14s} → {target:<14s}  cov={r["coverage"]:.3f}  acc={acc_str}  n={r["n_test"]}')
                    transfer_results[f'{model}_{source}_to_{target}'] = r
        all_results['transfer'] = transfer_results
        with open(out / 'transfer_all_results.json', 'w') as f:
            json.dump(transfer_results, f, indent=2)
    
    print('\n=== DONE ===')
    print(f'Results saved to {out}/cascade_all_results.json and {out}/transfer_all_results.json')


if __name__ == '__main__':
    main()
