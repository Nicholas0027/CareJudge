#!/usr/bin/env python3
"""Negative-control experiment: non-equivalent rubrics (criterion perturbation)
vs. equivalent rubrics (wording perturbation).

Uses the EXISTING 1.5B model (already cached) on JudgeBench (620 items, fast).
Compares flip rates: if non-equivalent rubrics produce >> higher flip rates
than equivalent ones, the rubric-stability signal reflects genuine judge
instability rather than criterion changes.

Usage: python scripts/negative_control.py
"""
import os, sys, json, time
sys.path.insert(0, '/data/lab/CareJudge')
os.chdir('/data/lab/CareJudge')
for l in open('.env'):
    if '=' in l and not l.startswith('#'):
        k, v = l.strip().split('=', 1)
        os.environ[k] = v

import numpy as np
from care_judge.judges.local_hf import LocalHFJudge
from care_judge.schemas import PairItem

# Equivalent rubrics (wording varies, criterion = "which is better" held fixed)
EQUIV_RUBRICS = [
    "You are an expert evaluator. Compare Response A and Response B. Output ONLY the letter A or B.",
    "Which response is better? Compare A and B carefully and output ONLY A or B.",
    "Compare the two responses. Which one would a careful evaluator prefer? Output ONLY A or B.",
]

# Non-equivalent rubrics (criterion itself changes: brevity vs thoroughness vs formality)
NONEQUIV_RUBRICS = [
    "Which response is more CONCISE? Prefer shorter, more direct answers. Compare A and B. Output ONLY A or B.",
    "Which response is more THOROUGH? Prefer longer, more detailed answers. Compare A and B. Output ONLY A or B.",
    "Which response is more FORMAL? Prefer professional tone and structure. Compare A and B. Output ONLY A or B.",
]

def collect_flip_rates(judge, items, rubrics, label):
    """Collect verdicts under K rubrics, compute flip rate."""
    flips = 0
    total = 0
    for item_dict in items:
        item = PairItem(
            id=item_dict['id'], prompt=item_dict['prompt'],
            response_a=item_dict['response_a'], response_b=item_dict['response_b'],
            label=item_dict.get('label'),
        )
        verdicts = []
        for rubric in rubrics:
            try:
                r = judge.judge(item, rubric=rubric, temperature=0.0)
                verdicts.append(r.winner)
            except:
                verdicts.append('error')
        valid = [v for v in verdicts if v in ('A', 'B')]
        if len(valid) >= 2:
            total += 1
            if len(set(valid)) > 1:
                flips += 1
    rate = flips / total if total > 0 else 0
    print(f"  {label}: {flips}/{total} flipped ({rate:.1%})", flush=True)
    return {'label': label, 'flips': flips, 'total': total, 'flip_rate': rate}

def main():
    print("Loading Qwen2.5-1.5B-Instruct...", flush=True)
    judge = LocalHFJudge(model_name='Qwen/Qwen2.5-1.5B-Instruct', max_new_tokens=80)
    print(f"Model loaded.", flush=True)

    # Use JudgeBench (620 items, fastest)
    items = [json.loads(l) for l in open('data/judgebench_2k.jsonl') if l.strip()]
    print(f"JudgeBench: {len(items)} items", flush=True)

    print("\n=== Equivalent rubrics (wording perturbation) ===", flush=True)
    equiv = collect_flip_rates(judge, items, EQUIV_RUBRICS, "equivalent")

    print("\n=== Non-equivalent rubrics (criterion perturbation) ===", flush=True)
    nonequiv = collect_flip_rates(judge, items, NONEQUIV_RUBRICS, "non-equivalent")

    result = {
        'equivalent': equiv,
        'non_equivalent': nonequiv,
        'ratio': nonequiv['flip_rate'] / equiv['flip_rate'] if equiv['flip_rate'] > 0 else float('inf'),
        'interpretation': 'Non-equivalent rubrics produce substantially higher flip rates, '
                         'supporting that flips under equivalent rubrics reflect judge instability '
                         'rather than criterion changes.' if nonequiv['flip_rate'] > equiv['flip_rate'] * 1.5
                         else 'Flip rates are similar — criterion vs wording not clearly separated.',
    }
    print(f"\n=== RESULT ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)

    os.makedirs('outputs/supplementary', exist_ok=True)
    json.dump(result, open('outputs/supplementary/negative_control.json', 'w'), indent=2)
    print(f"\nSaved -> outputs/supplementary/negative_control.json", flush=True)

if __name__ == '__main__':
    main()
