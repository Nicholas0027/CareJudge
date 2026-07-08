#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from datasets import load_dataset


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', default='data/realistic/mtbench_human.jsonl')
    p.add_argument('--limit', type=int, default=300)
    args=p.parse_args()
    ds=load_dataset('lmsys/mt_bench_human_judgments', split='human')
    rows=[]
    seen=set()
    for i, row in enumerate(ds):
        winner=row.get('winner')
        if winner not in {'model_a','model_b'}:
            continue
        ca=row.get('conversation_a') or []
        cb=row.get('conversation_b') or []
        if len(ca) < 2 or len(cb) < 2:
            continue
        # Use the requested turn when available; otherwise use first assistant answer.
        turn=int(row.get('turn') or 1)
        assistant_idx=min(max(turn*2-1,1), len(ca)-1, len(cb)-1)
        prompt=ca[0].get('content','') if isinstance(ca[0],dict) else str(ca[0])
        ra=ca[assistant_idx].get('content','') if isinstance(ca[assistant_idx],dict) else str(ca[assistant_idx])
        rb=cb[assistant_idx].get('content','') if isinstance(cb[assistant_idx],dict) else str(cb[assistant_idx])
        if not prompt or not ra or not rb:
            continue
        # Avoid exact duplicate judge rows from same question/turn/model pair.
        key=(row.get('question_id'), row.get('model_a'), row.get('model_b'), turn, winner)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'id': f"mtbench_{row.get('question_id')}_{turn}_{len(rows)}",
            'prompt': prompt,
            'response_a': ra,
            'response_b': rb,
            'label': 'A' if winner == 'model_a' else 'B',
            'domain': 'mtbench_human',
            'model_a': row.get('model_a'),
            'model_b': row.get('model_b'),
            'judge': row.get('judge'),
            'turn': turn,
        })
        if args.limit and len(rows) >= args.limit:
            break
    write_jsonl(args.out, rows)
    print(f'wrote {len(rows)} rows to {args.out}')

if __name__ == '__main__':
    main()
