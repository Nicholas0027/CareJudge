#!/usr/bin/env python3
"""Optimized Qwen feature collection using multiprocessing.

Each worker loads its own model instance, avoiding thread-safety issues with
HuggingFace generate(). Workers process items in parallel on the shared GPU.

Resume: tracks done item IDs (order-independent, crash-safe).

Usage:
  python scripts/collect_qwen_mp.py --bench tldr --workers 4
  python scripts/collect_qwen_mp.py --bench rewardbench --workers 4
  python scripts/collect_qwen_mp.py --bench lmaarena --workers 4
"""
import os, sys, json, time, argparse
from multiprocessing import Pool

# ── Config ──
RUBRICS = [
    "You are an expert evaluator. Compare Response A and Response B. Output ONLY the letter A or B.",
    "Which response is better? Compare A and B carefully and output ONLY A or B.",
    "Compare the two responses. Which one would a careful evaluator prefer? Output ONLY A or B.",
]

# Global per-worker state
_judge = None

def init_worker():
    global _judge
    sys.path.insert(0, '/data/lab/CareJudge')
    os.chdir('/data/lab/CareJudge')
    from care_judge.judges.local_hf import LocalHFJudge
    _judge = LocalHFJudge(model_name='Qwen/Qwen2.5-1.5B-Instruct', max_new_tokens=80)
    print(f"[Worker {os.getpid()}] Model loaded", flush=True)

def process_item(item_dict):
    global _judge
    from care_judge.schemas import PairItem
    from care_judge.uncertainty.collector_extended import collect_with_call_trace
    item = PairItem(
        id=item_dict['id'],
        prompt=item_dict['prompt'],
        response_a=item_dict['response_a'],
        response_b=item_dict['response_b'],
        label=item_dict.get('label'),
    )
    try:
        row = collect_with_call_trace(item, _judge, RUBRICS, k_self=3, temperature=0.7)
        return json.dumps(row)
    except Exception as e:
        print(f"  [WARN] item {item_dict['id']}: {e}", flush=True)
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True, choices=['tldr', 'rewardbench', 'lmaarena'])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, '/data/lab/CareJudge')
    os.chdir('/data/lab/CareJudge')

    # Load all items
    items = [json.loads(l) for l in open(f'data/{args.bench}_2k.jsonl') if l.strip()]
    print(f"Total items in {args.bench}: {len(items)}", flush=True)

    out_path = f'outputs/scale/qwen-1.5b_{args.bench}_features.jsonl'

    # Resume: track done IDs (order-independent)
    done_ids = set()
    done_count = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)['id'])
                    done_count += 1
                except Exception:
                    pass
    print(f"Already done: {done_count} items", flush=True)

    remaining = [it for it in items if it['id'] not in done_ids]
    print(f"Remaining: {len(remaining)} items", flush=True)
    if not remaining:
        print("Nothing to do.", flush=True)
        return

    # Process with multiprocessing Pool
    t0 = time.time()
    n_written = 0
    n_failed = 0
    with open(out_path, 'a') as fout, Pool(args.workers, initializer=init_worker) as pool:
        for result in pool.imap(process_item, remaining, chunksize=4):
            if result is not None:
                fout.write(result + '\n')
                fout.flush()
                n_written += 1
            else:
                n_failed += 1
            if (n_written + n_failed) % 50 == 0:
                elapsed = time.time() - t0
                rate = (n_written + n_failed) / elapsed * 3600
                total_done = done_count + n_written + n_failed
                print(f"  {total_done}/{len(items)} ({n_written} written, {n_failed} failed) "
                      f"rate={rate:.0f}/h elapsed={elapsed/60:.1f}min", flush=True)

    elapsed = time.time() - t0
    print(f"DONE: {n_written} written, {n_failed} failed in {elapsed/60:.1f}min "
          f"({n_written/elapsed*3600:.0f}/h)", flush=True)

    # Verify final count
    final = sum(1 for _ in open(out_path))
    print(f"Final file: {final} records", flush=True)

if __name__ == '__main__':
    main()
