#!/usr/bin/env python3
"""Generic local-HF feature collector for the capability ladder.

Parameterized by model name so we can build a CONTROLLED capability ladder
using same-family Qwen2.5 models (1.5B / 7B / 14B) that vary only in size,
holding training recipe and API configuration fixed. This addresses the
reviewer's request to disentangle judge-capability from size/recipe/API.

Each worker loads its own model instance. Resume tracks done item IDs.

Usage:
  python scripts/collect_ladder.py --model Qwen/Qwen2.5-7B-Instruct  --tag qwen-7b  --bench judgebench --workers 2
  python scripts/collect_ladder.py --model Qwen/Qwen2.5-14B-Instruct --tag qwen-14b --bench judgebench --workers 1
"""
import os, sys, json, time, argparse
from multiprocessing import Pool

RUBRICS = [
    "You are an expert evaluator. Compare Response A and Response B. Output ONLY the letter A or B.",
    "Which response is better? Compare A and B carefully and output ONLY A or B.",
    "Compare the two responses. Which one would a careful evaluator prefer? Output ONLY A or B.",
]

_judge = None
_MODEL = None

def init_worker(model_name):
    global _judge, _MODEL
    _MODEL = model_name
    sys.path.insert(0, '/data/lab/CareJudge')
    os.chdir('/data/lab/CareJudge')
    from care_judge.judges.local_hf import LocalHFJudge
    _judge = LocalHFJudge(model_name=model_name, max_new_tokens=80)
    print(f"[Worker {os.getpid()}] Loaded {model_name}", flush=True)

def process_item(item_dict):
    global _judge
    from care_judge.schemas import PairItem
    from care_judge.uncertainty.collector_extended import collect_with_call_trace
    item = PairItem(
        id=item_dict['id'], prompt=item_dict['prompt'],
        response_a=item_dict['response_a'], response_b=item_dict['response_b'],
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
    ap.add_argument('--model', required=True, help='HF model name')
    ap.add_argument('--tag', required=True, help='output filename prefix, e.g. qwen-7b')
    ap.add_argument('--bench', required=True, choices=['judgebench','tldr','rewardbench','lmaarena'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0, help='cap items for cheaper ladder (0=all)')
    args = ap.parse_args()

    sys.path.insert(0, '/data/lab/CareJudge'); os.chdir('/data/lab/CareJudge')
    items = [json.loads(l) for l in open(f'data/{args.bench}_2k.jsonl') if l.strip()]
    if args.limit > 0:
        items = items[:args.limit]
    print(f"Total items {args.tag}/{args.bench}: {len(items)}", flush=True)

    out_path = f'outputs/scale/{args.tag}_{args.bench}_features.jsonl'
    done_ids = set(); done = 0
    if os.path.exists(out_path):
        for line in open(out_path):
            line = line.strip()
            if not line: continue
            try:
                done_ids.add(json.loads(line)['id']); done += 1
            except: pass
    print(f"Already done: {done}", flush=True)
    remaining = [it for it in items if it['id'] not in done_ids]
    print(f"Remaining: {len(remaining)}", flush=True)
    if not remaining:
        print("Nothing to do."); return

    t0 = time.time(); nw = 0; nf = 0
    with open(out_path, 'a') as fout, Pool(args.workers, initializer=init_worker, initargs=(args.model,)) as pool:
        for result in pool.imap(process_item, remaining, chunksize=2):
            if result is not None:
                fout.write(result + '\n'); fout.flush(); nw += 1
            else:
                nf += 1
            if (nw + nf) % 25 == 0:
                el = time.time() - t0
                print(f"  {done+nw+nf}/{len(items)} ({nw}w {nf}f) rate={ (nw+nf)/el*3600:.0f}/h", flush=True)
    print(f"DONE {args.tag}/{args.bench}: {nw} written, {nf} failed in {(time.time()-t0)/60:.1f}min", flush=True)
    print(f"Final: {sum(1 for _ in open(out_path))} records", flush=True)

if __name__ == '__main__':
    main()
