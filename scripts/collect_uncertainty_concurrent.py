#!/usr/bin/env python3
"""Concurrent uncertainty collection for API judges (thread pool), lean call budget.

Per example: base + (k_self-1) self + swap + n_rubrics rubric calls.
Simulated annotators optional. Designed to keep API runs under ~1 hour.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.data.loaders import load_jsonl_pairs
from care_judge.judges.factory import make_judge
from care_judge.judges.base import swap_item, unswap_winner
from care_judge.judges.prompts import BASE_RUBRIC, RUBRIC_VARIANTS
from care_judge.schemas import PairItem
from care_judge.utils import entropy_binary, majority, vote_share, write_jsonl


def std(v):
    if not v: return 0.0
    mu=sum(v)/len(v); return math.sqrt(sum((x-mu)**2 for x in v)/len(v))


def build_item_features(item, judge, k_self, n_rubrics, temperature, use_swap):
    base=judge.judge(item, rubric=BASE_RUBRIC, temperature=0.0)
    votes=[base.normalized_winner()]; confs=[base.confidence]
    calls=1
    for _ in range(max(0,k_self-1)):
        j=judge.judge(item, rubric=BASE_RUBRIC, temperature=temperature)
        votes.append(j.normalized_winner()); confs.append(j.confidence); calls+=1
    self_share=vote_share(votes)
    swap_consistency=1.0; swap_gap=0.0
    if use_swap:
        sw=judge.judge(swap_item(item), rubric=BASE_RUBRIC, temperature=0.0)
        mapped=unswap_winner(sw.normalized_winner())
        swap_consistency=1.0 if mapped==base.normalized_winner() else 0.0
        swap_gap=abs(base.confidence-sw.confidence); calls+=1
    rub_votes=[]; rub_confs=[]
    for rub in RUBRIC_VARIANTS[:n_rubrics]:
        j=judge.judge(item, rubric=rub, temperature=0.0)
        rub_votes.append(j.normalized_winner()); rub_confs.append(j.confidence); calls+=1
    rub_share=vote_share(rub_votes)
    p_a=max(1e-9,min(1-1e-9,rub_votes.count('A')/max(1,len(rub_votes))))
    pred=majority(votes+rub_votes)
    if pred=='tie': pred=base.normalized_winner()
    all_conf=confs+rub_confs
    feats={
        'base_conf':float(base.confidence),
        'mean_conf':sum(all_conf)/max(1,len(all_conf)),
        'std_conf':std(all_conf),
        'self_vote_share':self_share,'self_entropy':entropy_binary(self_share),'adaptive_calls':float(calls),
        'swap_consistency':swap_consistency,'swap_conf_gap':swap_gap,
        'rubric_vote_share':rub_share,'rubric_entropy':entropy_binary(p_a),
        'rubric_flip':1.0 if len(set(rub_votes))>1 else 0.0,
        'sim_vote_share':self_share,'sim_entropy':entropy_binary(self_share),'sim_flip':0.0,
        'length_gap_norm':abs(len(item.response_a)-len(item.response_b))/max(1,len(item.response_a)+len(item.response_b)),
        'cost':0.0,
    }
    confidence=sum([feats['base_conf'],feats['self_vote_share'],feats['rubric_vote_share'],feats['swap_consistency']])/4
    row={'id':item.id,'pred':pred,'label':item.label,
         'correct':None if item.label is None else int(pred==item.label),
         'confidence':confidence,'domain':item.domain}
    row.update({f'feat_{k}':v for k,v in feats.items()})
    return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--judge', required=True)
    ap.add_argument('--limit', type=int, default=120)
    ap.add_argument('--k-self', type=int, default=2)
    ap.add_argument('--n-rubrics', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-swap', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args=ap.parse_args()
    items=load_jsonl_pairs(args.input, limit=None)
    random.Random(args.seed).shuffle(items)
    items=items[:args.limit]
    judge=make_judge(args.judge)
    rows=[]; done=0; lock=threading.Lock(); t0=time.time()
    def work(it):
        return build_item_features(it, judge, args.k_self, args.n_rubrics, args.temperature, not args.no_swap)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(work, it): it for it in items}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                print('ITEM_ERR', type(e).__name__, str(e)[:120])
            with lock:
                done+=1
                if done%20==0 or done==len(items):
                    print(f'{done}/{len(items)} elapsed={time.time()-t0:.0f}s', flush=True)
    write_jsonl(args.out, rows)
    print(f'wrote {len(rows)} rows to {args.out} in {time.time()-t0:.0f}s')


if __name__=='__main__':
    main()
