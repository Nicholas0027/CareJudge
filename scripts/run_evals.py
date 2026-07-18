#!/usr/bin/env python3
"""run_evals.py — Post-collection: merge 14B halves, run rigorous_analysis.py on every
feature file, then extract paper-table cells into a single summary.

Run this AFTER all collect_6call jobs have finished.
"""
import json, os, subprocess, sys, glob

LAB = "/data/lab/CareJudge"
OUT = f"{LAB}/outputs"
RIG = f"{OUT}/rigorous"
os.makedirs(RIG, exist_ok=True)

# (feature_file, dataset_tag) pairs. 14B files are merged from halves first.
JOBS = [
    ("outputs/qwen15b_jb.jsonl", "qwen15b_jb"),
    ("outputs/qwen15b_rb.jsonl", "qwen15b_rb"),
    ("outputs/qwen3b_jb.jsonl",  "qwen3b_jb"),
    ("outputs/qwen3b_rb.jsonl",  "qwen3b_rb"),
    ("outputs/qwen7b_jb.jsonl",  "qwen7b_jb"),
    ("outputs/qwen7b_rb.jsonl",  "qwen7b_rb"),
    ("outputs/mistral7b_jb.jsonl","mistral7b_jb"),
    ("outputs/qwen14b_jb.jsonl", "qwen14b_jb"),   # merged from h1+h2
    ("outputs/qwen14b_rb.jsonl", "qwen14b_rb"),   # merged from h1+h2
]

def merge(h1, h2, out):
    if os.path.exists(out):
        print(f"[merge] {out} exists, skip"); return
    rows = []
    for f in (h1, h2):
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f)]
        else:
            print(f"[merge] WARN missing {f}")
    with open(out, "w") as o:
        for r in rows: o.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] {h1}+{h2} -> {out} ({len(rows)} rows)")

def main():
    os.chdir(LAB)
    # 1. merge 14B halves
    merge(f"{OUT}/qwen14b_jb_h1.jsonl", f"{OUT}/qwen14b_jb_h2.jsonl", f"{OUT}/qwen14b_jb.jsonl")
    merge(f"{OUT}/qwen14b_rb_h1.jsonl", f"{OUT}/qwen14b_rb_h2.jsonl", f"{OUT}/qwen14b_rb.jsonl")

    # 2. run rigorous_analysis.py on each feature file
    for feat, tag in JOBS:
        feat_path = f"{LAB}/{feat}"
        if not os.path.exists(feat_path):
            print(f"[skip] {feat} missing"); continue
        nrow = sum(1 for _ in open(feat_path))
        print(f"[rigorous] {tag} ({nrow} rows) ...", flush=True)
        cmd = ["python3", "scripts/rigorous_analysis.py",
               "--features", feat_path, "--dataset", tag,
               "--out-dir", RIG, "--seeds", "20", "--delta", "0.10", "--min-keep", "20"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[rigorous] FAIL {tag}: {r.stderr[-500:]}"); 
        else:
            print(f"[rigorous] OK {tag} -> {RIG}/{tag}_rigorous.json")

    # 3. extract cells summary
    print("\n=== SUMMARY (alpha=0.15) ===")
    print(f"{'dataset':<18} {'AUROC':<14} {'best1':<8} {'Δ':<7} {'cov':<6} {'acc':<6} {'viol':<5}")
    for _, tag in JOBS:
        rj = f"{RIG}/{tag}_rigorous.json"
        if not os.path.exists(rj):
            print(f"{tag:<18} MISSING"); continue
        d = json.load(open(rj))
        ma = d.get("method_auroc", {})
        lg = ma.get("logistic", {}); b1 = ma.get("best_single", {})
        bma = d.get("by_method_alpha", {}).get("logistic@0.15", {})
        delta = (lg.get("mean",0) - b1.get("mean",0))
        print(f"{tag:<18} {lg.get('mean',0):.3f}±{lg.get('std',0):.3f}  {b1.get('mean',0):.3f}   {delta:+.3f}  {bma.get('pooled_cov',0):.3f}  {bma.get('pooled_acc',0):.3f}  {bma.get('violation_rate',0):.2f}")

if __name__ == "__main__":
    main()
