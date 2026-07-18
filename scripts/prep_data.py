#!/usr/bin/env python3
"""prep_data.py — Download JudgeBench + RewardBench and normalize to CARE JSONL.

Output schema (matches care_judge/data/loaders.py aliases):
  {"id","prompt","response_a","response_b","label","domain"}

JudgeBench: concat claude(270)+gpt(350)=620 pairs. Fields: question/pair_id/response_A/response_B/label/source.
RewardBench: filtered split (2985) -> seeded shuffle -> 1000-pair reproducible subset for the ladder.
"""
import json, random, os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

# ---- JudgeBench ----
jb = load_dataset("ScalerLab/JudgeBench")
rows = []
for split in ["gpt", "claude"]:
    for r in jb[split]:
        lab = str(r.get("label", "")).strip().upper()
        if lab in {"A>B", "A"}:
            lab = "A"
        elif lab in {"B>A", "B"}:
            lab = "B"
        else:
            continue
        rows.append({
            "id": str(r.get("pair_id", f"jb_{len(rows)}")),
            "prompt": r.get("question", ""),
            "response_a": r.get("response_A", ""),
            "response_b": r.get("response_B", ""),
            "label": lab,
            "domain": f"judgebench_{split}",
        })
with open("data/judgebench.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"JudgeBench: wrote {len(rows)} rows -> data/judgebench.jsonl")

# ---- RewardBench (1000-pair reproducible subset) ----
rb = load_dataset("allenai/reward-bench", split="filtered")
rb_rows = []
for i, r in enumerate(rb):
    rb_rows.append({
        "id": str(r.get("id", f"rb_{i}")),
        "prompt": r.get("prompt", ""),
        "response_a": r.get("chosen", ""),     # chosen = A
        "response_b": r.get("rejected", ""),   # rejected = B
        "label": "A",
        "domain": str(r.get("subset", "rewardbench")),
    })
# seeded shuffle for a representative 1000-pair subset
rng = random.Random(42)
rng.shuffle(rb_rows)
subset = rb_rows[:1000]
with open("data/rewardbench_1k.jsonl", "w") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
# also keep full for reference
with open("data/rewardbench_full.jsonl", "w") as f:
    for r in rb_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"RewardBench: wrote 1000-pair subset -> data/rewardbench_1k.jsonl (full {len(rb_rows)} -> data/rewardbench_full.jsonl)")

# sanity: label distribution
from collections import Counter
print("JudgeBench labels:", Counter(r["label"] for r in rows))
print("RewardBench subset domains:", Counter(r["domain"] for r in subset))
print("Sample JB row keys:", list(rows[0].keys()))
print("Sample RB row keys:", list(subset[0].keys()))
