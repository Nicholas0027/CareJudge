#!/usr/bin/env python3
"""extract_cells.py — extract paper-table cells from a *_rigorous.json file.

Usage: python3 extract_cells.py <rigorous.json> [alpha=0.15]
Prints: AUROC mean±std (logistic), pooled coverage, pooled accepted acc, violation rate,
        plus per-method AUROC and the best-single-signal AUROC for the Δ column.
"""
import json, sys, os

path = sys.argv[1]
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
d = json.load(open(path))
ds = d.get("dataset", os.path.basename(path))

ma = d.get("method_auroc", {})
log = ma.get("logistic", {})
best1 = ma.get("best_single", {})
base = ma.get("base_conf", {})
rub = ma.get("rubric", {})
swp = ma.get("swap", {})
slf = ma.get("self", {})

key = f"logistic@{alpha}"
bma = d.get("by_method_alpha", {}).get(key, {})

print(f"=== {ds} (alpha={alpha}) ===")
print(f"  AUROC (logistic fusion): {log.get('mean'):.3f} ± {log.get('std'):.3f}")
print(f"  AUROC best_single:       {best1.get('mean'):.3f} ± {best1.get('std'):.3f}")
print(f"  AUROC base_conf:         {base.get('mean'):.3f}")
print(f"  AUROC rubric:            {rub.get('mean'):.3f}")
print(f"  AUROC swap:              {swp.get('mean'):.3f}")
print(f"  AUROC self:              {slf.get('mean'):.3f}")
print(f"  Delta (fusion - best1):  {log.get('mean') - best1.get('mean'):+.3f}")
print(f"  Pooled coverage:         {bma.get('pooled_cov'):.3f}")
print(f"  Pooled accepted acc:     {bma.get('pooled_acc'):.3f}")
print(f"  Violation rate:          {bma.get('violation_rate'):.2f}")
print(f"  n_total: {d.get('n_total')}  seeds: {d.get('seeds')}")
