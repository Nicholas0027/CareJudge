#!/usr/bin/env python3
"""download_models.py — robustly download all 5 judge models to the HF cache."""
import os, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import snapshot_download

models = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-14B-Instruct",
]
for m in models:
    t0 = time.time()
    try:
        p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "vocab*", "merges*"], max_workers=8)
        sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(p) for f in fn) / 1e9
        print(f"OK {m} in {time.time()-t0:.0f}s size={sz:.1f}GB", flush=True)
    except Exception as e:
        print(f"FAIL {m}: {repr(e)[:300]}", flush=True)
print("=== ALL DOWNLOADS DONE ===", flush=True)
