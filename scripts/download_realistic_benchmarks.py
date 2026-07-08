#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick(row: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def norm_label(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in {"A", "0", "1", "LEFT", "RESPONSE_A", "A>B", "A > B"}:
        return "A"
    if s in {"B", "2", "RIGHT", "RESPONSE_B", "B>A", "B > A"}:
        return "B"
    return None


def generic(rows: Iterable[Dict[str, Any]], domain: str, limit: int | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if limit is not None and len(out) >= limit:
            break
        prompt = pick(row, ["prompt", "question", "query", "instruction", "input"])
        a = pick(row, ["response_a", "response_A", "answer_a", "answer_A", "output_a", "response_1", "chosen"])
        b = pick(row, ["response_b", "response_B", "answer_b", "answer_B", "output_b", "response_2", "rejected"])
        label = norm_label(pick(row, ["label", "winner", "preference", "gold", "decision"], None))
        if label is None and "chosen" in row and "rejected" in row:
            label = "A"
        if prompt and a and b and label in {"A", "B"}:
            out.append({"id": str(pick(row, ["id", "example_id", "uid"], f"{domain}_{i}")), "prompt": str(prompt), "response_a": str(a), "response_b": str(b), "label": label, "domain": str(pick(row, ["domain", "category", "subset"], domain))})
    return out


def load_dataset_safe(name: str, split: str, config: str | None = None):
    from datasets import load_dataset
    if config:
        return load_dataset(name, config, split=split)
    return load_dataset(name, split=split)


def judgebench(limit: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ["gpt", "claude"]:
        try:
            ds = load_dataset_safe("ScalerLab/JudgeBench", split)
            rows.extend(generic((dict(x) for x in ds), f"judgebench_{split}", None))
        except Exception as exc:
            print(f"[WARN] JudgeBench/{split} failed: {exc}")
    return rows[:limit] if limit else rows


def lmarena(limit: int | None) -> List[Dict[str, Any]]:
    for name, split in [("sumuks/lmarena", "train"), ("lmsys/chatbot_arena_conversations", "train")]:
        try:
            ds = load_dataset_safe(name, split)
            rows = generic((dict(x) for x in ds), "lmarena", limit)
            if rows:
                return rows
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}")
    return []


def tldr(limit: int | None) -> List[Dict[str, Any]]:
    candidates = [("openai/summarize_from_feedback", "comparisons", None), ("CarperAI/openai_summarize_comparisons", "train", None)]
    for name, split, config in candidates:
        try:
            ds = load_dataset_safe(name, split, config)
            out: List[Dict[str, Any]] = []
            for i, row0 in enumerate(ds):
                if limit is not None and len(out) >= limit:
                    break
                row = dict(row0)
                prompt = pick(row, ["prompt", "post", "article", "input"])
                if isinstance(prompt, dict):
                    prompt = pick(prompt, ["post", "title", "text"], json.dumps(prompt))
                summaries = pick(row, ["summaries", "responses", "completions"], None)
                choice = pick(row, ["choice", "chosen", "label", "winner"], None)
                if isinstance(summaries, list) and len(summaries) >= 2:
                    a, b = summaries[0], summaries[1]
                    if isinstance(a, dict):
                        a = pick(a, ["text", "summary", "response"], json.dumps(a))
                    if isinstance(b, dict):
                        b = pick(b, ["text", "summary", "response"], json.dumps(b))
                    label = "A" if str(choice) in {"0", "A", "a"} else "B" if str(choice) in {"1", "B", "b"} else None
                    if prompt and a and b and label:
                        out.append({"id": str(pick(row, ["id"], f"tldr_{i}")), "prompt": str(prompt), "response_a": str(a), "response_b": str(b), "label": label, "domain": "tldr"})
                else:
                    out.extend(generic([row], "tldr", 1))
            if out:
                return out[:limit] if limit else out
        except Exception as exc:
            print(f"[WARN] {name} failed: {exc}")
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/realistic")
    p.add_argument("--judgebench-limit", type=int, default=620)
    p.add_argument("--lmarena-limit", type=int, default=300)
    p.add_argument("--tldr-limit", type=int, default=300)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {"judgebench": judgebench(args.judgebench_limit), "lmarena": lmarena(args.lmarena_limit), "tldr": tldr(args.tldr_limit)}
    summary = {}
    for name, rows in datasets.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        summary[name] = {"n": len(rows), "path": str(path)}
        print(f"{name}: {len(rows)} rows -> {path}")
    with open(out_dir / "download_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
