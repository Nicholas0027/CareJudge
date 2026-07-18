#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_judge.utils import normalize_label, read_jsonl, write_jsonl


def _get(row: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def convert_generic(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        prompt = _get(row, ["prompt", "question", "query", "instruction", "input"])
        a = _get(row, ["response_a", "answer_a", "output_a", "response_1", "chosen"])
        b = _get(row, ["response_b", "answer_b", "output_b", "response_2", "rejected"])
        label = normalize_label(_get(row, ["label", "winner", "preference", "gold"], None))
        if label is None and "chosen" in row and "rejected" in row:
            label = "A"
        out.append({"id": str(_get(row, ["id", "example_id", "uid"], i)), "prompt": prompt, "response_a": a, "response_b": b, "label": label, "domain": str(_get(row, ["domain", "category", "subset"], "general"))})
    return out


def convert_judgebench(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        # JudgeBench-like files vary; keep fallbacks broad.
        prompt = _get(row, ["question", "prompt", "query"])
        a = _get(row, ["response_A", "response_a", "answer_A", "answer_a", "A"])
        b = _get(row, ["response_B", "response_b", "answer_B", "answer_b", "B"])
        winner = _get(row, ["label", "winner", "decision", "gold"], None)
        label = normalize_label(winner)
        if label is None and str(winner).upper() in {"A>B", "A"}:
            label = "A"
        if label is None and str(winner).upper() in {"B>A", "B"}:
            label = "B"
        out.append({"id": str(_get(row, ["id", "example_id"], i)), "prompt": prompt, "response_a": a, "response_b": b, "label": label, "domain": str(_get(row, ["domain", "category", "dataset"], "judgebench"))})
    return out


def convert_rewardbench2(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        prompt = _get(row, ["prompt", "query", "instruction"])
        responses = _get(row, ["responses", "answers", "completions"], None)
        if isinstance(responses, list) and len(responses) >= 2:
            chosen_idx = int(_get(row, ["chosen_idx", "winner_idx", "best_idx"], 0) or 0)
            for j, resp in enumerate(responses):
                if j == chosen_idx:
                    continue
                out.append({"id": f"{_get(row, ['id'], i)}_{chosen_idx}_vs_{j}", "prompt": prompt, "response_a": responses[chosen_idx], "response_b": resp, "label": "A", "domain": str(_get(row, ["subset", "category"], "rewardbench2"))})
        else:
            out.extend(convert_generic([row]))
    return out


def load_input(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "examples", "rows"]:
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("Unsupported JSON structure")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert common judge datasets to CARE pairwise JSONL")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["generic", "judgebench", "rewardbench2", "lmarena", "tldr", "contextual_judge", "if_rewardbench"], default="generic")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    rows = load_input(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.format == "judgebench":
        out = convert_judgebench(rows)
    elif args.format == "rewardbench2":
        out = convert_rewardbench2(rows)
    else:
        # LMArena, TL;DR, ContextualJudgeBench, IF-RewardBench are accepted if
        # they expose prompt/chosen/rejected or prompt/response_a/response_b.
        out = convert_generic(rows)
    write_jsonl(args.out, out)
    print(f"converted {len(out)} examples to {args.out}")


if __name__ == "__main__":
    main()
