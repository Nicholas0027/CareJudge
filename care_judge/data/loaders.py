from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from care_judge.schemas import PairItem
from care_judge.utils import normalize_label, read_jsonl


def _pick(row: Dict[str, Any], names: List[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def load_jsonl_pairs(path: str | Path, limit: Optional[int] = None) -> List[PairItem]:
    """Load a flexible pairwise JSONL format.

    Supported field aliases:
      prompt: prompt, question, query, instruction, input
      response_a: response_a, answer_a, chosen, response_1, output_a
      response_b: response_b, answer_b, rejected, response_2, output_b
      label: label, winner, preference, gold

    If rows use chosen/rejected and no label is present, label defaults to A.
    """
    items: List[PairItem] = []
    for i, row in enumerate(read_jsonl(path)):
        if limit is not None and len(items) >= limit:
            break
        prompt = _pick(row, ["prompt", "question", "query", "instruction", "input"])
        a = _pick(row, ["response_a", "answer_a", "output_a", "response_1", "chosen"])
        b = _pick(row, ["response_b", "answer_b", "output_b", "response_2", "rejected"])
        label = normalize_label(_pick(row, ["label", "winner", "preference", "gold"], None))
        if label is None and "chosen" in row and "rejected" in row:
            label = "A"
        items.append(
            PairItem(
                id=str(_pick(row, ["id", "example_id", "uid"], i)),
                prompt=str(prompt),
                response_a=str(a),
                response_b=str(b),
                label=label,
                domain=str(_pick(row, ["domain", "category", "subset"], "general")),
                metadata={k: v for k, v in row.items() if k not in {"prompt", "question", "query", "instruction", "input", "response_a", "answer_a", "output_a", "response_1", "chosen", "response_b", "answer_b", "output_b", "response_2", "rejected"}},
            )
        )
    return items


def load_hf_pairs(dataset_name: str, split: str = "train", limit: Optional[int] = None, **kwargs: Any) -> List[PairItem]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install with `pip install -e .[hf]` to load HuggingFace datasets") from exc
    ds = load_dataset(dataset_name, split=split, **kwargs)
    rows = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        rows.append(dict(row, id=row.get("id", i)))
    tmp = Path("outputs/_hf_tmp.jsonl")
    from care_judge.utils import write_jsonl
    write_jsonl(tmp, rows)
    return load_jsonl_pairs(tmp, limit=limit)
