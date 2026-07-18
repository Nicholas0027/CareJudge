#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten_report(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    rows = []
    care = dict(report.get("care_selective", {}))
    care.update(report.get("care_calibration", {}))
    care["method"] = "CARE"
    care["dataset"] = Path(report.get("dataset_path", path.parent.name)).stem
    rows.append(care)
    for b in report.get("baselines", []):
        row = dict(b)
        row["method"] = b.get("baseline", "baseline")
        row["dataset"] = care["dataset"]
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Create CSV table from experiment report.json files")
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    for item in args.reports:
        path = Path(item)
        if path.is_dir():
            path = path / "report.json"
        rows.extend(flatten_report(path))
    cols = sorted({k for r in rows for k in r})
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = [str(r.get(c, "")).replace("\n", " ").replace(",", ";") for c in cols]
            f.write(",".join(vals) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
