#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    manifest = read_jsonl(Path(args.manifest))
    states = {row["sample_id"]: row for row in read_jsonl(Path(args.states))}
    eligible = []
    for row in manifest:
        state = states[row["sample_id"]]
        if state["state"] == "POINT" and state.get("gt_within_profile_support") is True:
            eligible.append(row)
    selected = []
    missing = []
    trees = sorted({row["tree_id"] for row in manifest})
    sizes = sorted({int(row["image_size"][0]) for row in manifest})
    for tree in trees:
        for size in sizes:
            for inside in (False, True):
                candidates = [row for row in eligible if row["tree_id"] == tree and int(row["image_size"][0]) == size and bool(row.get("metadata", {}).get("pith_in_patch")) == inside]
                if not candidates:
                    missing.append({"tree_id": tree, "size": size, "pith_in_patch": inside})
                    continue
                candidates.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['sample_id']}".encode()).hexdigest())
                chosen = candidates[0]
                chosen.setdefault("metadata", {})["contribution_sentinel_stratum"] = {"tree_id": tree, "crop_size_px": size, "pith_in_patch": inside}
                selected.append(chosen)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "eligible_samples": len(eligible),
        "selected_samples": len(selected),
        "selected_by_tree": dict(Counter(row["tree_id"] for row in selected)),
        "selected_by_size": dict(Counter(str(row["image_size"][0]) for row in selected)),
        "selected_by_pith_in_patch": dict(Counter(str(bool(row.get("metadata", {}).get("pith_in_patch"))) for row in selected)),
        "missing_strata": missing,
        "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
