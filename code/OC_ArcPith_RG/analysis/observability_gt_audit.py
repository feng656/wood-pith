#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oc_arcpith_rg.config import load_config
from oc_arcpith_rg.geometry import h_from_point, to_norm
from oc_arcpith_rg.io import sample_from_dict
from oc_arcpith_rg.objectives import objective
from oc_arcpith_rg.preprocess import prepare


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--model", default="m0", choices=["m0", "m1rpc"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = read_jsonl(Path(args.manifest))
    states = {row["sample_id"]: row for row in read_jsonl(Path(args.states))}
    if set(states) != {row["sample_id"] for row in manifest}:
        raise ValueError("states and manifest sample IDs do not match")
    delta = float(cfg["profile"]["support_delta"])
    output = []
    for index, source in enumerate(manifest, 1):
        prep = prepare(sample_from_dict(source), cfg)
        row = states[prep.sample.sample_id]
        if prep.sample.pith_px is not None:
            gt = to_norm(prep.sample.pith_px, prep.center_px, prep.scale_px)
            gt_loss = float(objective(h_from_point(gt), prep, cfg, args.model))
            row["gt_loss"] = gt_loss
            row["gt_loss_gap"] = float(gt_loss - row["min_loss"])
            row["gt_within_profile_support"] = bool(gt_loss <= row["min_loss"] + delta)
        output.append(row)
        if index % 500 == 0:
            print(f"GT profile audit {index}/{len(manifest)}", flush=True)
    path = Path(args.states)
    with path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "samples": len(output),
        "gt_within_profile_support": sum(row.get("gt_within_profile_support") is True for row in output),
        "support_delta": delta,
        "states": str(path.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
