#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated one-crop config/manifest for subarc contribution smoke testing"
    )
    parser.add_argument("--config-template", required=True)
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--crop-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subarc-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.subarc_fraction <= 1.0:
        raise ValueError("--subarc-fraction must lie in (0,1]")

    rows = [row for row in read_jsonl(Path(args.crop_manifest)) if str(row["crop_id"]) == args.crop_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one manifest row for {args.crop_id}, found {len(rows)}")

    config = json.loads(Path(args.config_template).read_text(encoding="utf-8"))
    config["protocol_name"] = "RAC-Pith-v2-grayscale-mask-subarc-contribution-smoke"
    config["calibration_status"] = "PROVISIONAL_LOCAL_CONTRIBUTION_SMOKE_ONLY"
    contribution = config["contribution"]
    contribution["levels"] = ["ring", "subarc"]
    contribution["primary_subarc_fraction"] = args.subarc_fraction
    contribution["audit_subarc_fractions"] = [args.subarc_fraction]
    contribution["audit_phases"] = [0.0]
    contribution["maximum_crops_per_tree"] = 0
    contribution["replay_count"] = 0

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(output / "crop_manifest.jsonl", rows)
    print(f"prepared contribution smoke input for {args.crop_id} at {output}")


if __name__ == "__main__":
    main()
