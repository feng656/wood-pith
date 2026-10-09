#!/usr/bin/env python3
"""Create or validate the one-time sealed-test opening marker."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from racpith.config import load_config
from racpith.provenance import atomic_write_json, read_json_object


def main() -> None:
    parser = argparse.ArgumentParser(description="Open one frozen sealed-test run, or resume the same run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()
    frozen = load_config(args.config)
    if frozen.data.get("calibration_status") != "DEVELOPMENT_FROZEN":
        raise ValueError("cannot open sealed test with a provisional configuration")
    marker = Path(args.marker).expanduser().resolve()
    if marker.exists():
        existing = read_json_object(marker)
        if existing.get("config_hash") != frozen.sha256 or existing.get("run_id") != args.run_id:
            raise ValueError(
                "sealed test was already opened under a different run/config; only exact resume is allowed"
            )
        print(f"resuming sealed run {args.run_id} with unchanged config")
        return
    atomic_write_json(
        marker,
        {
            "schema_version": "racpith.sealed_opening.v1",
            "opened_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": args.run_id,
            "config_hash": frozen.sha256,
            "config_path": str(Path(args.config).expanduser().resolve()),
        },
    )
    print(f"sealed test opened once for run {args.run_id}; marker={marker}")


if __name__ == "__main__":
    main()
