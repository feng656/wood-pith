#!/usr/bin/env python3
"""Resolve configured RAC-Pith data and output paths for shell entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from racpith.config import load_config, resolve_runtime_paths


FIELDS = {
    "dataset_root",
    "output_root",
    "prepared_root",
    "development_root",
    "sealed_root",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one resolved runtime path")
    parser.add_argument("--config", required=True)
    parser.add_argument("--field", required=True, choices=sorted(FIELDS))
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    runtime = resolve_runtime_paths(
        load_config(args.config),
        project_root=project_root,
    )
    print(getattr(runtime, args.field))


if __name__ == "__main__":
    main()
