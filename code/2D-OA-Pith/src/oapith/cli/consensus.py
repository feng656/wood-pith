from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from oapith.data.consensus import collapse_multi_annotator_curves
from oapith.data.manifest import dump_manifest, load_manifest


def run_consensus(input_manifest: str | Path, output_manifest: str | Path) -> None:
    """Collapse matched multi-annotator curves without using the CLI."""
    records = load_manifest(input_manifest)
    collapsed = [
        replace(record, curves=collapse_multi_annotator_curves(record.curves))
        for record in records
    ]
    dump_manifest(collapsed, output_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build robust sub-pixel consensus centerlines from matched annotators"
    )
    parser.add_argument("--input", required=True, help="Input JSONL manifest")
    parser.add_argument("--output", required=True, help="Output JSONL manifest")
    args = parser.parse_args()
    run_consensus(args.input, args.output)


if __name__ == "__main__":
    main()
