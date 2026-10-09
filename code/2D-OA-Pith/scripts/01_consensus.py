"""Build sub-pixel consensus centerlines from matched annotator curves.

Edit the settings below, then run this file directly with Python.  No command-
line arguments are read.
"""

from __future__ import annotations

import os
from pathlib import Path

from oapith.workflows import run_consensus


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- Experiment settings: edit these paths for your data. ----
INPUT_MANIFEST = PROJECT_ROOT / "data/raw_multi_annotator.jsonl"
OUTPUT_MANIFEST = PROJECT_ROOT / "data/consensus.jsonl"


def main() -> None:
    os.chdir(PROJECT_ROOT)
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    run_consensus(INPUT_MANIFEST, OUTPUT_MANIFEST)
    print(f"Consensus manifest saved to: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
