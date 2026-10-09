"""Train 2D-OA-Pith through the programmatic Python API.

Edit the settings below, then run this file directly with Python.  Dataset and
model hyperparameters remain in ``configs/base.yaml`` so the saved checkpoint
keeps the same configuration/coordinate contract as the original implementation.
"""

from __future__ import annotations

import os
from pathlib import Path

from oapith.workflows import run_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- Experiment settings. ----
CONFIG_PATH = PROJECT_ROOT / "configs/base.yaml"
RESUME_CHECKPOINT: Path | None = None
DEVICE: str | None = None  # None = CUDA when available, otherwise CPU; e.g. "cuda:0".
ALLOW_LEGACY_CHECKPOINT = False


def main() -> None:
    os.chdir(PROJECT_ROOT)
    run_training(
        CONFIG_PATH,
        resume=RESUME_CHECKPOINT,
        allow_legacy_checkpoint=ALLOW_LEGACY_CHECKPOINT,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
