"""Run pith inference, uncertainty analysis, and conditional arc contribution.

Edit the settings below, then run this file directly with Python.  No command-
line arguments are read.
"""

from __future__ import annotations

import os
from pathlib import Path

from oapith.workflows import run_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- Required experiment settings. ----
CONFIG_PATH = PROJECT_ROOT / "configs/base.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "runs/oa_pith/best.pt"
IMAGE_PATH = PROJECT_ROOT / "sample.png"
OUTPUT_JSON = PROJECT_ROOT / "runs/oa_pith/prediction.json"
OVERLAY_PNG: Path | None = PROJECT_ROOT / "runs/oa_pith/contribution.png"

# ---- Optional calibration/physical-scale settings. ----
# Set this to a fitted conformal_95.json after running scripts/04_calibrate.py.
CALIBRATOR_PATH: Path | None = None
# Keep None unless the true square-pixel spacing is known.  If supplied, X and Y
# must be equal in the current implementation, e.g. (0.042, 0.042).
MM_PER_PIXEL: tuple[float, float] | None = None

# ---- Contribution / uncertainty audit settings. ----
EXACT_CONTRIBUTIONS = True
BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_CORRELATION = 0.92
DEVICE: str | None = None  # None = CUDA when available, otherwise CPU.
ALLOW_LEGACY_CHECKPOINT = False


def main() -> None:
    os.chdir(PROJECT_ROOT)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if OVERLAY_PNG is not None:
        OVERLAY_PNG.parent.mkdir(parents=True, exist_ok=True)
    run_inference(
        CONFIG_PATH,
        CHECKPOINT_PATH,
        IMAGE_PATH,
        OUTPUT_JSON,
        overlay_path=OVERLAY_PNG,
        calibrator_path=CALIBRATOR_PATH,
        device=DEVICE,
        allow_legacy_checkpoint=ALLOW_LEGACY_CHECKPOINT,
        mm_per_pixel=MM_PER_PIXEL,
        exact_contributions=EXACT_CONTRIBUTIONS,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_correlation=BOOTSTRAP_CORRELATION,
    )
    print(f"Prediction saved to: {OUTPUT_JSON}")
    if OVERLAY_PNG is not None:
        print(f"Overlay saved to: {OVERLAY_PNG}")


if __name__ == "__main__":
    main()
