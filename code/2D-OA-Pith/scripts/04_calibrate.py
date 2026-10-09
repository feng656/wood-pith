"""Fit the tree/disc-grouped conformal threshold through the Python API.

The input is the same calibration-prediction JSONL contract documented in
``docs/python_script_experiments.md``.  Edit the settings below and run directly.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from oapith.workflows import run_calibration


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- Experiment settings. ----
CALIBRATION_MANIFEST = PROJECT_ROOT / "data/calibration_predictions.jsonl"
OUTPUT_CALIBRATOR = PROJECT_ROOT / "runs/oa_pith/conformal_95.json"
ALPHA = 0.05
GROUP_REDUCTION = "max"  # "max" is the conservative default; "first" is supported.


def main() -> None:
    os.chdir(PROJECT_ROOT)
    OUTPUT_CALIBRATOR.parent.mkdir(parents=True, exist_ok=True)
    calibrator = run_calibration(
        CALIBRATION_MANIFEST,
        OUTPUT_CALIBRATOR,
        alpha=ALPHA,
        group_reduction=GROUP_REDUCTION,
    )
    threshold = calibrator.threshold
    threshold_text = "infinity" if threshold is not None and math.isinf(threshold) else threshold
    print(f"Calibrator saved to: {OUTPUT_CALIBRATOR}")
    print(f"groups={calibrator.number_groups}, threshold={threshold_text}")


if __name__ == "__main__":
    main()
