"""Programmatic entry points for the reproducible experiment workflow.

These functions expose the same implementation used by the compatibility CLI,
but accept ordinary Python values instead of parsing command-line arguments.
"""

from oapith.cli.calibrate import run_calibration
from oapith.cli.consensus import run_consensus
from oapith.cli.infer import run_inference
from oapith.cli.train import run_training

__all__ = [
    "run_calibration",
    "run_consensus",
    "run_inference",
    "run_training",
]
