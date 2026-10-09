"""2D-OA-Pith: uncertainty-aware pith inference for partial cross-sections."""

from .types import ArcObservation, CoordinateFrame, PithState

CHECKPOINT_SCHEMA_VERSION = 3
COORDINATE_CONTRACT = "isotropic-longest-edge-pixel-center-align-corners-false-v1"

__all__ = [
    "ArcObservation",
    "CoordinateFrame",
    "PithState",
    "CHECKPOINT_SCHEMA_VERSION",
    "COORDINATE_CONTRACT",
]
__version__ = "0.1.0"
