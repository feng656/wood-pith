"""RAC-Pith v2 implementation.

The package deliberately keeps ground truth outside the estimator.  Ground truth
is consumed only by evaluation and contribution-label code.
"""

from .config import FrozenConfig, load_config
from .contracts import EvidenceBundle, GeometryState, LocateResult
from .estimator import RacPithEstimator

__all__ = [
    "EvidenceBundle",
    "FrozenConfig",
    "GeometryState",
    "LocateResult",
    "RacPithEstimator",
    "load_config",
]

__version__ = "0.1.0"

