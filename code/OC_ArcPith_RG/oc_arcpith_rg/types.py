from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class Ring:
    ring_id: str
    points_px: np.ndarray
    order: int | None = None

@dataclass
class Sample:
    sample_id: str
    tree_id: str
    image_size: tuple[int,int]
    rings: list[Ring]
    pith_px: np.ndarray | None = None
    mm_per_pixel: float | None = None
    metadata: dict[str,Any] | None = None

@dataclass
class Arc:
    arc_id: str
    ring_id: str
    points: np.ndarray
    tangents: np.ndarray
    ds: np.ndarray
    omega: float

@dataclass
class PreparedSample:
    sample: Sample
    center_px: np.ndarray
    scale_px: float
    # One parent ring may enter a crop as several disconnected visible fragments.
    # Keep them separate: concatenating them would invent a chord across missing data.
    points_by_ring: dict[str,list[np.ndarray]]
    tangents_by_ring: dict[str,list[np.ndarray]]
    arcs: list[Arc]
    ring_order: list[str]
