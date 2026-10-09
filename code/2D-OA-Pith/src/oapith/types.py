from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from typing import Any

import torch


class PithState(IntEnum):
    """What the current observation supports, not the physical existence of a pith."""

    NEAR = 0
    FAR = 1
    INFINITY = 2
    NULL = 3


@dataclass(frozen=True)
class CoordinateFrame:
    """Isotropic pixel-center frame mapping a rectangular image into a square chart.

    The longest image extent maps to [-1, 1] at its *edges*. Pixel centers therefore
    lie just inside this interval (the ``align_corners=False`` convention). Points
    outside the image, including the pith, are deliberately not clipped.
    """

    width: int
    height: int
    mm_per_pixel_x: float | None = None
    mm_per_pixel_y: float | None = None

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("width and height must both be >= 2")
        if (self.mm_per_pixel_x is None) != (self.mm_per_pixel_y is None):
            raise ValueError("provide both pixel spacings or neither")
        if self.mm_per_pixel_x is not None:
            if self.mm_per_pixel_x <= 0 or self.mm_per_pixel_y <= 0:
                raise ValueError("pixel spacing must be positive")
            if not math.isclose(
                self.mm_per_pixel_x, self.mm_per_pixel_y, rel_tol=1e-6, abs_tol=1e-12
            ):
                raise ValueError(
                    "anisotropic physical pixels must be resampled to square pixels "
                    "before Euclidean ring fitting"
                )

    @property
    def center_px(self) -> tuple[float, float]:
        return ((self.width - 1.0) / 2.0, (self.height - 1.0) / 2.0)

    @property
    def scale_px(self) -> float:
        return max(self.width, self.height) / 2.0

    def pixel_to_normalized(self, xy: torch.Tensor) -> torch.Tensor:
        center = xy.new_tensor(self.center_px)
        return (xy - center) / self.scale_px

    def normalized_to_pixel(self, xy: torch.Tensor) -> torch.Tensor:
        center = xy.new_tensor(self.center_px)
        return xy * self.scale_px + center

    def covariance_pixel_to_normalized(self, covariance: torch.Tensor) -> torch.Tensor:
        return covariance / (self.scale_px**2)

    def covariance_normalized_to_pixel(self, covariance: torch.Tensor) -> torch.Tensor:
        return covariance * (self.scale_px**2)

    def curvature_pixel_to_normalized(self, curvature: torch.Tensor) -> torch.Tensor:
        # x_norm=x_px/s, hence kappa_norm=s*kappa_px.
        return curvature * self.scale_px

    def curvature_normalized_to_pixel(self, curvature: torch.Tensor) -> torch.Tensor:
        return curvature / self.scale_px

    def normalized_delta_to_mm(self, delta: torch.Tensor) -> torch.Tensor:
        if self.mm_per_pixel_x is None or self.mm_per_pixel_y is None:
            raise ValueError("physical pixel spacing is unavailable")
        scale = delta.new_tensor(
            [self.scale_px * self.mm_per_pixel_x, self.scale_px * self.mm_per_pixel_y]
        )
        return delta * scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "center_px": list(self.center_px),
            "scale_px": self.scale_px,
            "mm_per_pixel": [self.mm_per_pixel_x, self.mm_per_pixel_y],
            "pixel_center_convention": "align_corners_false",
        }


@dataclass
class ArcObservation:
    """Vector observation for one contiguous ring arc in normalized coordinates."""

    arc_id: str
    ring_id: int
    points: torch.Tensor
    tangents: torch.Tensor
    sigma: torch.Tensor
    prior_reliability: float = 0.95
    embedding: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.arc_id, str) or not self.arc_id:
            raise ValueError("arc_id must be a non-empty string")
        if not isinstance(self.ring_id, int):
            raise ValueError("ring_id must be an integer instance label")
        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError("points must have shape [N,2]")
        if self.tangents.shape != self.points.shape:
            raise ValueError("tangents must match points")
        if self.sigma.ndim not in (0, 1):
            raise ValueError("sigma must be scalar or [N]")
        if self.sigma.ndim == 1 and self.sigma.shape[0] != self.points.shape[0]:
            raise ValueError("per-point sigma must have length N")
        if self.points.shape[0] < 3:
            raise ValueError("an arc requires at least three samples")
        if not bool(torch.isfinite(self.points).all()) or not bool(
            torch.isfinite(self.tangents).all()
        ):
            raise ValueError("arc points and tangents must be finite")
        if not bool(
            (torch.linalg.vector_norm(self.points[1:] - self.points[:-1], dim=-1) > 0).all()
        ):
            raise ValueError("arc points must have positive consecutive spacing")
        if not bool((torch.linalg.vector_norm(self.tangents, dim=-1) > 0).all()):
            raise ValueError("arc tangents must be nonzero")
        if not bool(torch.isfinite(self.sigma).all()) or not bool((self.sigma > 0).all()):
            raise ValueError("arc sigma must be finite and strictly positive")
        if not (0.0 < self.prior_reliability < 1.0):
            raise ValueError("prior_reliability must lie strictly in (0,1)")
