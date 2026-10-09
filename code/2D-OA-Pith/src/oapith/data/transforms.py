from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as functional

from oapith.geometry.coordinates import similarity_matrix


def warp_normalized_square(
    tensor: torch.Tensor,
    forward_matrix: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp [C,H,W] or [B,C,H,W] using old->new normalized coordinates."""
    squeeze = tensor.ndim == 3
    if squeeze:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError("tensor must be square [C,H,W] or [B,C,H,W]")
    matrix = forward_matrix.to(device=tensor.device, dtype=tensor.dtype)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0).expand(tensor.shape[0], -1, -1)
    inverse = torch.linalg.inv(matrix)
    grid = functional.affine_grid(inverse[:, :2, :], tensor.shape, align_corners=False)
    warped = functional.grid_sample(
        tensor,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    return warped.squeeze(0) if squeeze else warped


@dataclass
class RandomSimilarity:
    max_rotation_radians: float = 3.141592653589793
    min_scale: float = 0.85
    max_scale: float = 1.15
    max_translation: float = 0.10

    def __post_init__(self) -> None:
        if self.min_scale <= 0 or self.max_scale < self.min_scale:
            raise ValueError("invalid scale interval")
        if self.max_translation < 0 or self.max_rotation_radians < 0:
            raise ValueError("augmentation bounds must be nonnegative")

    def sample(
        self,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        def uniform(low: float, high: float) -> float:
            value = torch.rand((), generator=generator).item()
            return low + (high - low) * value

        angle = uniform(-self.max_rotation_radians, self.max_rotation_radians)
        scale = uniform(self.min_scale, self.max_scale)
        tx = uniform(-self.max_translation, self.max_translation)
        ty = uniform(-self.max_translation, self.max_translation)
        return similarity_matrix(angle, scale, (tx, ty), device=device, dtype=dtype)

    def sample_native_view(
        self,
        native_scale: float,
        *,
        jitter: tuple[float, float] = (0.8, 1.25),
        content_half_extent: tuple[float, float] = (1.0, 1.0),
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample an axis-scale-matched crop anywhere in the original field.

        ``native_scale=max(image_side)/network_side`` makes one output pixel
        correspond approximately to one source pixel.  Sampling the crop centre
        across the full valid field is what makes training match native-resolution
        tiled inference; scale jitter alone would keep presenting the centre crop.
        """
        if (
            native_scale <= 0
            or jitter[0] <= 0
            or jitter[1] < jitter[0]
            or min(content_half_extent) <= 0
            or max(content_half_extent) > 1.0
        ):
            raise ValueError("invalid native-view scale or jitter")

        def uniform(low: float, high: float) -> float:
            value = torch.rand((), generator=generator).item()
            return low + (high - low) * value

        angle = uniform(-self.max_rotation_radians, self.max_rotation_radians)
        scale = native_scale * uniform(jitter[0], jitter[1])
        # The inverse-rotated square has an axis-aligned half-extent
        # (|cos a|+|sin a|)/s in *each* original-chart coordinate.  Using only 1/s
        # lets rotated crop corners leave the content rectangle, especially for a
        # non-square source image.
        crop_half_extent = (abs(math.cos(angle)) + abs(math.sin(angle))) / max(
            scale, 1e-8
        )
        center_extent = (
            max(0.0, content_half_extent[0] - crop_half_extent),
            max(0.0, content_half_extent[1] - crop_half_extent),
        )
        center = torch.tensor(
            [
                uniform(-center_extent[0], center_extent[0]),
                uniform(-center_extent[1], center_extent[1]),
            ],
            device=device,
            dtype=dtype,
        )
        c = math.cos(angle)
        s = math.sin(angle)
        rotation = center.new_tensor([[c, -s], [s, c]])
        translation = -(scale * rotation) @ center
        return similarity_matrix(
            angle,
            scale,
            (float(translation[0]), float(translation[1])),
            device=device,
            dtype=dtype,
        )
