from __future__ import annotations

import math

import torch

from oapith.types import CoordinateFrame


def normalized_sampling_grid(
    frame: CoordinateFrame,
    output_size: int,
    *,
    forward_matrix: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a source grid for an optionally transformed normalized square.

    ``forward_matrix`` maps the original normalized image chart to the requested
    output chart.  Its inverse is composed with rectangular-source normalization
    before sampling, so an augmented/native-scale crop is read from the original
    image in one interpolation step.  This is materially different from first
    shrinking the whole image to ``output_size`` and then magnifying that tensor:
    the latter irreversibly removes the high-frequency rings that native tiling is
    intended to preserve.
    """
    if output_size < 2:
        raise ValueError("output_size must be >= 2")
    centers = (torch.arange(output_size, device=device, dtype=dtype) + 0.5) * (
        2.0 / output_size
    ) - 1.0
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    if forward_matrix is not None:
        matrix = torch.as_tensor(forward_matrix, device=device, dtype=dtype)
        if matrix.shape != (3, 3):
            raise ValueError("forward_matrix must have shape [3,3]")
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("forward_matrix must be finite")
        inverse = torch.linalg.inv(matrix)
        output_homogeneous = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        source_homogeneous = output_homogeneous @ inverse.T
        denominator = source_homogeneous[..., 2]
        epsilon = torch.finfo(dtype).eps
        safe_denominator = torch.where(
            denominator.abs() >= epsilon,
            denominator,
            torch.where(
                denominator < 0,
                denominator.new_full(denominator.shape, -epsilon),
                denominator.new_full(denominator.shape, epsilon),
            ),
        )
        xx = source_homogeneous[..., 0] / safe_denominator
        yy = source_homogeneous[..., 1] / safe_denominator
    longest = float(max(frame.width, frame.height))
    # Convert square normalized coordinates to the rectangular source grid.
    gx = xx * (longest / frame.width)
    gy = yy * (longest / frame.height)
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    valid = ((gx.abs() <= 1.0) & (gy.abs() <= 1.0)).to(dtype).unsqueeze(0).unsqueeze(0)
    return grid, valid


def apply_homography(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply [...,3,3] homogeneous maps to [...,N,2] points without clipping."""
    if points.shape[-1] != 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("expected points [...,N,2] and matrix [...,3,3]")
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat([points, ones], dim=-1)
    mapped = torch.matmul(homogeneous, matrix.transpose(-1, -2))
    denominator = mapped[..., 2:]
    epsilon = torch.finfo(mapped.dtype).eps
    signed_epsilon = torch.where(
        denominator < 0,
        denominator.new_full(denominator.shape, -epsilon),
        denominator.new_full(denominator.shape, epsilon),
    )
    denominator = torch.where(denominator.abs() < epsilon, signed_epsilon, denominator)
    return mapped[..., :2] / denominator


def transform_tangents(tangents: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    linear = matrix[..., :2, :2]
    mapped = torch.matmul(tangents, linear.transpose(-1, -2))
    return torch.nn.functional.normalize(mapped, dim=-1, eps=1e-8)


def transform_covariance(covariance: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    linear = matrix[..., :2, :2]
    return linear @ covariance @ linear.transpose(-1, -2)


def similarity_matrix(
    angle_radians: float,
    scale: float = 1.0,
    translation: tuple[float, float] = (0.0, 0.0),
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("scale must be positive")
    c, s = math.cos(angle_radians), math.sin(angle_radians)
    return torch.tensor(
        [
            [scale * c, -scale * s, translation[0]],
            [scale * s, scale * c, translation[1]],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def cartesian_to_far(
    relative_xy: torch.Tensor, switch_radius: float = 2.0, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact far chart: unit direction and eta=switch_radius/distance.

    eta approaches zero at observational infinity. This chart is intended for
    distances >= switch_radius; near-zero points belong to the Cartesian branch.
    """
    if switch_radius <= 0:
        raise ValueError("switch_radius must be positive")
    distance = torch.linalg.vector_norm(relative_xy, dim=-1, keepdim=True)
    direction = relative_xy / distance.clamp_min(eps)
    eta = switch_radius / distance.clamp_min(eps)
    return direction, eta.squeeze(-1)


def far_to_cartesian(
    direction: torch.Tensor,
    eta: torch.Tensor,
    switch_radius: float = 2.0,
    eta_floor: float = 1e-6,
) -> torch.Tensor:
    if direction.shape[-1] != 2:
        raise ValueError("direction must end in 2")
    if switch_radius <= 0 or eta_floor <= 0:
        raise ValueError("radii must be positive")
    unit = torch.nn.functional.normalize(direction, dim=-1, eps=1e-8)
    return unit * (switch_radius / eta.clamp_min(eta_floor)).unsqueeze(-1)
