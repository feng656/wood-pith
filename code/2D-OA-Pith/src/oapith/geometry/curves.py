from __future__ import annotations

import torch


def cumulative_arclength(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.shape[-1] != 2 or points.shape[0] < 2:
        raise ValueError("points must have shape [N,2], N>=2")
    lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1)
    return torch.cat([points.new_zeros(1), torch.cumsum(lengths, dim=0)])


def resample_polyline(points: torch.Tensor, spacing: float) -> torch.Tensor:
    """Linearly resample a polyline at approximately independent physical spacing."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    s = cumulative_arclength(points)
    total = float(s[-1])
    if total <= spacing:
        # Geometry fitting requires a tangent-bearing arc, not a two-point chord.
        # Keep a midpoint for a short but otherwise valid annotated polyline.
        return points[[0, points.shape[0] // 2, -1]].clone()
    count = max(2, int(total / spacing) + 1)
    targets = torch.linspace(0.0, total, count, device=points.device, dtype=points.dtype)
    right = torch.searchsorted(s, targets, right=True).clamp(1, points.shape[0] - 1)
    left = right - 1
    denom = (s[right] - s[left]).clamp_min(1e-8)
    alpha = ((targets - s[left]) / denom).unsqueeze(-1)
    return points[left] * (1.0 - alpha) + points[right] * alpha


def estimate_tangents(points: torch.Tensor, window: int = 2) -> torch.Tensor:
    if window < 1:
        raise ValueError("window must be >= 1")
    n = points.shape[0]
    indices = torch.arange(n, device=points.device)
    left = (indices - window).clamp_min(0)
    right = (indices + window).clamp_max(n - 1)
    tangent = points[right] - points[left]
    return torch.nn.functional.normalize(tangent, dim=-1, eps=1e-8)


def doubled_angle(tangents: torch.Tensor) -> torch.Tensor:
    """Encode an unoriented tangent as (cos 2theta, sin 2theta)."""
    t = torch.nn.functional.normalize(tangents, dim=-1, eps=1e-8)
    return torch.stack([t[..., 0].square() - t[..., 1].square(), 2 * t[..., 0] * t[..., 1]], -1)


def doubled_angle_rotate(field: torch.Tensor, angle_radians: torch.Tensor) -> torch.Tensor:
    c = torch.cos(2.0 * angle_radians)
    s = torch.sin(2.0 * angle_radians)
    x, y = field[..., 0], field[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)
