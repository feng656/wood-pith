from __future__ import annotations

import itertools
import math

import torch

from oapith.types import ArcObservation


def _flatten(arcs: list[ArcObservation]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points, tangents, weights = [], [], []
    for arc in arcs:
        arc.validate()
        sigma = arc.sigma.expand(arc.points.shape[0]) if arc.sigma.ndim == 0 else arc.sigma
        points.append(arc.points)
        tangents.append(torch.nn.functional.normalize(arc.tangents, dim=-1, eps=1e-8))
        weights.append(arc.prior_reliability / sigma.square().clamp_min(1e-8))
    return torch.cat(points), torch.cat(tangents), torch.cat(weights)


def weighted_normal_intersection(arcs: list[ArcObservation]) -> tuple[torch.Tensor, float]:
    """Least-squares intersection of lines through points along ring normals."""
    points, tangents, weights = _flatten(arcs)
    weights = weights / weights.mean().clamp_min(1e-8)
    outer = tangents[..., :, None] * tangents[..., None, :]
    system = (weights[..., None, None] * outer).sum(0)
    rhs = (weights[..., None] * torch.matmul(outer, points[..., None]).squeeze(-1)).sum(0)
    eigenvalues = torch.linalg.eigvalsh(system)
    condition = float(eigenvalues[-1] / eigenvalues[0].clamp_min(1e-12))
    center = torch.linalg.pinv(system, rtol=1e-10) @ rhs
    return center, condition


def _pair_intersection(
    point_a: torch.Tensor,
    tangent_a: torch.Tensor,
    point_b: torch.Tensor,
    tangent_b: torch.Tensor,
) -> torch.Tensor | None:
    normal_a = torch.stack([-tangent_a[1], tangent_a[0]])
    normal_b = torch.stack([-tangent_b[1], tangent_b[0]])
    matrix = torch.stack([normal_a, -normal_b], dim=1)
    determinant = torch.det(matrix)
    if determinant.abs() < 1e-4:
        return None
    distance = torch.linalg.solve(matrix, point_b - point_a)[0]
    return point_a + distance * normal_a


def _deduplicate(candidates: list[torch.Tensor], tolerance: float = 0.05) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for candidate in candidates:
        if torch.isfinite(candidate).all() and not any(
            torch.linalg.vector_norm(candidate - previous) < tolerance for previous in result
        ):
            result.append(candidate)
    return result


def generate_candidates(
    arcs: list[ArcObservation],
    *,
    neural_candidates: torch.Tensor | None = None,
    max_pair_candidates: int = 24,
    far_distances: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
) -> list[torch.Tensor]:
    """APD-style normals, pairwise intersections, neural modes, and log-distance starts."""
    points, tangents, _ = _flatten(arcs)
    candidates: list[torch.Tensor] = []
    if neural_candidates is not None:
        candidates.extend(neural_candidates.detach().to(points).unbind(0))
    least_squares, _ = weighted_normal_intersection(arcs)
    candidates.append(least_squares)

    # Deterministically subsample pairs across the observation rather than relying on
    # a fragile single RANSAC result.
    count = points.shape[0]
    stride = max(1, count // max(8, int(math.sqrt(max_pair_candidates * 2))))
    indices = list(range(0, count, stride))
    pairs = list(itertools.combinations(indices, 2))
    if len(pairs) > max_pair_candidates:
        step = len(pairs) / max_pair_candidates
        pairs = [pairs[min(int(i * step), len(pairs) - 1)] for i in range(max_pair_candidates)]
    for first, second in pairs:
        candidate = _pair_intersection(
            points[first], tangents[first], points[second], tangents[second]
        )
        if candidate is not None:
            candidates.append(candidate)

    # Near-parallel lines constrain an axial direction. Keep both signs because an
    # unoriented tangent cannot resolve the pith side by itself.
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)
    scatter = normals.T @ normals
    _, eigenvectors = torch.linalg.eigh(scatter)
    direction = eigenvectors[:, -1]
    for sign in (-1.0, 1.0):
        for distance in far_distances:
            candidates.append(direction * (sign * distance))
    candidates.append(points.mean(0))
    return _deduplicate(candidates)

