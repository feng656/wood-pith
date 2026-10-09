from __future__ import annotations

from collections import defaultdict

import torch

from oapith.data.manifest import RingCurve
from oapith.geometry.curves import cumulative_arclength, estimate_tangents


def _resample(values: torch.Tensor, count: int, *, closed: bool = False) -> torch.Tensor:
    if count < 3:
        raise ValueError("count must be at least three")
    if closed and not torch.allclose(values[0], values[-1]):
        values = torch.cat([values, values[:1]], dim=0)
    positions = cumulative_arclength(values)
    if float(positions[-1]) <= 0:
        raise ValueError("a curve must have positive arclength")
    positions = positions / positions[-1]
    targets = (
        torch.arange(count, device=values.device, dtype=values.dtype) / count
        if closed
        else torch.linspace(0, 1, count, device=values.device, dtype=values.dtype)
    )
    right = torch.searchsorted(positions, targets, right=True).clamp(1, values.shape[0] - 1)
    left = right - 1
    alpha = ((targets - positions[left]) / (positions[right] - positions[left]).clamp_min(1e-8))
    return values[left] * (1 - alpha[:, None]) + values[right] * alpha[:, None]


def _resample_scalar(
    values: torch.Tensor, points: torch.Tensor, count: int, *, closed: bool = False
) -> torch.Tensor:
    if closed and not torch.allclose(points[0], points[-1]):
        points = torch.cat([points, points[:1]], dim=0)
        values = torch.cat([values, values[:1]], dim=0)
    positions = cumulative_arclength(points)
    positions = positions / positions[-1].clamp_min(1e-8)
    targets = (
        torch.arange(count, device=points.device, dtype=points.dtype) / count
        if closed
        else torch.linspace(0, 1, count, device=points.device, dtype=points.dtype)
    )
    right = torch.searchsorted(positions, targets, right=True).clamp(1, points.shape[0] - 1)
    left = right - 1
    alpha = (targets - positions[left]) / (positions[right] - positions[left]).clamp_min(1e-8)
    return values[left] * (1 - alpha) + values[right] * alpha


def consensus_curve(
    curves: list[RingCurve],
    *,
    samples: int = 256,
    minimum_sigma_px: float = 0.25,
) -> RingCurve:
    """Robust registered centerline and local normal uncertainty from 2+ raters.

    Curves must describe the same topological arc.  Topology disagreements are not
    averaged here; they remain separate hypotheses in the manifest.
    """
    if len(curves) < 2:
        raise ValueError("consensus requires at least two independent curves")
    if minimum_sigma_px <= 0:
        raise ValueError("minimum_sigma_px must be positive")
    identity = {(curve.ring_id, curve.arc_id) for curve in curves}
    if len(identity) != 1:
        raise ValueError("all curves must share ring_id and arc_id")
    annotators = [curve.annotator_id for curve in curves]
    if any(value is None for value in annotators) or len(set(annotators)) != len(annotators):
        raise ValueError("consensus curves need distinct annotator_id values")
    closed_values = [bool(curve.metadata.get("closed", False)) for curve in curves]
    if any(value != closed_values[0] for value in closed_values):
        raise ValueError("open and closed topology hypotheses may not be averaged")
    closed = closed_values[0]

    point_sets: list[torch.Tensor] = []
    sigma_sets: list[torch.Tensor] = []
    reference: torch.Tensor | None = None
    for curve in curves:
        curve.validate()
        points = torch.tensor(curve.points_px, dtype=torch.float64)
        sampled = _resample(points, samples, closed=closed)
        sigma = torch.as_tensor(curve.sigma_px, dtype=torch.float64)
        if sigma.ndim == 0:
            sampled_sigma = sigma.expand(samples)
        else:
            sampled_sigma = _resample_scalar(sigma, points, samples, closed=closed)
        if reference is None:
            reference = sampled
        elif closed:
            best_value = None
            best_points = sampled
            best_sigma = sampled_sigma
            for candidate_points, candidate_sigma in (
                (sampled, sampled_sigma),
                (sampled.flip(0), sampled_sigma.flip(0)),
            ):
                for shift in range(samples):
                    aligned = candidate_points.roll(shift, dims=0)
                    value = (aligned - reference).square().mean()
                    if best_value is None or value < best_value:
                        best_value = value
                        best_points = aligned
                        best_sigma = candidate_sigma.roll(shift, dims=0)
            sampled, sampled_sigma = best_points, best_sigma
        else:
            forward = (sampled - reference).square().mean()
            reverse = (sampled.flip(0) - reference).square().mean()
            if reverse < forward:
                sampled = sampled.flip(0)
                sampled_sigma = sampled_sigma.flip(0)
        point_sets.append(sampled)
        sigma_sets.append(sampled_sigma)

    stack = torch.stack(point_sets)
    centerline = torch.quantile(stack, 0.5, dim=0, interpolation="midpoint")
    tangents = estimate_tangents(centerline)
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], -1)
    normal_offsets = ((stack - centerline[None]) * normals[None]).sum(-1)
    median_offset = torch.quantile(
        normal_offsets, 0.5, dim=0, interpolation="midpoint"
    )
    centerline = centerline + median_offset[:, None] * normals
    absolute_deviation = torch.quantile(
        (normal_offsets - median_offset[None]).abs(),
        0.5,
        dim=0,
        interpolation="midpoint",
    )
    robust_between_rater_sigma = 1.4826 * absolute_deviation
    declared_variance = torch.quantile(
        torch.stack(sigma_sets).square(), 0.5, dim=0, interpolation="midpoint"
    )
    sigma = torch.sqrt(
        robust_between_rater_sigma.square()
        + declared_variance
        + minimum_sigma_px**2
    )
    if closed:
        centerline = torch.cat([centerline, centerline[:1]], dim=0)
        sigma = torch.cat([sigma, sigma[:1]], dim=0)
    visibility = float(
        torch.quantile(
            torch.tensor([curve.visibility for curve in curves]),
            0.5,
            interpolation="midpoint",
        )
    )
    ring_id, arc_id = next(iter(identity))
    return RingCurve(
        ring_id=ring_id,
        points_px=centerline.tolist(),
        sigma_px=sigma.tolist(),
        visibility=visibility,
        annotator_id="consensus",
        arc_id=arc_id,
        metadata={
            "consensus_annotators": annotators,
            "consensus_method": "arclength-registration + normal-coordinate median/MAD",
            "number_annotators": len(curves),
            "closed": closed,
        },
    )


def collapse_multi_annotator_curves(curves: list[RingCurve]) -> list[RingCurve]:
    """Collapse only explicitly matched arc IDs; leave topology hypotheses separate."""
    groups: dict[tuple[int, str], list[RingCurve]] = defaultdict(list)
    passthrough: list[RingCurve] = []
    for curve in curves:
        if curve.arc_id is None or curve.annotator_id is None:
            passthrough.append(curve)
        else:
            groups[(curve.ring_id, curve.arc_id)].append(curve)
    output = list(passthrough)
    for values in groups.values():
        if len(values) == 1:
            output.extend(values)
        else:
            output.append(consensus_curve(values))
    return output
