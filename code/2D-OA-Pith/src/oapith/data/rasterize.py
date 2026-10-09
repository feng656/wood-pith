from __future__ import annotations

from dataclasses import dataclass

import torch

from oapith.geometry.curves import doubled_angle


@dataclass
class NormalizedCurve:
    ring_id: int
    points: torch.Tensor
    sigma: torch.Tensor
    visibility: float = 1.0
    arc_id: str | None = None


def _segments(curves: list[NormalizedCurve]) -> tuple[torch.Tensor, ...]:
    starts, ends, sigma, ring_ids, visibility = [], [], [], [], []
    for curve in curves:
        if curve.points.shape[0] < 2:
            continue
        starts.append(curve.points[:-1])
        ends.append(curve.points[1:])
        if curve.sigma.ndim == 0:
            segment_sigma = curve.sigma.expand(curve.points.shape[0] - 1)
        else:
            segment_sigma = 0.5 * (curve.sigma[:-1] + curve.sigma[1:])
        sigma.append(segment_sigma)
        ring_ids.append(
            torch.full(
                (curve.points.shape[0] - 1,),
                curve.ring_id,
                device=curve.points.device,
                dtype=torch.long,
            )
        )
        visibility.append(segment_sigma.new_full(segment_sigma.shape, curve.visibility))
    if not starts:
        raise ValueError("at least one non-empty curve is required")
    return (
        torch.cat(starts),
        torch.cat(ends),
        torch.cat(sigma).clamp_min(1e-5),
        torch.cat(ring_ids),
        torch.cat(visibility),
    )


def rasterize_curves(
    curves: list[NormalizedCurve],
    size: int,
    *,
    truncation: float = 0.04,
    label_band_sigma: float = 3.0,
    point_chunk: int = 16384,
    segment_chunk: int = 512,
) -> dict[str, torch.Tensor]:
    """Rasterize uncertain sub-pixel centerlines to soft targets.

    ``sigma`` is annotation uncertainty/tolerance in normalized coordinates; it is
    not converted into an arbitrary positive-class thickness.
    """
    if (
        size < 2
        or truncation <= 0
        or label_band_sigma <= 0
        or point_chunk < 1
        or segment_chunk < 1
    ):
        raise ValueError("invalid rasterization parameters")
    a, b, segment_sigma, segment_ring, segment_visibility = _segments(curves)
    device, dtype = a.device, a.dtype
    centers = (torch.arange(size, device=device, dtype=dtype) + 0.5) * (2.0 / size) - 1.0
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    pixels = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    ab = b - a
    ab2 = ab.square().sum(-1).clamp_min(1e-12)
    tangent = torch.nn.functional.normalize(ab, dim=-1, eps=1e-8)

    best_d2: list[torch.Tensor] = []
    best_index: list[torch.Tensor] = []
    for start in range(0, pixels.shape[0], point_chunk):
        p = pixels[start : start + point_chunk]
        values = p.new_full((p.shape[0],), torch.inf)
        indices = torch.zeros(p.shape[0], device=device, dtype=torch.long)
        for segment_start in range(0, a.shape[0], segment_chunk):
            segment_stop = min(segment_start + segment_chunk, a.shape[0])
            chunk_a = a[segment_start:segment_stop]
            chunk_ab = ab[segment_start:segment_stop]
            relative = p[:, None, :] - chunk_a[None, :, :]
            alpha = (
                (relative * chunk_ab[None, :, :]).sum(-1)
                / ab2[segment_start:segment_stop][None, :]
            ).clamp(0.0, 1.0)
            projection = chunk_a[None, :, :] + alpha[..., None] * chunk_ab[None, :, :]
            d2_chunk = (p[:, None, :] - projection).square().sum(-1)
            local_values, local_indices = d2_chunk.min(dim=1)
            update = local_values < values
            values = torch.where(update, local_values, values)
            indices = torch.where(update, local_indices + segment_start, indices)
        best_d2.append(values)
        best_index.append(indices)
    d2 = torch.cat(best_d2)
    nearest = torch.cat(best_index)
    # A raster cell represents a box, not a point sample.  Add the exact variance
    # of a one-pixel-wide box kernel so sub-pixel annotations remain visible after
    # resampling instead of producing an almost-zero target everywhere.
    pixel_sigma = (2.0 / size) / (12.0**0.5)
    nearest_sigma = torch.sqrt(segment_sigma[nearest].square() + pixel_sigma**2)
    nearest_visibility = segment_visibility[nearest]
    probability = torch.exp(-0.5 * d2 / nearest_sigma.square()) * nearest_visibility
    distance = d2.sqrt().clamp_max(truncation) / truncation
    orientation = doubled_angle(tangent[nearest])
    within_band = (d2.sqrt() <= label_band_sigma * nearest_sigma) & (
        nearest_visibility > 0
    )
    ring_index = torch.where(
        within_band,
        segment_ring[nearest],
        torch.full_like(nearest, -1, dtype=torch.long),
    )
    # Ambiguous labels contribute softly and less strongly, but are not discarded.
    reference_sigma = 2.0 / size
    label_weight = nearest_visibility / (1.0 + (nearest_sigma / reference_sigma).square())
    shape = (size, size)
    return {
        "ring_probability": probability.reshape(1, *shape),
        "distance": distance.reshape(1, *shape),
        "orientation": orientation.reshape(*shape, 2).permute(2, 0, 1),
        "orientation_valid": within_band.reshape(1, *shape).to(dtype),
        "ring_index": ring_index.reshape(*shape),
        "annotation_sigma": nearest_sigma.reshape(1, *shape),
        "annotation_sigma_distance": (nearest_sigma / truncation).reshape(1, *shape),
        "label_weight": label_weight.reshape(1, *shape),
    }
