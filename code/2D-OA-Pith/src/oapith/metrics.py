from __future__ import annotations

import math

import torch

from oapith.types import CoordinateFrame


def localization_errors(
    prediction: torch.Tensor, target: torch.Tensor, frame: CoordinateFrame | None = None
) -> dict[str, torch.Tensor]:
    delta = prediction - target
    total = torch.linalg.vector_norm(delta, dim=-1)
    radial_direction = torch.nn.functional.normalize(target, dim=-1, eps=1e-8)
    tangential_direction = torch.stack([-radial_direction[..., 1], radial_direction[..., 0]], -1)
    radial = (delta * radial_direction).sum(-1).abs()
    tangential = (delta * tangential_direction).sum(-1).abs()
    target_direction = torch.atan2(target[..., 1], target[..., 0])
    predicted_direction = torch.atan2(prediction[..., 1], prediction[..., 0])
    angular = torch.atan2(
        torch.sin(predicted_direction - target_direction),
        torch.cos(predicted_direction - target_direction),
    ).abs()
    log_distance = (
        torch.log(torch.linalg.vector_norm(prediction, dim=-1).clamp_min(1e-8))
        - torch.log(torch.linalg.vector_norm(target, dim=-1).clamp_min(1e-8))
    ).abs()
    result = {
        "normalized_error": total,
        "radial_error": radial,
        "tangential_error": tangential,
        "direction_error_radians": angular,
        "log_distance_error": log_distance,
    }
    if frame is not None:
        result["pixel_error"] = total * frame.scale_px
        if frame.mm_per_pixel_x is not None:
            result["millimeter_error"] = torch.linalg.vector_norm(
                frame.normalized_delta_to_mm(delta), dim=-1
            )
    return result


def symmetric_curve_distance(first: torch.Tensor, second: torch.Tensor) -> dict[str, torch.Tensor]:
    distance = torch.cdist(first, second)
    first_to_second = distance.min(-1).values
    second_to_first = distance.min(-2).values
    all_distances = torch.cat([first_to_second, second_to_first])
    return {
        "symmetric_chamfer": 0.5 * (first_to_second.mean() + second_to_first.mean()),
        "hausdorff_95": torch.quantile(all_distances, 0.95),
        "hausdorff": all_distances.max(),
    }


def tolerance_curve_fscore(
    prediction: torch.Tensor, target: torch.Tensor, tolerance: float
) -> dict[str, torch.Tensor]:
    """Thin-line precision/recall without arbitrary mask thickening."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    distance = torch.cdist(prediction, target)
    precision = (distance.min(-1).values <= tolerance).to(distance.dtype).mean()
    recall = (distance.min(-2).values <= tolerance).to(distance.dtype).mean()
    fscore = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "tolerance_precision": precision,
        "tolerance_recall": recall,
        "tolerance_fscore": fscore,
    }


def risk_coverage(error: torch.Tensor, uncertainty: torch.Tensor) -> dict[str, torch.Tensor]:
    order = torch.argsort(uncertainty)
    sorted_error = error[order]
    cumulative_risk = torch.cumsum(sorted_error, 0) / torch.arange(
        1, error.numel() + 1, device=error.device, dtype=error.dtype
    )
    coverage = torch.arange(1, error.numel() + 1, device=error.device, dtype=error.dtype) / error.numel()
    aurc = torch.trapezoid(cumulative_risk, coverage)
    return {"coverage": coverage, "risk": cumulative_risk, "aurc": aurc}


def empirical_coverage(contains: torch.Tensor) -> tuple[float, float]:
    count = int(contains.numel())
    coverage = float(contains.float().mean()) if count else math.nan
    standard_error = math.sqrt(coverage * (1.0 - coverage) / count) if count else math.nan
    return coverage, standard_error
