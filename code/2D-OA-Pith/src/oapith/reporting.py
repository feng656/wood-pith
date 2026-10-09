from __future__ import annotations

import colorsys
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from oapith.calibration import MixturePrediction, joint_state_decision
from oapith.explain.contribution import ArcContribution
from oapith.inference.posterior import ModePosterior
from oapith.inference.bootstrap import BootstrapDraw
from oapith.types import ArcObservation, CoordinateFrame


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def serialize_posteriors(
    posteriors: list[ModePosterior], frame: CoordinateFrame
) -> dict[str, object]:
    modes = []
    for posterior in posteriors:
        result = posterior.result
        center_px = (
            frame.normalized_to_pixel(result.center).detach().cpu().tolist()
            if result.center is not None
            else None
        )
        center_mm = None
        if center_px is not None and frame.mm_per_pixel_x is not None:
            center_mm = [
                center_px[0] * frame.mm_per_pixel_x,
                center_px[1] * frame.mm_per_pixel_y,
            ]
        modes.append(
            {
                "mode": result.mode.value,
                "weight": posterior.weight,
                "center_normalized": None
                if result.center is None
                else result.center.detach().cpu().tolist(),
                "center_pixel_unclipped": center_px,
                "center_mm_from_image_origin": center_mm,
                "direction": None
                if result.direction is None
                else result.direction.detach().cpu().tolist(),
                "inverse_distance": result.inverse_distance,
                "reliability": result.reliabilities,
                "evidence_score": result.evidence_score,
                "data_information": {
                    "rank": posterior.data.rank,
                    "dimension": posterior.data.dimension,
                    "condition_number": posterior.data.condition_number,
                    "minimum_eigenvalue": posterior.data.minimum_eigenvalue,
                    "pseudo_logdet": posterior.data.pseudo_logdet,
                    "valid": posterior.data.valid,
                    "invalid_reason": posterior.data.invalid_reason,
                    "covariance": None
                    if posterior.data.covariance is None
                    else posterior.data.covariance.cpu().tolist(),
                },
                "posterior_information": {
                    "rank": posterior.posterior.rank,
                    "dimension": posterior.posterior.dimension,
                    "condition_number": posterior.posterior.condition_number,
                    "valid": posterior.posterior.valid,
                    "invalid_reason": posterior.posterior.invalid_reason,
                    "covariance": None
                    if posterior.posterior.covariance is None
                    else posterior.posterior.covariance.cpu().tolist(),
                },
                "diagnostics": result.diagnostics,
            }
        )
    return {"coordinate_frame": frame.as_dict(), "modes": modes}


def save_prediction_json(
    path: str | Path,
    posteriors: list[ModePosterior],
    frame: CoordinateFrame,
    contributions: list[ArcContribution] | None = None,
    conformal: dict[str, object] | None = None,
    bootstrap: list[BootstrapDraw] | None = None,
    neural_state_probabilities: list[float] | None = None,
    joint_state_probabilities: list[float] | None = None,
    prediction: MixturePrediction | None = None,
) -> None:
    value = serialize_posteriors(posteriors, frame)
    if neural_state_probabilities is not None:
        value["neural_state_probabilities"] = neural_state_probabilities
    if joint_state_probabilities is not None:
        value["joint_state_probabilities"] = joint_state_probabilities
        state_index = int(joint_state_decision(joint_state_probabilities))
        value["state_decision"] = ["near", "far", "infinity", "null"][state_index]
        value["status"] = "rejected" if state_index == 3 else "ok"
    else:
        state_index = -1
        value["status"] = "ok"
    if state_index == 3:
        value["output_type"] = "null"
    else:
        active_kinds = {
            component.kind
            for component in (prediction.components if prediction is not None else [])
            if component.weight > 1e-12
        }
        if active_kinds and active_kinds <= {"infinity"}:
            value["output_type"] = "direction_only"
        elif active_kinds & {"infinity", "near_unbounded", "far_unbounded"}:
            # Topology belongs to the full mixture, not merely its MAP component.
            value["output_type"] = "unbounded_mixture"
        elif active_kinds:
            value["output_type"] = "finite_multimodal"
        else:
            value["output_type"] = "unresolved"
    value["arc_contributions"] = (
        [asdict(contribution) for contribution in contributions] if contributions is not None else None
    )
    value["conformal"] = conformal
    value["annotation_bootstrap"] = (
        [asdict(draw) for draw in bootstrap] if bootstrap is not None else None
    )
    Path(path).write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _color(value: float, minimum: float, maximum: float, reliability: float) -> tuple[int, int, int]:
    normalized = 0.5 if maximum <= minimum else (value - minimum) / (maximum - minimum)
    hue = (2.0 / 3.0) * (1.0 - min(max(normalized, 0.0), 1.0))
    saturation = 0.85
    brightness = 0.35 + 0.65 * min(max(reliability, 0.0), 1.0)
    rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return tuple(int(round(channel * 255)) for channel in rgb)


def _border_intersection(
    origin: tuple[float, float], direction: torch.Tensor, width: int, height: int
) -> tuple[float, float]:
    dx, dy = float(direction[0]), float(direction[1])
    values = []
    if abs(dx) > 1e-9:
        values.extend([(0 - origin[0]) / dx, ((width - 1) - origin[0]) / dx])
    if abs(dy) > 1e-9:
        values.extend([(0 - origin[1]) / dy, ((height - 1) - origin[1]) / dy])
    points = []
    for scale in values:
        if scale <= 0:
            continue
        x, y = origin[0] + scale * dx, origin[1] + scale * dy
        if -1e-6 <= x <= width - 1 + 1e-6 and -1e-6 <= y <= height - 1 + 1e-6:
            points.append((scale, x, y))
    if not points:
        return origin
    _, x, y = min(points)
    return x, y


def save_overlay(
    image_path: str | Path,
    output_path: str | Path,
    arcs: list[ArcObservation],
    frame: CoordinateFrame,
    best: ModePosterior,
    contributions: list[ArcContribution] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    contribution_map = {value.arc_id: value for value in contributions or []}
    finite_information = [
        value.information_gain
        for value in contribution_map.values()
        if value.information_gain is not None and math.isfinite(value.information_gain)
    ]
    minimum = min(finite_information) if finite_information else 0.0
    maximum = max(finite_information) if finite_information else 1.0
    for arc in arcs:
        contribution = contribution_map.get(arc.arc_id)
        reliability = (
            contribution.reliability if contribution is not None else arc.prior_reliability
        )
        if contribution is not None and contribution.structural_required:
            color = (255, 0, 255)
        else:
            information = contribution.information_gain if contribution is not None else 0.0
            color = _color(float(information or 0.0), minimum, maximum, reliability)
        points_px = frame.normalized_to_pixel(arc.points).detach().cpu().tolist()
        if len(points_px) >= 2:
            draw.line([tuple(value) for value in points_px], fill=color, width=3)

    result = best.result
    if result.center is not None:
        center_px_tensor = frame.normalized_to_pixel(result.center)
        center_px = (float(center_px_tensor[0]), float(center_px_tensor[1]))
        inside = 0 <= center_px[0] < frame.width and 0 <= center_px[1] < frame.height
        if inside:
            radius = 7
            draw.ellipse(
                [center_px[0] - radius, center_px[1] - radius, center_px[0] + radius, center_px[1] + radius],
                outline=(255, 255, 0),
                width=3,
            )
            covariance = (
                best.posterior.covariance
                if best.data.valid and best.data.rank == best.data.dimension
                else None
            )
            if covariance is not None and covariance.shape == (2, 2):
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                angles = torch.linspace(0, 2 * math.pi, 97, dtype=covariance.dtype)
                circle = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
                ellipse = result.center + 2.4477 * (circle @ (
                    eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt())
                ).T)
                ellipse_px = frame.normalized_to_pixel(ellipse).cpu().tolist()
                draw.line([tuple(value) for value in ellipse_px], fill=(255, 255, 0), width=2)
        else:
            origin = frame.center_px
            direction_px = torch.tensor([center_px[0] - origin[0], center_px[1] - origin[1]])
            endpoint = _border_intersection(origin, direction_px, frame.width, frame.height)
            draw.line([origin, endpoint], fill=(255, 255, 0), width=4)
            draw.ellipse(
                [endpoint[0] - 5, endpoint[1] - 5, endpoint[0] + 5, endpoint[1] + 5],
                fill=(255, 255, 0),
            )
    elif result.direction is not None:
        origin = frame.center_px
        direction_px = result.direction.detach().cpu()
        endpoint = _border_intersection(origin, direction_px, frame.width, frame.height)
        draw.line([origin, endpoint], fill=(255, 255, 0), width=4)
        draw.text((max(0, endpoint[0] - 70), max(0, endpoint[1] - 18)), "distance unresolved", fill=(255, 255, 0))
    image.save(output_path)
