from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from ..provenance import read_json_object, read_jsonl


STATE_COLORS = {
    "POINT": "#0072B2",
    "RANGE": "#E69F00",
    "RAY": "#D55E00",
    "AXIS": "#CC79A7",
    "MULTIMODAL": "#7A3E9D",
    "REJECT": "#555555",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return read_json_object(path)


def _draw_outside_marker(
    axis: plt.Axes,
    point: np.ndarray,
    width: int,
    height: int,
    color: str,
    label: str,
) -> None:
    if 0 <= point[0] < width and 0 <= point[1] < height:
        axis.scatter(point[0], point[1], marker="x", s=80, linewidth=2.0, color=color, label=label)
        return
    center = np.asarray([width / 2.0, height / 2.0])
    vector = point - center
    scale = min(
        (0.46 * width) / max(abs(vector[0]), 1e-12),
        (0.46 * height) / max(abs(vector[1]), 1e-12),
    )
    end = center + max(min(scale, 1.0), 0.0) * vector
    axis.annotate(
        f"{label}\noutside: {np.linalg.norm(vector):.1f}px",
        xy=end,
        xytext=center,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
        color=color,
        fontsize=8,
    )


def render_sample_card(
    crop_image_path: str | Path,
    crop_annotation_path: str | Path,
    result_path: str | Path,
    output_path: str | Path,
    *,
    pith_gt_crop_px: list[float] | tuple[float, float] | None = None,
    uncertainty_path: str | Path | None = None,
) -> None:
    image_bgr = cv2.imread(str(crop_image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(crop_image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    annotation = _load_json(crop_annotation_path)
    result = _load_json(result_path)
    uncertainty = _load_json(uncertainty_path) if uncertainty_path is not None else None
    height, width = image.shape[:2]
    d_fov = float(result.get("diagnostics", {}).get("normalization_scale_px", np.hypot(width, height)))
    center_px = np.asarray([width / 2.0, height / 2.0])

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    overview = axes[0, 0]
    overview.imshow(image)
    for ring in annotation["rings"]:
        for arc in ring["arcs"]:
            points = np.asarray(arc["points_crop_px"], dtype=float)
            overview.plot(points[:, 0], points[:, 1], lw=0.8, alpha=0.75)
    # GT is deliberately supplied from the evaluation manifest.  The curve
    # annotation consumed by evidence construction is GT-free.
    if pith_gt_crop_px is not None:
        _draw_outside_marker(
            overview,
            np.asarray(pith_gt_crop_px, dtype=float),
            width,
            height,
            "#009E73",
            "GT",
        )
    raw = result.get("raw_center_norm")
    if raw is not None:
        raw_px = center_px + d_fov * np.asarray(raw, dtype=float)
        _draw_outside_marker(
            overview,
            raw_px,
            width,
            height,
            STATE_COLORS.get(result["state"], "#0072B2"),
            f"estimate ({result['state']})",
        )
    if uncertainty is not None:
        ellipse = uncertainty.get("ellipse", {})
        ellipse_center = ellipse.get("center_median_norm")
        if ellipse.get("valid") and ellipse_center is not None:
            location = center_px + d_fov * np.asarray(ellipse_center, dtype=float)
            overview.add_patch(
                Ellipse(
                    xy=location,
                    width=2.0 * d_fov * float(ellipse["major_semi_axis_norm"]),
                    height=2.0 * d_fov * float(ellipse["minor_semi_axis_norm"]),
                    angle=np.degrees(float(ellipse["major_axis_angle_rad"])),
                    fill=False,
                    edgecolor="#56B4E9",
                    linewidth=1.5,
                    linestyle="--",
                    label="structured-replay ellipse",
                )
            )
    overview.set_xlim(0, width)
    overview.set_ylim(height, 0)
    overview.set_title(f"{result['crop_id']} — {result['state']}")
    overview.set_xlabel("x / column [px]")
    overview.set_ylabel("y / row [px]")

    far_axis = axes[0, 1]
    far = result.get("far_scan", {})
    phi = np.asarray(far.get("angles_rad", []), dtype=float)
    loss = np.asarray(far.get("objective", far.get("loss", [])), dtype=float)
    if len(phi) and len(phi) == len(loss):
        far_axis.plot(np.degrees(phi), loss, color="#0072B2")
        minimum_angle = far.get("minimum_angle_rad")
        if minimum_angle is None and isinstance(far.get("best"), dict):
            minimum_angle = far["best"].get("phi_rad")
        if minimum_angle is not None:
            far_axis.axvline(np.degrees(float(minimum_angle)), color="#D55E00", ls="--")
        if result.get("objective") is not None:
            far_axis.axhline(float(result["objective"]), color="#009E73", ls=":", label="finite")
        far_axis.legend()
    else:
        far_axis.text(0.5, 0.5, "No far-scan trace", ha="center", va="center")
    far_axis.set_title("Analytic far-field scan")
    far_axis.set_xlabel("axis angle [deg, modulo 180°]")
    far_axis.set_ylabel("profiled objective")

    profile_axis = axes[1, 0]
    profiles = result.get("profiles", [])
    for index, profile in enumerate(profiles):
        z = np.asarray(profile.get("z", []), dtype=float)
        values = np.asarray(profile.get("objective", profile.get("loss", [])), dtype=float)
        if len(z) == len(values) and len(z):
            profile_axis.plot(z, values, lw=1.1, label=profile.get("label", f"profile {index}"))
    if profiles:
        profile_axis.legend(fontsize=7)
    else:
        profile_axis.text(0.5, 0.5, "Profile not triggered", ha="center", va="center")
    profile_axis.set_title("Conditional compact-distance profiles")
    profile_axis.set_xlabel("z = s / (1 + |s|)")
    profile_axis.set_ylabel("profiled objective")

    text_axis = axes[1, 1]
    text_axis.axis("off")
    lines = [
        f"state: {result['state']}",
        f"search adequate: {result.get('search_adequate')}",
        f"condition ratio: {result.get('condition_ratio')}",
        f"model risk: {result.get('model_risk')}",
        f"production usable: {result.get('production_usable')}",
        f"uncertainty state: {uncertainty.get('adjudicated_state') if uncertainty else 'not supplied'}",
        f"S3 re-audit: {uncertainty.get('requires_search_reaudit') if uncertainty else 'not supplied'}",
        "reason codes:",
        *(f"  • {value}" for value in result.get("reason_codes", [])),
        "finite modes:",
        *(f"  • {mode}" for mode in result.get("finite_modes", [])),
    ]
    text_axis.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _arc_segment(points: np.ndarray, interval: tuple[float, float] | list[float]) -> np.ndarray:
    if len(points) < 2:
        return points
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if length[-1] <= 0:
        return points[:1]
    fraction = length / length[-1]
    lo = float(np.clip(float(interval[0]), 0.0, 1.0))
    hi = float(np.clip(float(interval[1]), 0.0, 1.0))
    if hi < lo:
        raise ValueError("subarc interval must be ordered")
    interior = fraction[(fraction > lo) & (fraction < hi)]
    sample_fraction = np.unique(np.r_[lo, interior, hi])
    return np.column_stack(
        [
            np.interp(sample_fraction, fraction, points[:, 0]),
            np.interp(sample_fraction, fraction, points[:, 1]),
        ]
    )


def render_contribution_overlay(
    crop_image_path: str | Path,
    crop_annotation_path: str | Path,
    contribution_path: str | Path,
    output_path: str | Path,
    partition_id: str,
) -> None:
    image_bgr = cv2.imread(str(crop_image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(crop_image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    annotation = _load_json(crop_annotation_path)
    records = [row for row in read_jsonl(contribution_path) if row["partition_id"] == partition_id]
    by_arc: dict[str, np.ndarray] = {}
    for ring in annotation["rings"]:
        for arc in ring["arcs"]:
            by_arc[str(arc["arc_id"])] = np.asarray(arc["points_crop_px"], dtype=float)
    colors = {
        "BENEFICIAL_GT": "#009E73",
        "HARMFUL_GT": "#D55E00",
        "NEUTRAL": "#888888",
        "UNCERTAIN": "#E69F00",
    }
    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    axis.imshow(image)
    for record in records:
        arc_id = record.get("arc_id")
        if arc_id is None or arc_id not in by_arc:
            continue
        points = by_arc[arc_id]
        interval = record.get("interval_fraction")
        segment = _arc_segment(points, interval) if interval is not None else points
        label = record.get("gt_label") or "STATE_ONLY"
        color = colors.get(label, "#56B4E9")
        linewidth = 4.0 if record.get("gt_label") in {"BENEFICIAL_GT", "HARMFUL_GT"} else 2.0
        axis.plot(segment[:, 0], segment[:, 1], color=color, lw=linewidth, alpha=0.9)
        if record.get("roles"):
            midpoint = segment[len(segment) // 2]
            axis.text(midpoint[0], midpoint[1], ",".join(record["roles"][:2]), fontsize=6, color=color)
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_title(f"Contribution partition: {partition_id}\nBlue = state-only/abstained")
    axis.set_xlabel("x / column [px]")
    axis.set_ylabel("y / row [px]")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
