from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

from .contracts import EvidenceBundle
from .provenance import read_json_object, sha256_file


@dataclass(frozen=True)
class _SampledArc:
    ring_id: str
    arc_id: str
    xy_px: np.ndarray
    tangent: np.ndarray
    ds_px: np.ndarray
    source_s_px: np.ndarray
    arc_fraction: np.ndarray


def _drop_adjacent_duplicates(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        return points, values
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    return points[keep], values[keep]


def _registered_boundaries(
    length: float,
    step: float,
    subarc_fractions: list[float],
    phases: list[float],
) -> np.ndarray:
    regular = np.arange(0.0, length, step, dtype=np.float64)
    values = [0.0, length, *regular.tolist()]
    for fraction in subarc_fractions:
        if not 0 < fraction <= 1:
            raise ValueError(f"invalid subarc fraction {fraction}")
        block = fraction * length
        for phase in phases:
            if not 0 <= phase < 1:
                raise ValueError(f"invalid subarc phase {phase}")
            start = phase * block
            if start > 0:
                values.append(start)
            values.extend(np.arange(start, length, block, dtype=np.float64).tolist())
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, length)
    return np.unique(np.round(clipped, decimals=12))


def _sample_visible_arc(
    ring_id: str,
    arc: Mapping[str, Any],
    evidence_cfg: Mapping[str, Any],
    contribution_cfg: Mapping[str, Any],
) -> _SampledArc | None:
    points = np.asarray(arc["points_crop_px"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"arc {arc.get('arc_id')} points must have shape (N,2)")
    source_s = np.asarray(
        arc.get("source_s_px", np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]),
        dtype=np.float64,
    )
    if source_s.shape != (len(points),):
        raise ValueError(f"arc {arc.get('arc_id')} source_s_px length mismatch")
    points, source_s = _drop_adjacent_duplicates(points, source_s)
    closed = bool(arc.get("closed", False))
    if closed and len(points) > 1 and np.linalg.norm(points[0] - points[-1]) < 1e-8:
        points, source_s = points[:-1], source_s[:-1]
    if len(points) < 4:
        return None

    chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if closed:
        chord = np.r_[chord, np.linalg.norm(points[0] - points[-1])]
    total_chord = float(chord.sum())
    if total_chord < float(evidence_cfg["min_fragment_length_px"]):
        return None
    u = np.r_[0.0, np.cumsum(chord[:-1])] / total_chord if closed else np.r_[0.0, np.cumsum(chord)] / total_chord
    smooth = float(evidence_cfg["spline_smoothing_px2_per_point"]) * len(points)
    degree = min(3, len(points) - 1)
    tck, _ = splprep(points.T, u=u, s=smooth, k=degree, per=closed)

    dense_count = max(256, int(np.ceil(total_chord / 0.25)))
    dense_u = np.linspace(0.0, 1.0, dense_count, endpoint=not closed)
    dense_xy = np.column_stack(splev(dense_u, tck, der=0))
    if closed:
        loop_xy = np.vstack([dense_xy, dense_xy[0]])
        dense_length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(loop_xy, axis=0), axis=1))]
        dense_u_length = np.r_[dense_u, 1.0]
    else:
        dense_length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense_xy, axis=0), axis=1))]
        dense_u_length = dense_u
    spline_length = float(dense_length[-1])
    guard = 0.0 if closed else float(evidence_cfg["endpoint_guard_px"])
    lo, hi = guard, spline_length - guard
    if hi - lo < float(evidence_cfg["min_fragment_length_px"]):
        return None

    registered_fractions = sorted(
        {
            float(contribution_cfg["primary_subarc_fraction"]),
            *(float(value) for value in contribution_cfg["audit_subarc_fractions"]),
        }
    )
    registered_phases = sorted(
        {0.0, *(float(value) for value in contribution_cfg["audit_phases"])}
    )
    boundaries = _registered_boundaries(
        hi - lo,
        float(evidence_cfg["quadrature_step_px"]),
        registered_fractions,
        registered_phases,
    ) + lo
    if len(boundaries) - 1 < int(evidence_cfg["min_nodes_per_fragment"]):
        # Add nodes without discarding any registered scale/phase boundary.
        # Otherwise one quadrature cell could straddle two deletion groups.
        supplemental = np.linspace(
            lo,
            hi,
            int(evidence_cfg["min_nodes_per_fragment"]) + 1,
            dtype=np.float64,
        )
        boundaries = np.unique(
            np.round(np.concatenate((boundaries, supplemental)), decimals=12)
        )
    mid_length = 0.5 * (boundaries[:-1] + boundaries[1:])
    ds = np.diff(boundaries)
    u_mid = np.interp(mid_length, dense_length, dense_u_length)
    xy = np.column_stack(splev(u_mid, tck, der=0))
    derivative = np.column_stack(splev(u_mid, tck, der=1))
    tangent_norm = np.linalg.norm(derivative, axis=1)
    if np.any(tangent_norm <= 1e-12):
        raise ValueError(f"arc {arc.get('arc_id')} has a zero spline derivative")
    tangent = derivative / tangent_norm[:, None]

    source_u = u
    source_values = source_s
    if closed:
        period = max(float(source_s[-1] - source_s[0]), total_chord)
        source_u = np.r_[source_u, 1.0]
        source_values = np.r_[source_values, source_s[0] + period]
    sampled_source_s = np.interp(u_mid, source_u, source_values)
    fraction = np.clip((mid_length - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return _SampledArc(
        ring_id=ring_id,
        arc_id=str(arc["arc_id"]),
        xy_px=xy,
        tangent=tangent,
        ds_px=ds,
        source_s_px=sampled_source_s,
        arc_fraction=fraction,
    )


def _bilinear(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(xy[:, 0], 0.0, width - 1.000001)
    y = np.clip(xy[:, 1], 0.0, height - 1.000001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return (
        image[y0, x0] * (1 - wx) * (1 - wy)
        + image[y0, x1] * wx * (1 - wy)
        + image[y1, x0] * (1 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(values, [0.10, 0.90])
    if hi <= lo + 1e-12:
        return np.full_like(values, 0.5, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _image_quality(
    image_bgr: np.ndarray,
    xy: np.ndarray,
    tangent: np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    band = int(cfg["quality_normal_band_px"])
    offsets = np.arange(-band, band + 1, dtype=np.float64)
    normal_response = []
    for offset in offsets:
        sample_xy = xy + offset * normal
        normal_response.append(
            np.abs(_bilinear(gx, sample_xy) * normal[:, 0] + _bilinear(gy, sample_xy) * normal[:, 1])
        )
    support_raw = np.max(np.vstack(normal_response), axis=0)
    support = _robust_unit_interval(support_raw)

    window = int(cfg["quality_tensor_window_px"])
    if window % 2 == 0:
        window += 1
    jxx = cv2.GaussianBlur(gx * gx, (window, window), 0)
    jyy = cv2.GaussianBlur(gy * gy, (window, window), 0)
    jxy = cv2.GaussianBlur(gx * gy, (window, window), 0)
    a = _bilinear(jxx, xy)
    b = _bilinear(jxy, xy)
    d = _bilinear(jyy, xy)
    trace = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    coherence = disc / np.maximum(trace, 1e-12)
    nx, ny = normal[:, 0], normal[:, 1]
    normal_energy = a * nx * nx + 2 * b * nx * ny + d * ny * ny
    orientation = np.clip(normal_energy / np.maximum(0.5 * (trace + disc), 1e-12), 0.0, 1.0)
    orientation *= np.clip(coherence, 0.0, 1.0)

    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
    anomaly = _robust_unit_interval(_bilinear(lap, xy))
    q = (
        float(cfg["quality_support_weight"]) * support
        + float(cfg["quality_orientation_weight"]) * orientation
        - float(cfg["quality_anomaly_weight"]) * anomaly
    )
    q = np.clip(q, 0.0, 1.0)
    return q, anomaly


def build_evidence(
    crop_annotation_path: str | Path,
    crop_image_path: str | Path,
    full_config: Mapping[str, Any],
    *,
    config_hash: str | None = None,
    crop_annotation_sha256: str | None = None,
    crop_image_sha256: str | None = None,
) -> EvidenceBundle:
    actual_annotation_sha256 = sha256_file(crop_annotation_path)
    actual_image_sha256 = sha256_file(crop_image_path)
    if (
        crop_annotation_sha256 is not None
        and actual_annotation_sha256 != crop_annotation_sha256
    ):
        raise ValueError("crop annotation content differs from its manifest hash")
    if crop_image_sha256 is not None and actual_image_sha256 != crop_image_sha256:
        raise ValueError("crop image content differs from its manifest hash")
    annotation = read_json_object(crop_annotation_path)
    if annotation.get("schema_version") != "racpith.crop_annotation.v1":
        raise ValueError("unsupported crop annotation schema")
    evidence_cfg = full_config["evidence"]
    contribution_cfg = full_config["contribution"]
    image = cv2.imread(str(crop_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read crop image {crop_image_path}")
    height, width = image.shape[:2]
    declared_width, declared_height = annotation["crop_size_px"]
    if (width, height) != (declared_width, declared_height):
        raise ValueError("crop image dimensions disagree with annotation")

    sampled: list[_SampledArc] = []
    for ring in annotation["rings"]:
        ring_id = str(ring["ring_id"])
        for arc in ring["arcs"]:
            if arc.get("qualifies_for_evidence") is False:
                continue
            item = _sample_visible_arc(ring_id, arc, evidence_cfg, contribution_cfg)
            if item is not None:
                sampled.append(item)
    valid_ring_ids = tuple(dict.fromkeys(item.ring_id for item in sampled))
    if len(valid_ring_ids) < int(full_config["solver"]["min_parent_rings"]):
        raise ValueError("crop has too few valid parent rings after continuous-arc checks")
    arc_ids = tuple(item.arc_id for item in sampled)
    if len(set(arc_ids)) != len(arc_ids):
        raise ValueError("arc IDs are not unique within crop")
    ring_lookup = {value: index for index, value in enumerate(valid_ring_ids)}
    arc_lookup = {value: index for index, value in enumerate(arc_ids)}

    xy_px = np.vstack([item.xy_px for item in sampled])
    mask_domain_clamp_count = 0
    out_of_domain = (
        (xy_px[:, 0] < 0.0)
        | (xy_px[:, 0] >= float(width))
        | (xy_px[:, 1] < 0.0)
        | (xy_px[:, 1] >= float(height))
    )
    if np.any(out_of_domain):
        # Rasterised dataset_grid curves can lie exactly on the crop border;
        # a smoothing spline may overshoot by a tiny numerical amount.  Keep
        # the source annotation unchanged and clamp only these sampled mask
        # points back into the half-open image domain.  Vector/RGB inputs remain
        # fail-closed as before.
        is_dataset_grid = (
            annotation.get("metadata", {}).get("source_mask_convention") == "dataset_grid"
        )
        if not is_dataset_grid:
            raise ValueError(
                "smoothed quadrature evidence leaves the half-open crop domain"
            )
        mask_domain_clamp_count = int(np.count_nonzero(out_of_domain))
        xy_px[:, 0] = np.clip(xy_px[:, 0], 0.0, np.nextafter(float(width), 0.0))
        xy_px[:, 1] = np.clip(xy_px[:, 1], 0.0, np.nextafter(float(height), 0.0))
    tangent = np.vstack([item.tangent for item in sampled])
    ds_px = np.concatenate([item.ds_px for item in sampled])
    source_s = np.concatenate([item.source_s_px for item in sampled])
    arc_fraction = np.concatenate([item.arc_fraction for item in sampled])
    ring_index = np.concatenate(
        [np.full(len(item.xy_px), ring_lookup[item.ring_id], dtype=np.int64) for item in sampled]
    )
    arc_index = np.concatenate(
        [np.full(len(item.xy_px), arc_lookup[item.arc_id], dtype=np.int64) for item in sampled]
    )

    d_fov = float(np.hypot(width, height))
    center = np.asarray([width / 2.0, height / 2.0])
    xy_norm = (xy_px - center) / d_fov
    alpha = 1.0 / len(valid_ring_ids)
    base_weight = np.empty(len(xy_px), dtype=np.float64)
    for ring_i in range(len(valid_ring_ids)):
        mask = ring_index == ring_i
        visible_length = float(ds_px[mask].sum())
        if visible_length <= 0:
            raise ValueError("parent ring has non-positive effective visible length")
        base_weight[mask] = alpha * ds_px[mask] / visible_length

    if bool(evidence_cfg["image_quality_enabled"]):
        quality, anomaly = _image_quality(image, xy_px, tangent, evidence_cfg)
    else:
        quality = np.ones(len(xy_px), dtype=np.float64)
        anomaly = np.zeros(len(xy_px), dtype=np.float64)
    floor_px = float(evidence_cfg["sigma_x_floor_px"])
    ceiling_px = float(evidence_cfg["sigma_x_ceiling_px"])
    degradation = np.clip(1.0 - quality + 0.25 * anomaly, 0.0, 1.0)
    sigma_px = floor_px * np.exp(np.log(ceiling_px / floor_px) * degradation)
    sigma_x = sigma_px / d_fov
    sigma_alg = np.maximum(2.0 * (np.linalg.norm(xy_norm, axis=1) + 1.0) * sigma_x, 1e-12)

    bundle = EvidenceBundle(
        crop_id=str(annotation["crop_id"]),
        tree_id=str(annotation["tree_id"]),
        section_id=str(annotation["section_id"]),
        points_norm=xy_norm.astype(np.float64),
        points_crop_px=xy_px.astype(np.float64),
        tangents=tangent.astype(np.float64),
        sigma_x_norm=sigma_x.astype(np.float64),
        sigma_alg_norm2=sigma_alg.astype(np.float64),
        base_weight=base_weight,
        ring_index=ring_index,
        arc_index=arc_index,
        source_s_px=source_s,
        arc_fraction=arc_fraction,
        quality=quality,
        ring_ids=valid_ring_ids,
        arc_ids=arc_ids,
        crop_origin_source_px=tuple(annotation["crop_origin_source_px"]),
        crop_size_px=(width, height),
        normalization_scale_px=d_fov,
        metadata={
            "annotation_path": str(Path(crop_annotation_path).resolve()),
            "image_path": str(Path(crop_image_path).resolve()),
            "crop_annotation_sha256": actual_annotation_sha256,
            "crop_image_sha256": actual_image_sha256,
            "evidence_config_hash": config_hash,
            # Ring order is source annotation provenance, not a quantity inferred
            # from a candidate centre.  Preserve it so the state machine may
            # distinguish an oriented RAY from an unoriented AXIS only when the
            # crop adapter has certified the ordering.
            "ring_order_reliable": bool(
                annotation.get("metadata", {}).get("ring_order_reliable", False)
            ),
            "ring_order_by_id": {
                str(ring["ring_id"]): int(ring["ring_order"])
                for ring in annotation["rings"]
                if str(ring["ring_id"]) in valid_ring_ids
            },
            "quality_is_candidate_independent": True,
            "sigma_x_units": "normalized_length",
            "sigma_alg_units": "normalized_length_squared",
            "parent_budget_policy": "equal_parent_frozen_mass",
            "mask_domain_clamp_count": mask_domain_clamp_count,
        },
    )
    bundle.validate()
    return bundle
