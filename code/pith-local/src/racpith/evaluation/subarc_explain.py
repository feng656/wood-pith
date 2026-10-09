"""Sub-arc explainability layer (geometric diagnostics, not formal contributions).

Slices every stage-1 fragment into 10 arc-length sub-arcs (matching the
``subarc:f=0.1:phase=0`` partition of the contribution pipeline) and computes,
per sub-arc, a geometric explanation of why it should (or should not) influence
the localization:

- pointing agreement (mean |cos| between the local normal and the direction to
  the final estimated center);
- angular span, arc length, distance to center;
- crop-boundary adjacency, ring zone (inner/mid/outer);
- inherited ring-level delete-refit role from the stage-6 exact contributions.

Boundary: this is a diagnostic layer. It does not override ``gt_label`` /
``contrib_gt_norm``, and HARMFUL_GT roles are post-hoc audit, not a deployment
rule for deleting arcs.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

SUBARC_EXPLAIN_SCHEMA = "racpith.subarc_explain.v1"
SUBARC_FRACTION = 0.1
BOUNDARY_MARGIN_PX = 5.0

ROLE_PRIORITY = [
    "HARMFUL_GT",
    "BENEFICIAL_GT",
    "DIRECTION_CRITICAL",
    "HIGH_LEVERAGE",
    "RANGE_CRITICAL",
    "MODE_EXCLUSION",
    "REDUNDANT",
    "UNRESOLVED",
]


def subarc_bins(
    ds_norm: Sequence[float],
    fraction: float = SUBARC_FRACTION,
    phase: float = 0.0,
) -> list[tuple[int, float, float, np.ndarray]]:
    """(bin_index, lo, hi, node_mask) per non-empty sub-arc bin, arc-length based.

    Mirrors ``racpith.groups.build_deletion_groups`` for subarc partitions:
    nodes are assigned by their cumulative arc-length fraction, hi is clipped to
    1.0, and empty tail bins are skipped.
    """
    ds = np.asarray(ds_norm, dtype=np.float64)
    total = float(ds.sum())
    if total <= 1e-12 or len(ds) == 0:
        return []
    cum = np.concatenate([[0.0], np.cumsum(ds)]) / total  # node boundary fractions
    fracs = np.minimum(0.5 * (cum[:-1] + cum[1:]), np.nextafter(1.0, 0.0))  # node midpoints
    indices = np.floor((fracs - phase * fraction) / fraction).astype(np.int64)
    bins: list[tuple[int, float, float, np.ndarray]] = []
    for index in range(int(np.floor(1.0 / fraction)) + 1):
        lo = max(0.0, (float(index) + phase) * fraction)
        hi = min(1.0, (float(index) + phase + 1.0) * fraction)
        if hi <= lo:
            continue
        mask = indices == index
        if not mask.any():
            continue
        bins.append((index, lo, hi, mask))
    return bins


def agreement_value(
    points: np.ndarray,
    tangents: np.ndarray,
    center: np.ndarray,
) -> float | None:
    """Mean |cos| between local normals and the direction toward ``center``."""
    if len(points) == 0 or not np.all(np.isfinite(center)):
        return None
    dvec = center[None, :] - points
    dist = np.linalg.norm(dvec, axis=1)
    valid = dist > 1e-9
    if not valid.any():
        return None
    t = tangents[valid]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    cos = np.abs(np.sum(n * (dvec[valid] / dist[valid, None]), axis=1))
    return float(cos.mean())


def _span_rad(tangents: np.ndarray) -> float:
    if len(tangents) < 2:
        return 0.0
    t0, t1 = tangents[:-1], tangents[1:]
    cos = np.clip(np.sum(t0 * t1, axis=1), -1.0, 1.0)
    return float(np.arccos(cos).sum())


def _boundary_hits(points: np.ndarray, width: float, height: float, margin: float) -> bool:
    if len(points) == 0:
        return False
    return bool(
        np.any(points[:, 0] <= margin)
        or np.any(points[:, 0] >= width - margin)
        or np.any(points[:, 1] <= margin)
        or np.any(points[:, 1] >= height - margin)
    )


def classify_subarc(
    boundary: bool,
    agreement: float | None,
    length_px: float,
) -> tuple[str, str]:
    """Return (category, effect hypothesis). Deterministic rule chain."""
    if agreement is None:
        return "NO_CENTER", "无法计算(最终估计无坐标)"
    if boundary:
        return "BOUNDARY_TRUNCATED", "贴边截断→曲率约束弱→删除后影响小"
    if agreement >= 0.90:
        if length_px >= 30.0:
            return "SUPPORTING", "高指向一致度+有效弧长→支撑共识方向→删除后误差升"
        return "SUPPORTING_SHORT", "指向一致但弧长不足→支撑有限"
    if agreement < 0.70:
        return "DEVIATING", "法线与共识方向偏离→指向冲突→删除后误差降"
    return "MODERATE", "指向中间→影响取决于邻段与整体约束"


def build_why_text(
    category: str,
    effect: str,
    *,
    agreement: float | None,
    length_px: float,
    span_rad: float,
    boundary: bool,
    ring_zone: str,
    role: str | None,
) -> str:
    if category == "NO_CENTER":
        return "最终估计无坐标(REJECT),无法计算指向一致度;几何属性仅作记录。"
    parts = []
    if boundary:
        parts.append("子弧段贴 crop 边界、有效弧长不足")
    parts.append(f"指向一致度 mean|cos|={agreement:.2f}" if agreement is not None else "")
    parts.append(f"弧长 {length_px:.1f}px、角跨度 {math.degrees(span_rad):.1f}°、{ring_zone} 环区")
    role_txt = f"环级 delete-refit 角色 {role}" if role else "该环无 stage-6 精确贡献记录"
    return "原因链:" + ";".join(p for p in parts if p) + f"。推断:{effect}。实测对照:{role_txt}。"


def explain_fragment(
    fragment: Mapping[str, Any],
    *,
    scale_px: float,
    center_px: Sequence[float],
    image_size: Sequence[float],
    final_center: np.ndarray | None,
    ring_role: Sequence[str] | None,
    max_ring: int,
) -> list[dict[str, Any]]:
    """Build one explanation record per non-empty sub-arc of a fragment."""
    points_norm = np.asarray(fragment["points_norm"], dtype=np.float64)
    tangents = np.asarray(fragment["tangents"], dtype=np.float64)
    ds = np.asarray(fragment["ds_norm"], dtype=np.float64)
    points_px = points_norm * float(scale_px) + np.asarray(center_px, dtype=np.float64)[None, :]
    width, height = (float(v) for v in image_size)
    ring_id = str(fragment["ring_id"])
    try:
        zone = "INNER" if int(ring_id) <= 1 else ("OUTER" if int(ring_id) >= max_ring - 1 else "MID")
    except ValueError:
        zone = "MID"
    primary_role = next((r for r in ROLE_PRIORITY if r in (ring_role or [])), None)
    records: list[dict[str, Any]] = []
    for index, lo, hi, mask in subarc_bins(ds):
        pts = points_px[mask]
        tgs = tangents[mask]
        if len(pts) < 2:
            continue
        length_px = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        agreement = agreement_value(pts, tgs, final_center) if final_center is not None else None
        boundary = _boundary_hits(pts, width, height, BOUNDARY_MARGIN_PX)
        category, effect = classify_subarc(boundary, agreement, length_px)
        span = _span_rad(tgs)
        # subsample for compact storage (<= 12 points)
        step = max(1, int(np.ceil(len(pts) / 12.0)))
        records.append(
            {
                "schema_version": SUBARC_EXPLAIN_SCHEMA,
                "crop_id": fragment["crop_id"],
                "ring_id": ring_id,
                "fragment_id": str(fragment["fragment_id"]),
                "interval_fraction": [lo, hi],
                "category": category,
                "effect_hypothesis": effect,
                "agreement": agreement,
                "length_px": length_px,
                "span_rad": span,
                "dist_to_center_px": (
                    float(np.linalg.norm(pts - final_center[None, :], axis=1).mean())
                    if final_center is not None
                    else None
                ),
                "boundary": boundary,
                "ring_zone": zone,
                "parent_theta_total_rad": fragment.get("theta_total_rad"),
                "parent_closed": fragment.get("closed"),
                "parent_endpoint_guard_nodes": fragment.get("endpoint_guard_nodes"),
                "uncertainty_source": fragment.get("uncertainty_source"),
                "ring_role": primary_role,
                "ring_roles": sorted(ring_role) if ring_role else None,
                "why": build_why_text(
                    category,
                    effect,
                    agreement=agreement,
                    length_px=length_px,
                    span_rad=span,
                    boundary=boundary,
                    ring_zone=zone,
                    role=primary_role,
                ),
                "points_px": np.round(pts[::step], 2).tolist(),
            }
        )
    return records


def explain_from_evidence(
    bundle: Mapping[str, Any],
    *,
    crop_id: str,
    image_size: Sequence[float],
    final_center: np.ndarray | None,
    max_ring: int,
) -> list[dict[str, Any]]:
    """Build explanation records directly from an evidence bundle (smoke crops).

    Uses the exact sampled quadrature nodes (``points_crop_px``/``tangents``/
    ``arc_fraction``) so sub-arc boundaries align with the contribution groups.
    """
    width, height = (float(v) for v in image_size)
    records: list[dict[str, Any]] = []
    for arc_id, index in bundle["arc_index"].items():
        ring_id = str(arc_id).split(":", 1)[0]
        try:
            zone = "INNER" if int(ring_id) <= 1 else ("OUTER" if int(ring_id) >= max_ring - 1 else "MID")
        except ValueError:
            zone = "MID"
        arc_mask = bundle["arc_indices"] == int(index)
        fractions = bundle["arc_fractions"][arc_mask]
        for bin_index in range(10):
            lo, hi = bin_index * SUBARC_FRACTION, min(1.0, (bin_index + 1) * SUBARC_FRACTION)
            if hi <= lo:
                continue
            mask = arc_mask.copy()
            mask[mask] = (fractions >= lo - 1e-12) & (fractions <= hi + 1e-12)
            pts = bundle["points_crop_px"][mask]
            tgs = bundle["tangents"][mask]
            if len(pts) < 2:
                continue
            length_px = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            agreement = agreement_value(pts, tgs, final_center) if final_center is not None else None
            boundary = _boundary_hits(pts, width, height, BOUNDARY_MARGIN_PX)
            category, effect = classify_subarc(boundary, agreement, length_px)
            step = max(1, int(np.ceil(len(pts) / 12.0)))
            records.append(
                {
                    "schema_version": SUBARC_EXPLAIN_SCHEMA,
                    "crop_id": crop_id,
                    "ring_id": ring_id,
                    "fragment_id": arc_id.split(":", 1)[1] if ":" in arc_id else "0",
                    "interval_fraction": [lo, hi],
                    "category": category,
                    "effect_hypothesis": effect,
                    "agreement": agreement,
                    "length_px": length_px,
                    "span_rad": _span_rad(tgs),
                    "dist_to_center_px": (
                        float(np.linalg.norm(pts - final_center[None, :], axis=1).mean())
                        if final_center is not None
                        else None
                    ),
                    "boundary": boundary,
                    "ring_zone": zone,
                    "ring_role": None,
                    "ring_roles": None,
                    "why": build_why_text(
                        category,
                        effect,
                        agreement=agreement,
                        length_px=length_px,
                        span_rad=_span_rad(tgs),
                        boundary=boundary,
                        ring_zone=zone,
                        role=None,
                    ),
                    "points_px": np.round(pts[::step], 2).tolist(),
                }
            )
    return records
