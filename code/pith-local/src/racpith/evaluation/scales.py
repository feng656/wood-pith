"""Frozen parent-section scales for position-metric evaluation (ArcPith).

Implements the scale table required by shusui.txt for the three-layer error
metrics. All quantities are computed once from full-section GT annotations and
must never be recomputed from predictions:

- ``R_F = D_F / 2`` where ``D_F`` is the convex-hull diameter of the outermost
  ring polyline of the FULL parent section (fixed contour of the section, shared
  by every crop of that section);
- ``w_ref`` (representative adjacent ring spacing) and ``w_max`` (p90 of the same
  spacings), estimated center-free from local normals at fixed arc-length
  samples on each outer ring of adjacent ring pairs.

Boundary (shusui.txt): this scale table only enters the evaluator. It must not
be fed back into localization inference.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import ConvexHull

from ..provenance import sha256_file

SCALE_TABLE_SCHEMA = "racpith.section_scales.v1"

# Frozen method parameters (recorded into scales_config.json; changing any of
# them changes the config hash and invalidates downstream lineage).
DEFAULT_PARAMS: dict[str, Any] = {
    "outer_ring_policy": "max_polygon_area",
    "span_policy": "convex_hull_diameter",
    "wref_ring_pairs": "all_adjacent_label_pairs",
    "wref_samples_per_ring": 12,
    "wref_pca_window": 8,
    "wref_cone_angle_deg": 35.0,
    "wref_max_spacing_px": 400.0,
    "wref_quantile": 0.50,
    "wmax_quantile": 0.90,
    "grouping_near_ratio": 0.35,
    "grouping_mid_ratio": 0.70,
}


def ring_polylines(ann_path: str | Path) -> dict[int, list[np.ndarray]]:
    """Load LabelMe annual-ring annotation, keyed by integer ring label."""
    import json

    payload = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    rings: dict[int, list[np.ndarray]] = {}
    for shape in payload.get("shapes", []):
        try:
            label = int(shape["label"])
        except (ValueError, TypeError):
            continue
        pts = np.asarray(shape["points"], dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 4:
            rings.setdefault(label, []).append(pts)
    return rings


def poly_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _hull_diameter(pts: np.ndarray) -> float:
    """Max pairwise distance of the convex hull of ``pts`` (fixed contour span)."""
    if len(pts) < 3:
        if len(pts) == 2:
            return float(np.linalg.norm(pts[0] - pts[1]))
        return float("nan")
    hull = ConvexHull(pts)
    hp = pts[hull.vertices]
    if len(hp) > 400:  # hull rarely this large; keep the quadratic step bounded
        idx = np.linspace(0, len(hp) - 1, 400).astype(int)
        hp = hp[idx]
    diff = hp[:, None, :] - hp[None, :, :]
    return float(np.sqrt((diff**2).sum(-1)).max())


def arc_length_sample(pts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` fixed arc-length sample points on a closed polyline."""
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(segs.sum())
    if total <= 0.0:
        return pts[:1].copy(), np.asarray([0.0])
    s = np.concatenate([[0.0], np.cumsum(segs)])
    out = np.empty((n, 2), dtype=np.float64)
    for k, target in enumerate(np.linspace(0.0, total, n, endpoint=False)):
        i = int(np.searchsorted(s, target) - 1)
        i = max(0, min(i, len(pts) - 2))
        frac = (target - s[i]) / segs[i] if segs[i] > 0 else 0.0
        out[k] = pts[i] * (1.0 - frac) + pts[i + 1] * frac
    return out, segs


def normals_at(pts: np.ndarray, window: int) -> np.ndarray:
    """Unit local normals (min-eigenvector of the PCA window) at each point."""
    n = len(pts)
    if n < 3:
        return np.zeros((n, 2))
    out = np.zeros((n, 2), dtype=np.float64)
    ext = np.vstack([pts[-window:], pts, pts[:window]]) if window else pts
    centroid = pts.mean(axis=0)
    for i in range(n):
        win = ext[i : i + 2 * window + 1]
        cov = (win - win.mean(axis=0)).T @ (win - win.mean(axis=0))
        w, v = np.linalg.eigh(cov)
        normal = v[:, 0]
        if float(normal @ (pts[i] - centroid)) < 0.0:
            normal = -normal
        out[i] = normal
    return out


def spacing_to_next_ring(
    pts: np.ndarray,
    normals: np.ndarray,
    other_rings: list[np.ndarray],
    *,
    cone_angle_deg: float,
    max_spacing_px: float,
) -> list[float]:
    """Distance from each sample point to the adjacent ring along the local normal."""
    cos_lim = math.cos(math.radians(cone_angle_deg))
    values: list[float] = []
    for p, nrm in zip(pts, normals):
        best = None
        for other in other_rings:
            dvec = other - p
            dist = np.linalg.norm(dvec, axis=1)
            ok = dist > 1e-6
            if not ok.any():
                continue
            cos = np.abs(dvec[ok] @ nrm) / dist[ok]
            near = ok & (cos >= cos_lim) & (dist <= max_spacing_px)
            if near.any():
                d = float(dist[near].min())
                best = d if best is None else min(best, d)
        if best is not None:
            values.append(best)
    return values


def compute_section_scales(
    ann_path: str | Path,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compute the frozen scale record for one full section annotation.

    Returns None when the annotation has no usable ring polygons.
    """
    cfg = dict(DEFAULT_PARAMS)
    if params:
        cfg.update(params)
    rings = ring_polylines(ann_path)
    if not rings:
        return None
    areas = {
        label: max(poly_area(poly) for poly in polys) for label, polys in rings.items() if polys
    }
    if not areas:
        return None
    outer_label = max(areas, key=areas.get)
    outer = max(rings[outer_label], key=poly_area)
    d_f = _hull_diameter(outer)
    r_f = d_f / 2.0

    labels = sorted(areas)
    spacings: list[float] = []
    for i in range(len(labels) - 1):
        lo, hi = labels[i], labels[i + 1]
        for poly in rings[lo]:
            if poly_area(poly) < 0.5 * areas[lo]:
                continue  # skip non-primary fragments of the same ring
            pts, _ = arc_length_sample(poly, int(cfg["wref_samples_per_ring"]))
            nrm = normals_at(pts, int(cfg["wref_pca_window"]))
            spacings.extend(
                spacing_to_next_ring(
                    pts,
                    nrm,
                    rings[hi],
                    cone_angle_deg=float(cfg["wref_cone_angle_deg"]),
                    max_spacing_px=float(cfg["wref_max_spacing_px"]),
                )
            )
    w_ref = float(np.quantile(spacings, float(cfg["wref_quantile"]))) if spacings else None
    w_max = float(np.quantile(spacings, float(cfg["wmax_quantile"]))) if spacings else None
    return {
        "schema_version": SCALE_TABLE_SCHEMA,
        "D_F_px": float(d_f),
        "R_F_px": float(r_f),
        "w_ref_px": w_ref,
        "w_max_px": w_max,
        "n_spacing_samples": len(spacings),
        "outer_ring_label": outer_label,
        "n_rings": len(labels),
        "spacing_source": "adjacent_ring_local_normal",
        "source_annotation_path": str(Path(ann_path).resolve()),
        "source_annotation_sha256": sha256_file(ann_path),
    }
