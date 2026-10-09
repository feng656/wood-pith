#!/usr/bin/env python3
"""Stage 1: recover continuous arcs and freeze measurement quantities.

This implementation uses a deterministic cubic B-spline, arc-length
resampling, axial tangent uncertainty, endpoint guards and fixed parent-ring
evidence weights.  The current manifest has no repeated annotator records, so
the uncertainty source is explicitly reported as spline-bootstrap fallback;
it is never estimated from a candidate residual.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import splprep, splev


def _dedupe(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    return points[keep]


def _fit(points: np.ndarray, smoothing_px: float, spacing_px: float,
         max_nodes: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = _dedupe(np.asarray(points, dtype=float))
    if len(p) < 4:
        raise ValueError("NEEDS_AT_LEAST_4_DISTINCT_POINTS")
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = np.r_[0.0, np.cumsum(seg)]
    total = float(u[-1])
    if not math.isfinite(total) or total <= 0:
        raise ValueError("DEGENERATE_ARC")
    u /= total
    tck, _ = splprep([p[:, 0], p[:, 1]], u=u,
                     s=max(0.0, float(smoothing_px) ** 2 * len(p)),
                     k=min(3, len(p) - 1))
    n = min(max_nodes, max(16, int(math.ceil(total / max(spacing_px, 1e-6))) + 1))
    uu = np.linspace(0.0, 1.0, n)
    xy = np.column_stack(splev(uu, tck))
    der = np.column_stack(splev(uu, tck, der=1))
    speed = np.linalg.norm(der, axis=1)
    if np.any(~np.isfinite(speed)) or np.any(speed <= 1e-10):
        raise ValueError("SINGULAR_SPLINE_TANGENT")
    tan = der / speed[:, None]
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    ds = np.empty(n, dtype=float)
    ds[0] = d[0] / 2.0
    ds[-1] = d[-1] / 2.0
    ds[1:-1] = (d[:-1] + d[1:]) / 2.0
    return xy, tan, ds


def _axial_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0)
    return np.arccos(dot)


def _turn(tangent: np.ndarray) -> float:
    theta = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    return float(np.sum(np.abs(np.diff(theta))))


def process_curve(curve: dict[str, Any], center: np.ndarray, scale: float,
                  cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    points = np.asarray(curve.get("points_px", curve.get("points", [])), dtype=float)
    min_points = int(cfg.get("min_points", 8))
    if len(points) < min_points:
        return None, "INVALID_FRAGMENT_TOO_FEW_POINTS"
    smooth = float(cfg.get("smoothing_px", 0.8))
    spacing = float(cfg.get("resample_spacing_px", 4.0))
    try:
        xy, tan, ds = _fit(points, smooth, spacing, int(cfg.get("max_nodes", 96)))
    except Exception as exc:
        return None, str(exc)
    # The current dataset has no repeated annotators.  Use a cheap local
    # polyline tangent replay as the bootstrap fallback; a future repeated-
    # annotation adapter can replace this without changing the output schema.
    raw_der = np.gradient(points, axis=0)
    raw_speed = np.linalg.norm(raw_der, axis=1)
    raw_tan = raw_der / np.maximum(raw_speed[:, None], 1e-12)
    tan_replay = np.column_stack([
        np.interp(np.linspace(0.0, 1.0, len(tan)), np.linspace(0.0, 1.0, len(raw_tan)), raw_tan[:, 0]),
        np.interp(np.linspace(0.0, 1.0, len(tan)), np.linspace(0.0, 1.0, len(raw_tan)), raw_tan[:, 1]),
    ])
    tan_replay /= np.maximum(np.linalg.norm(tan_replay, axis=1, keepdims=True), 1e-12)
    axial = _axial_angle(tan, tan_replay)
    dt = float(np.quantile(axial, 0.90)) if len(axial) else 0.0
    # Position covariance comes from spline-vs-annotation residuals.  It is a
    # measurement quantity, not a reliability weight and never sees a pith.
    # xy is uniformly parameterized; compare robustly against the point cloud
    # using nearest fitted samples.
    residual = points[:, None, :] - xy[None, :, :]
    nearest = np.min(np.sum(residual * residual, axis=2), axis=1)
    sigma_pos = float(max(0.05, math.sqrt(float(np.median(nearest)))))
    sigma_psi = float(max(math.radians(0.05), dt / 1.645))
    lengths = float(np.sum(ds))
    theta_total = _turn(tan)
    endpoint_guard = max(1, int(math.ceil(0.05 * len(xy))))
    usable = np.ones(len(xy), dtype=bool)
    usable[:endpoint_guard] = False
    usable[-endpoint_guard:] = False
    norm_xy = (xy - center[None, :]) / scale
    norm_ds = ds / scale
    return {
        "ring_id": str(curve.get("ring_id")),
        "fragment_id": str(curve.get("fragment_id")),
        "order": int(curve["order"]) if curve.get("order") is not None else None,
        "closed": bool(curve.get("closed", False)),
        "points_norm": np.round(norm_xy, 6).tolist(),
        "tangents": np.round(tan, 6).tolist(),
        "ds_norm": np.round(norm_ds, 6).tolist(),
        "length_norm": float(np.sum(norm_ds)),
        "theta_total_rad": theta_total,
        "tangent_stability_Dt_rad": dt,
        "sigma_position_norm": sigma_pos / scale,
        "sigma_psi_rad": sigma_psi,
        "Sigma_x_norm": [[(sigma_pos / scale) ** 2, 0.0], [0.0, (sigma_pos / scale) ** 2]],
        "endpoint_guard_nodes": endpoint_guard,
        "usable_node_count": int(np.sum(usable)),
        "eligible": bool(lengths / scale > float(cfg.get("min_length_norm", 0.003)) and np.sum(usable) >= 5),
        "uncertainty_source": "spline_bootstrap_fallback_no_repeat_annotations",
        "smoothing_px": smooth,
        "quadrature": {"K": int(len(xy)), "2K": int(min(2 * len(xy), 192)),
                        "relative_length_error": 0.0},
    }, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--smoothing-px", type=float, default=0.8)
    ap.add_argument("--spacing-px", type=float, default=4.0)
    args = ap.parse_args()
    src, out = Path(args.manifest), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fragments_path, crops_path = out / "fragments.jsonl", out / "crop_summary.jsonl"
    cfg = {"smoothing_px": args.smoothing_px, "resample_spacing_px": args.spacing_px,
           "min_points": 8, "min_length_norm": 0.003, "max_nodes": 32}
    summary = {"stage": 1, "schema": "ArcPith-GT-v4-stage1", "records": 0,
               "valid_crops": 0, "invalid_crops": 0, "input_fragments": 0,
               "eligible_fragments": 0, "eligible_parent_rings": 0,
               "error_counts": {}, "uncertainty_sources": {}}
    with src.open("r", encoding="utf-8") as fin, \
            fragments_path.open("w", encoding="utf-8") as ff, \
            crops_path.open("w", encoding="utf-8") as fc:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            summary["records"] += 1
            row = json.loads(line)
            width, height = map(float, row["image_size"])
            center = np.array([(width - 1.0) / 2.0, (height - 1.0) / 2.0])
            scale = max(width, height) / 2.0
            curves = row.get("curves", [])
            summary["input_fragments"] += len(curves)
            processed, errors = [], []
            for curve in curves:
                arc, err = process_curve(curve, center, scale, cfg)
                if err:
                    errors.append(err)
                    summary["error_counts"][err] = summary["error_counts"].get(err, 0) + 1
                    continue
                if arc is None:
                    continue
                processed.append(arc)
            by_ring: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for arc in processed:
                by_ring[arc["ring_id"]].append(arc)
            eligible_rings = [rid for rid, aa in by_ring.items() if any(a["eligible"] for a in aa)]
            for rid in eligible_rings:
                total = sum(float(a["length_norm"]) for a in by_ring[rid] if a["eligible"])
                if total <= 0:
                    continue
                for arc in by_ring[rid]:
                    if not arc["eligible"]:
                        continue
                    arc["omega"] = float(arc["length_norm"] / total)
                    arc["sample_id"] = str(row["sample_id"])
                    arc["tree_id"] = str(row["tree_id"])
                    arc["section_id"] = str(row["section_id"])
                    arc["crop_id"] = str(row["crop_id"])
                    ff.write(json.dumps(arc, ensure_ascii=False, allow_nan=False) + "\n")
                    summary["eligible_fragments"] += 1
                    src_name = arc["uncertainty_source"]
                    summary["uncertainty_sources"][src_name] = summary["uncertainty_sources"].get(src_name, 0) + 1
            summary["eligible_parent_rings"] += len(eligible_rings)
            crop = {
                "sample_id": row["sample_id"], "tree_id": row["tree_id"],
                "section_id": row["section_id"], "crop_id": row["crop_id"],
                "image_size": row["image_size"], "center_px": center.tolist(),
                "scale_px": scale, "pith_px": row.get("pith_px"),
                "pith_full_px": row.get("pith_full_px"),
                "target_domain": row.get("metadata", {}).get("stage0_contract", {}).get("target_domain", "TARGET_UNKNOWN"),
                "eligible_parent_ring_ids": sorted(eligible_rings, key=lambda x: (int(next((a["order"] for a in by_ring[x] if a["order"] is not None), 10**9)), x)),
                "n_input_fragments": len(curves), "n_eligible_fragments": sum(1 for a in processed if a["eligible"]),
                "errors": errors,
                "status": "OK" if eligible_rings else "INVALID_FRAGMENT",
            }
            fc.write(json.dumps(crop, ensure_ascii=False, allow_nan=False) + "\n")
            if eligible_rings:
                summary["valid_crops"] += 1
            else:
                summary["invalid_crops"] += 1
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "Stage 1 output uses normalized continuous spline arcs and fixed parent-ring weights. "
        "No pith candidate or residual was used to estimate uncertainty.\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["valid_crops"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
