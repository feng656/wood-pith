#!/usr/bin/env python3
"""Stage 0: validate and freeze the ArcPith data/target contract.

The output is a normalized JSONL manifest.  This stage never estimates a
pith and never uses a candidate loss.  Missing physical scale is reported as
PIXEL_ONLY instead of being silently invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED = ("sample_id", "tree_id", "section_id", "crop_id", "image",
            "image_size", "curves")


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite_pair(value: Any) -> bool:
    try:
        return len(value) == 2 and all(math.isfinite(float(x)) for x in value)
    except (TypeError, ValueError):
        return False


def validate_row(row: dict[str, Any], line_no: int) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for key in REQUIRED:
        if key not in row:
            errors.append(f"MISSING_{key.upper()}")
    if errors:
        return row, errors, warnings

    sample_id = str(row["sample_id"])
    image_size = row["image_size"]
    if not _finite_pair(image_size) or any(float(x) <= 0 for x in image_size):
        errors.append("INVALID_IMAGE_SIZE")
        return row, errors, warnings
    width, height = int(image_size[0]), int(image_size[1])

    origin = row.get("crop_origin_px", [0.0, 0.0])
    if not _finite_pair(origin):
        errors.append("INVALID_CROP_ORIGIN")
        origin = [0.0, 0.0]
    origin = [float(origin[0]), float(origin[1])]

    image_path = Path(str(row["image"]))
    if not image_path.is_absolute() or not image_path.exists():
        errors.append("IMAGE_NOT_FOUND")

    scale = row.get("mm_per_pixel")
    if scale is None:
        warnings.append("PIXEL_ONLY_NO_MM_PER_PIXEL")
        scale_status = "PIXEL_ONLY"
    else:
        try:
            if not math.isfinite(float(scale)) or float(scale) <= 0:
                errors.append("INVALID_MM_PER_PIXEL")
                scale_status = "INVALID"
            else:
                scale_status = "PHYSICAL_SCALE_AVAILABLE"
        except (TypeError, ValueError):
            errors.append("INVALID_MM_PER_PIXEL")
            scale_status = "INVALID"

    curves = row.get("curves", row.get("rings", []))
    if not isinstance(curves, list) or not curves:
        errors.append("NO_CURVES")
        curves = []
    seen_orders: dict[str, int] = {}
    normalized_curves = []
    for ci, curve in enumerate(curves):
        rid = str(curve.get("ring_id", ""))
        fid = str(curve.get("fragment_id", f"{rid}:fragment:{ci}"))
        points = curve.get("points_px", curve.get("points"))
        order = curve.get("order")
        if not rid:
            errors.append(f"CURVE_{ci}_MISSING_PARENT_RING_ID")
            continue
        if order is None:
            errors.append(f"CURVE_{ci}_MISSING_RING_ORDER")
        else:
            try:
                order = int(order)
                if rid in seen_orders and seen_orders[rid] != order:
                    errors.append(f"PARENT_RING_{rid}_CONFLICTING_ORDER")
                seen_orders[rid] = order
            except (TypeError, ValueError):
                errors.append(f"CURVE_{ci}_INVALID_RING_ORDER")
                order = None
        if not isinstance(points, list) or len(points) < 2:
            errors.append(f"CURVE_{ci}_TOO_FEW_POINTS")
            continue
        pts = []
        for pi, point in enumerate(points):
            if not _finite_pair(point):
                errors.append(f"CURVE_{ci}_POINT_{pi}_INVALID")
                continue
            pts.append([float(point[0]), float(point[1])])
        if len(pts) < 2:
            continue
        if len({tuple(p) for p in pts}) < 2:
            errors.append(f"CURVE_{ci}_DEGENERATE")
            continue
        normalized_curves.append({
            "ring_id": rid, "fragment_id": fid, "order": order,
            "closed": bool(curve.get("closed", False)),
            "visibility": curve.get("visibility", "unknown"),
            "points_px": pts,
        })

    # A parent ring may have several visible fragments.  Repeated order is
    # valid within that ring; only distinct parent rings must not collide.
    orders = [int(order) for rid, order in seen_orders.items() if order is not None]
    if len(orders) != len(set(orders)):
        errors.append("DUPLICATE_RING_ORDER")

    pith_full = row.get("pith_full_px")
    pith_local = row.get("pith_px")
    if pith_full is not None and not _finite_pair(pith_full):
        errors.append("INVALID_PITH_FULL")
        pith_full = None
    if pith_local is not None and not _finite_pair(pith_local):
        errors.append("INVALID_PITH_LOCAL")
        pith_local = None
    if pith_local is None and pith_full is not None:
        derived = [float(pith_full[0]) - origin[0], float(pith_full[1]) - origin[1]]
        if 0 <= derived[0] < width and 0 <= derived[1] < height:
            pith_local = derived
    trunc = {
        "left": bool(pith_full is not None and float(pith_full[0]) < origin[0]),
        "right": bool(pith_full is not None and float(pith_full[0]) >= origin[0] + width),
        "top": bool(pith_full is not None and float(pith_full[1]) < origin[1]),
        "bottom": bool(pith_full is not None and float(pith_full[1]) >= origin[1] + height),
    }
    trunc["pith_outside"] = any(trunc.values())
    if pith_full is None:
        warnings.append("PITH_GT_UNAVAILABLE")
    roundtrip_ok = True
    if pith_full is not None and pith_local is not None:
        roundtrip_ok = bool(np.allclose(np.asarray(pith_local) + np.asarray(origin),
                                        np.asarray(pith_full), atol=1e-6, rtol=0.0))
        if not roundtrip_ok:
            errors.append("PITH_SOURCE_CROP_ROUNDTRIP_FAILED")

    metadata = dict(row.get("metadata") or {})
    # Full-section context is absent in this manifest, so do not infer target
    # alignment from a local crop.
    target_domain = str(metadata.get("target_domain") or "TARGET_UNKNOWN")
    if target_domain not in {"TARGET_ALIGNED", "ECCENTRIC_GT", "TARGET_UNKNOWN"}:
        warnings.append("UNKNOWN_TARGET_DOMAIN_RESET")
        target_domain = "TARGET_UNKNOWN"
    contract = {
        "schema": "ArcPith-GT-v4-stage0",
        "line_no": line_no,
        "scale_status": scale_status,
        "coordinate_system": "crop_px_with_source_translation",
        "crop_origin_px": origin,
        "crop_polygon_px": [[0.0, 0.0], [float(width), 0.0],
                             [float(width), float(height)], [0.0, float(height)]],
        "truncation_flags": trunc,
        "target_domain": target_domain,
        "pith_gt_uncertainty": metadata.get("pith_gt_uncertainty"),
        "invariant_checks": {
            "source_crop_roundtrip": roundtrip_ok,
            "pith_outside_not_clipped": True,
            "parent_fragment_order_consistent": len(orders) == len(set(orders)),
            "fixed_transform": True,
        },
        "errors": errors,
        "warnings": warnings,
    }
    normalized = dict(row)
    normalized["curves"] = normalized_curves
    normalized.pop("rings", None)
    normalized["pith_px"] = pith_local
    normalized["crop_origin_px"] = origin
    metadata.update({"stage0_contract": contract})
    observation = {"curves": normalized_curves, "image_size": [width, height],
                   "pith_px": pith_local, "crop_origin_px": origin}
    lineage = {"sample_id": sample_id, "tree_id": str(row["tree_id"]),
               "section_id": str(row["section_id"]), "crop_id": str(row["crop_id"]),
               "parent_ring_ids": sorted({c["ring_id"] for c in normalized_curves}),
               "fragment_ids": [c["fragment_id"] for c in normalized_curves]}
    metadata["observation_hash"] = _json_hash(observation)
    metadata["lineage_hash"] = _json_hash(lineage)
    metadata["input_hash"] = _json_hash({"image": str(row["image"]),
                                          "image_size": [width, height],
                                          "observation_hash": metadata["observation_hash"],
                                          "lineage_hash": metadata["lineage_hash"]})
    normalized["metadata"] = metadata
    return normalized, errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    src, out = Path(args.manifest), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    valid_path = out / "manifest_validated.jsonl"
    fail_path = out / "failures.jsonl"
    summary = {"stage": 0, "schema": "ArcPith-GT-v4-stage0",
               "input": str(src.resolve()), "records": 0, "valid": 0,
               "invalid": 0, "warning_counts": {}, "error_counts": {},
               "trees": [], "sections": 0, "scale_status_counts": {}}
    trees, sections = set(), set()
    with src.open("r", encoding="utf-8") as fin, \
            valid_path.open("w", encoding="utf-8") as fv, \
            fail_path.open("w", encoding="utf-8") as ff:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            summary["records"] += 1
            try:
                row = json.loads(line)
                normalized, errors, warnings = validate_row(row, line_no)
            except Exception as exc:  # malformed input is a data failure
                normalized, errors, warnings = {}, [f"PARSER_ERROR:{type(exc).__name__}"], []
            for code in errors:
                summary["error_counts"][code] = summary["error_counts"].get(code, 0) + 1
            for code in warnings:
                summary["warning_counts"][code] = summary["warning_counts"].get(code, 0) + 1
            if errors:
                summary["invalid"] += 1
                ff.write(json.dumps({"line_no": line_no, "errors": errors,
                                     "warnings": warnings, "sample_id": normalized.get("sample_id")},
                                    ensure_ascii=False) + "\n")
                continue
            summary["valid"] += 1
            trees.add(str(normalized["tree_id"]))
            sections.add(str(normalized["section_id"]))
            status = normalized["metadata"]["stage0_contract"]["scale_status"]
            summary["scale_status_counts"][status] = summary["scale_status_counts"].get(status, 0) + 1
            fv.write(json.dumps(normalized, ensure_ascii=False, allow_nan=False) + "\n")
    summary["trees"] = sorted(trees)
    summary["sections"] = len(sections)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "Stage 0 freezes the normalized data contract. `PIXEL_ONLY` is intentional: "
        "no physical scale was fabricated. Invalid rows are excluded from downstream stages.\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
