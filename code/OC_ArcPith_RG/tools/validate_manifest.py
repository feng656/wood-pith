#!/usr/bin/env python3
"""Validate grouping, geometry, order, provenance, and image paths for Step A."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def resolve_image(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--biological-tree-field", default="tree_id")
    parser.add_argument("--require-order", action="store_true")
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--require-coordinate-provenance", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = []
    warnings = []
    sections = set()
    trees = set()
    curve_count = 0
    explicit_order_count = 0
    multifragment_records = 0
    seen_samples = set()
    for index, row in enumerate(rows):
        tag = str(row.get("sample_id", f"row:{index + 1}"))
        for key in ("sample_id", args.biological_tree_field, "image_size"):
            if key not in row:
                errors.append([tag, f"missing_{key}"])
        if tag in seen_samples:
            errors.append([tag, "duplicate_sample_id"])
        seen_samples.add(tag)
        tree = row.get(args.biological_tree_field)
        if tree is not None:
            trees.add(str(tree))
        section = row.get("section_id")
        if section is None:
            warnings.append([tag, "missing_section_id"])
        else:
            sections.add(str(section))
        rings = row.get("rings", row.get("curves", []))
        if not rings:
            errors.append([tag, "no_rings"])
        order_by_ring = defaultdict(set)
        fragments_by_ring = defaultdict(int)
        for ring_index, ring in enumerate(rings):
            curve_count += 1
            ring_id = str(ring.get("ring_id", ring_index))
            fragments_by_ring[ring_id] += 1
            points = ring.get("points_px", ring.get("points"))
            try:
                points_array = np.asarray(points, dtype=float)
                if points_array.ndim != 2 or points_array.shape[1] != 2 or len(points_array) < 4 or not np.all(np.isfinite(points_array)):
                    errors.append([tag, ring_id, "invalid_ring_points"])
            except Exception:
                errors.append([tag, ring_id, "invalid_ring_points"])
            order = ring.get("order")
            if order is not None:
                explicit_order_count += 1
                order_by_ring[ring_id].add(order)
            elif args.require_order:
                errors.append([tag, ring_id, "missing_explicit_parent_ring_order"])
        if any(count > 1 for count in fragments_by_ring.values()):
            multifragment_records += 1
        for ring_id, values in order_by_ring.items():
            if len(values) != 1:
                errors.append([tag, ring_id, "conflicting_fragment_orders"])
        parent_orders = [next(iter(values)) for values in order_by_ring.values() if len(values) == 1]
        if len(parent_orders) != len(set(parent_orders)):
            errors.append([tag, "duplicate_order_across_parent_rings"])
        pith = row.get("pith_px")
        if pith is not None and (not isinstance(pith, list) or len(pith) != 2):
            errors.append([tag, "invalid_pith_px"])
        if args.require_coordinate_provenance:
            full = row.get("pith_full_px")
            origin = row.get("crop_origin_px")
            if pith is None or full is None or origin is None:
                errors.append([tag, "missing_coordinate_provenance"])
            elif not np.allclose(np.asarray(full, float) - np.asarray(origin, float), np.asarray(pith, float), atol=1e-6, rtol=0):
                errors.append([tag, "pith_coordinate_provenance_mismatch"])
        if args.require_images:
            image = row.get("image")
            if not image or not resolve_image(manifest, str(image)).is_file():
                errors.append([tag, "missing_image"])
    report = {
        "manifest": str(manifest),
        "records": len(rows),
        "biological_tree_groups": len(trees),
        "tree_ids": sorted(trees),
        "sections": len(sections),
        "curves": curve_count,
        "curves_with_explicit_order": explicit_order_count,
        "records_with_multifragment_parent_rings": multifragment_records,
        "errors": errors,
        "warnings": warnings,
        "pass": not errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
