#!/usr/bin/env python3
"""Adapt the audited no-background crop manifest for the adjusted package.

The original 102 closed-ring annotations were audited before this adapter was
introduced: numeric labels are unique and polygon area increases strictly with
the label in every section. The adapter independently rechecks that invariant
and only then writes the numeric label as the explicit inner-to-outer order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def polygon_area(points) -> float:
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        raise ValueError("polygon needs at least three points")
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def audit_full_annotation_orders(directory: Path) -> dict[str, dict[str, int]]:
    orders = {}
    for path in sorted(directory.glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        labeled_areas = []
        for index, shape in enumerate(annotation.get("shapes", [])):
            points = shape.get("points", [])
            if len(points) < 3:
                continue
            label = str(shape.get("label", index))
            if not label.lstrip("-").isdigit():
                raise ValueError(f"{path.name}: non-numeric ring label {label!r}")
            labeled_areas.append((label, polygon_area(points)))
        if not labeled_areas:
            raise ValueError(f"{path.name}: no valid closed rings")
        if len({label for label, _ in labeled_areas}) != len(labeled_areas):
            raise ValueError(f"{path.name}: duplicate parent-ring labels")
        labeled_areas.sort(key=lambda item: int(item[0]))
        areas = [area for _, area in labeled_areas]
        if not all(right > left for left, right in zip(areas, areas[1:])):
            raise ValueError(f"{path.name}: numeric labels do not strictly follow polygon area")
        orders[path.stem] = {label: rank for rank, (label, _) in enumerate(labeled_areas)}
    if not orders:
        raise RuntimeError(f"no annotations found in {directory}")
    return orders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--full-annotations-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    annotations = Path(args.full_annotations_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    order_by_section = audit_full_annotation_orders(annotations)
    records = trees = curves = 0
    tree_ids = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            section = str(row.get("section_id") or row.get("metadata", {}).get("source_image_id") or "")
            if section not in order_by_section:
                raise ValueError(f"line {line_number}: no audited full annotation for {section!r}")
            parent_orders = order_by_section[section]
            parent_order_seen = {}
            for curve in row.get("rings", row.get("curves", [])):
                ring_id = str(curve.get("ring_id"))
                if ring_id not in parent_orders:
                    raise ValueError(f"line {line_number}: ring {ring_id!r} absent from {section}")
                curve["order"] = int(parent_orders[ring_id])
                parent_order_seen[ring_id] = int(parent_orders[ring_id])
                curves += 1
            metadata = row.setdefault("metadata", {})
            metadata["ring_topology_source"] = "full_closed_polygon_area_order"
            metadata["ring_order_audit"] = {
                "rule": "numeric labels are unique and full-polygon area strictly increases with label",
                "audited_sections": len(order_by_section),
                "parent_ring_order": parent_order_seen,
                "full_parent_ring_order": [
                    ring_id for ring_id, _ in sorted(parent_orders.items(), key=lambda item: item[1])
                ],
            }
            dst.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            records += 1
            tree_ids.add(str(row["tree_id"]))
    print(json.dumps({
        "records": records,
        "trees": len(tree_ids),
        "tree_ids": sorted(tree_ids),
        "audited_sections": len(order_by_section),
        "curves_with_explicit_order": curves,
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
