#!/usr/bin/env python3
"""Adapt the existing cropped annual-ring mask manifest to RAC-Pith inputs.

The grayscale dataset is already cropped and therefore cannot be indexed by the
official full-section UruDendro4 loader.  This adapter keeps the source PNGs
read-only, derives GT-free crop annotations from the manifest's curves, assigns
tree-level development/calibration/sealed splits, and writes explicit pith
position classes for visual inspection.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from racpith.data.split import make_grouped_split
from racpith.provenance import atomic_write_json, atomic_write_jsonl, read_json_object, read_jsonl, sha256_file


SECTION_RE = re.compile(r"^(T(?:0|2|4|6)_B(?:1|2|3)_N\d+_(?:ADAP|A|B|C|D))_s\d+_x\d+_y\d+$")
CLASS_DIRS = {"ON_IMAGE": "图像上", "NEAR_PITH": "近髓", "FAR_PITH": "远髓"}
ANOMALY_GAP_MIN_PX = 50.0
ANOMALY_GAP_RATIO = 4.0
MIN_SPLIT_FRAGMENT_POINTS = 4


def pith_class(row: dict[str, Any]) -> tuple[str, str, float]:
    width, height = (float(value) for value in row["image_size"])
    x, y = (float(value) for value in row["pith_px"])
    inside = 0.0 <= x < width and 0.0 <= y < height
    normalized = math.hypot(x - width / 2.0, y - height / 2.0) / math.hypot(width, height)
    if inside:
        return "ON_IMAGE", "inside", normalized
    if normalized <= 0.5:
        return "NEAR_PITH", "near_outside", normalized
    return "FAR_PITH", "far_outside", normalized


def derived_tree_id(section_id: str) -> str:
    parts = str(section_id).split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot derive biological tree from section_id={section_id!r}")
    return "_".join(parts[:3])


def section_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_section: dict[str, dict[str, Any]] = {}
    for row in rows:
        section_id = str(row["section_id"])
        current = by_section.get(section_id)
        if current is None:
            tree_id = derived_tree_id(section_id)
            treatment, block, tree_number = tree_id.split("_", 2)
            current = {
                "schema_version": "racpith.source_sample.v1",
                "section_id": section_id,
                "tree_id": tree_id,
                "treatment": treatment,
                "block": block,
                "tree_number": tree_number,
                "height": section_id.rsplit("_", 1)[-1],
            }
            by_section[section_id] = current
        if derived_tree_id(section_id) != current["tree_id"]:
            raise ValueError(f"section {section_id} maps to multiple derived tree IDs")
    return [by_section[key] for key in sorted(by_section)]


def split_curve_points(points: list[list[float]]) -> tuple[list[list[list[float]]], int]:
    """Defensively split long annotation jumps before building evidence arcs."""
    if len(points) < 2:
        return [], 0
    pending = [points]
    pieces: list[list[list[float]]] = []
    split_count = 0
    while pending:
        current = pending.pop()
        gaps = [
            math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1]))
            for left, right in zip(current, current[1:])
        ]
        positive = [value for value in gaps if value > 1e-9]
        if not positive:
            pieces.append(current)
            continue
        median_gap = sorted(positive)[len(positive) // 2]
        cuts = [
            index + 1
            for index, gap in enumerate(gaps)
            if gap >= ANOMALY_GAP_MIN_PX and gap >= ANOMALY_GAP_RATIO * max(median_gap, 1e-9)
        ]
        if not cuts:
            pieces.append(current)
            continue
        split_count += len(cuts)
        boundaries = [0, *cuts, len(current)]
        pending.extend(
            current[lo:hi]
            for lo, hi in zip(boundaries, boundaries[1:])
            if hi - lo >= MIN_SPLIT_FRAGMENT_POINTS
        )
    return pieces, split_count


def make_annotation(row: dict[str, Any], split: str, annotation_path: Path) -> dict[str, Any]:
    width, height = (int(value) for value in row["image_size"])
    rings: dict[str, dict[str, Any]] = {}
    total_fragments = short_fragments = visible_parent_rings = 0
    source_discontinuities_split = 0
    for curve in row.get("curves", []):
        ring_id = str(curve["ring_id"])
        points = [[float(point[0]), float(point[1])] for point in curve.get("points_px", [])]
        pieces, split_count = split_curve_points(points)
        source_discontinuities_split += split_count
        for piece in pieces:
            distances = [0.0]
            for left, right in zip(piece, piece[1:]):
                distances.append(distances[-1] + math.hypot(right[0] - left[0], right[1] - left[1]))
            visible_length = distances[-1]
            qualifies = len(piece) >= 4 and visible_length >= 12.0
            total_fragments += 1
            short_fragments += int(not qualifies)
            ring = rings.setdefault(ring_id, {"ring_id": ring_id, "ring_order": int(curve.get("order", 0)), "source_shape_index": int(curve.get("order", 0)), "source_label": ring_id, "source_area_px2": None, "arcs": []})
            ring["arcs"].append({
                "ring_id": ring_id,
                "ring_order": ring["ring_order"],
                "arc_id": f"{row['crop_id']}:{ring_id}:fragment_{len(ring['arcs']):03d}",
                "fragment_index": len(ring["arcs"]),
                "points_crop_px": piece,
                "source_s_px": distances,
                "visible_length_px": visible_length,
                "closed": False,
                "qualifies_for_evidence": qualifies,
                "source_discontinuity_split": bool(split_count),
            })
    visible_parent_rings = sum(any(arc["qualifies_for_evidence"] for arc in ring["arcs"]) for ring in rings.values())
    annotation = {
        "schema_version": "racpith.crop_annotation.v1",
        "crop_id": row["crop_id"],
        "tree_id": derived_tree_id(str(row["section_id"])),
        "section_id": row["section_id"],
        "split": split,
        "source_image_path": str(Path(row["image"]).resolve()),
        "source_annotation_path": str(Path(row.get("metadata", {}).get("source_annotation", annotation_path)).resolve()),
        "source_image_size_px": [width, height],
        "crop_box_source_px": [0, 0, width, height],
        "crop_origin_source_px": [0, 0],
        "crop_size_px": [width, height],
        "normalization_scale_px": math.hypot(width, height),
        "visible_parent_rings": visible_parent_rings,
        "rings": [rings[key] for key in sorted(rings, key=lambda key: (rings[key]["ring_order"], key))],
        "metadata": {
            "coordinate_convention": "Cartesian [x,y]; crop box is half-open",
            "ring_order": "inner_to_outer_zero_based",
            "ring_order_source": "grayscale_manifest_curves",
            "ring_order_reliable": True,
            "curve_crop_operation": "already_cropped_manifest_curve",
            "artificial_crop_boundary_edges_added": False,
            "minimum_fragment_length_px": 12.0,
            "maximum_vertex_spacing_px": None,
            "anomaly_gap_min_px": ANOMALY_GAP_MIN_PX,
            "anomaly_gap_ratio": ANOMALY_GAP_RATIO,
            "source_discontinuities_split": source_discontinuities_split,
            "total_fragment_count": total_fragments,
            "short_fragment_count": short_fragments,
            "source_mask_convention": row.get("metadata", {}).get("grayscale_annotation", {}).get("convention", "dataset_grid"),
        },
    }
    atomic_write_json(annotation_path, annotation)
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="manifest_grayscale.jsonl")
    parser.add_argument("--output", required=True, help="prepared output root")
    parser.add_argument("--class-root", required=True, help="root for 图像上/近髓/远髓 manifests and image links")
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--copy-images", action="store_true", help="copy class images instead of creating hard links")
    args = parser.parse_args()
    source_rows = read_jsonl(args.manifest)
    if not source_rows:
        raise ValueError("grayscale manifest is empty")
    output = Path(args.output).expanduser().resolve()
    class_root = Path(args.class_root).expanduser().resolve()
    crops_root = output / "crops"
    annotation_root = crops_root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    (output / "split").mkdir(parents=True, exist_ok=True)
    (output / "data_index").mkdir(parents=True, exist_ok=True)
    split_config = read_json_object(args.split_config)
    section_rows = section_source_rows(source_rows)
    split_plan = make_grouped_split(
        section_rows,
        split_config["splits"],
        random_seed=int(split_config["random_seed"]),
        search_trials=int(split_config["search_trials"]),
        stratify_keys=tuple(split_config.get("stratify_keys", ["treatment", "block"])),
    )
    split_by_tree = split_plan.split_by_tree
    section_with_split = [{**row, "split": split_by_tree[row["tree_id"]]} for row in section_rows]
    atomic_write_jsonl(output / "split" / "section_manifest.jsonl", section_with_split)
    atomic_write_jsonl(output / "split" / "tree_split.jsonl", split_plan.tree_manifest_rows())
    atomic_write_json(output / "split" / "split_summary.json", dict(split_plan.audit))
    atomic_write_jsonl(output / "data_index" / "source_manifest.jsonl", section_rows)

    crop_rows: list[dict[str, Any]] = []
    class_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    for row in source_rows:
        source_image = Path(str(row["image"])).expanduser().resolve()
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        if len(row.get("image_size", [])) != 2:
            raise ValueError(f"invalid image_size for {row['crop_id']}")
        tree_id = derived_tree_id(str(row["section_id"]))
        split = split_by_tree[tree_id]
        annotation_path = annotation_root / f"{row['crop_id']}.json"
        annotation = make_annotation(row, split, annotation_path)
        category, stratum, distance_norm = pith_class(row)
        width, height = (int(value) for value in row["image_size"])
        eligible = int(annotation["visible_parent_rings"]) >= 3
        crop = {
            "schema_version": "racpith.crop_manifest.v1",
            "crop_id": row["crop_id"],
            "tree_id": tree_id,
            "section_id": row["section_id"],
            "split": split,
            "analysis_role": "LOCAL_MASK_CROP",
            "eligible": eligible,
            "rejection_reasons": [] if eligible else ["TOO_FEW_VISIBLE_PARENT_RINGS"],
            "source_image_path": str(source_image),
            "source_annotation_path": str(annotation_path),
            "crop_image_path": str(source_image),
            "crop_annotation_path": str(annotation_path.resolve()),
            "crop_image_sha256": sha256_file(source_image),
            "crop_annotation_sha256": sha256_file(annotation_path),
            "crop_box_source_px": [0, 0, width, height],
            "crop_origin_source_px": [0, 0],
            "crop_size_px": [width, height],
            "normalization_scale_px": math.hypot(width, height),
            "foreground_fraction": None,
            "visible_parent_rings": annotation["visible_parent_rings"],
            "pith_source_px": list(row["pith_full_px"]),
            "pith_crop_px": list(row["pith_px"]),
            "pith_inside_crop": category == "ON_IMAGE",
            "distance_value": distance_norm,
            "distance_stratum": stratum,
            "pith_class": category,
            "source_manifest": str(Path(args.manifest).expanduser().resolve()),
        }
        crop_rows.append(crop)
        class_rows[category].append(crop)
        class_counts[category] += 1
    crop_rows.sort(key=lambda value: str(value["crop_id"]))
    atomic_write_jsonl(crops_root / "crop_manifest.jsonl", crop_rows)
    # This adapter has no honest full-section reference: the inputs are local crops.
    atomic_write_jsonl(crops_root / "full_section_reference_manifest.jsonl", [])
    for category, dirname in CLASS_DIRS.items():
        target = class_root / dirname
        image_dir = target / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        subset = sorted(class_rows[category], key=lambda value: str(value["crop_id"]))
        atomic_write_jsonl(target / "manifest.jsonl", subset)
        for item in subset:
            source = Path(item["crop_image_path"])
            destination = image_dir / source.name
            if not destination.exists():
                if args.copy_images:
                    shutil.copy2(source, destination)
                else:
                    try:
                        destination.hardlink_to(source)
                    except OSError:
                        shutil.copy2(source, destination)
        atomic_write_json(target / "summary.json", {"class": category, "directory": dirname, "count": len(subset), "split_counts": dict(Counter(item["split"] for item in subset)), "distance_stratum_counts": dict(Counter(item["distance_stratum"] for item in subset))})
    atomic_write_json(output / "mask_prepare_audit.json", {"schema_version": "racpith.mask_prepare_audit.v1", "source_manifest": str(Path(args.manifest).expanduser().resolve()), "dataset_root": str(Path(source_rows[0]["image"]).parent.resolve()), "crops": len(crop_rows), "eligible_crops": sum(bool(row["eligible"]) for row in crop_rows), "trees": sorted({str(row["tree_id"]) for row in crop_rows}), "sections": sorted({str(row["section_id"]) for row in crop_rows}), "class_counts": dict(class_counts), "class_root": str(class_root), "source_read_only": True, "gt_in_annotation": False, "gt_kept_in_manifest_only": True})
    print(json.dumps({"crops": len(crop_rows), "eligible": sum(bool(row["eligible"]) for row in crop_rows), "trees": len(split_by_tree), "classes": dict(class_counts), "output": str(output), "class_root": str(class_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
