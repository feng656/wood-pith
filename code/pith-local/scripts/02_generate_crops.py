#!/usr/bin/env python3
"""Materialise deterministic, overlapping UruDendro4 crops outside the source tree.

Ground-truth pith coordinates are written only to the crop manifest.  The crop
annotation consumed by the evidence builder contains curve geometry and lineage,
but no pith coordinate.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import cv2

from racpith.config import load_config, resolve_runtime_paths, stable_hash
from racpith.data.crop import (
    CropBox,
    build_crop_annotation,
    crop_pith_coordinates,
    materialize_prepared_crop,
    prepare_section_crops,
)
from racpith.data.urudendro4 import load_urudendro4
from racpith.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    is_sha256,
    read_json_object,
    read_jsonl,
    sha256_file,
)


def _unique_by_section(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        section_id = str(row["section_id"])
        if section_id in result:
            raise ValueError(f"duplicate section_id in section manifest: {section_id}")
        result[section_id] = row
    return result


def _with_split_lineage(
    row: Mapping[str, Any], section_row: Mapping[str, Any]
) -> dict[str, Any]:
    value = dict(row)
    value["inner_fold"] = section_row.get("inner_fold")
    return value


def _cached_pair(row: Mapping[str, Any], field: str) -> tuple[float, float]:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"cached crop {field} must contain exactly two values")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in pair):
        raise ValueError(f"cached crop {field} must be finite")
    return pair


def _validate_cached_coordinate_row(
    row: Mapping[str, Any],
    *,
    image_root: Path | None,
    annotation_root: Path,
) -> str:
    if row.get("schema_version") != "racpith.crop_manifest.v1":
        raise ValueError("cached crop manifest has an unsupported schema")
    crop_id = str(row.get("crop_id", ""))
    if not crop_id:
        raise ValueError("cached crop manifest row lacks crop_id")
    origin = _cached_pair(row, "crop_origin_source_px")
    if any(value != int(value) for value in origin):
        raise ValueError(f"cached crop {crop_id} has a non-integral crop origin")
    size = row.get("crop_size_px")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError(f"cached crop {crop_id} has invalid crop_size_px")
    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError(f"cached crop {crop_id} has non-positive dimensions")
    box = CropBox(
        int(origin[0]), int(origin[1]), int(origin[0]) + width, int(origin[1]) + height
    )
    pith_source = _cached_pair(row, "pith_source_px")
    expected_pith_crop, expected_inside = crop_pith_coordinates(pith_source, box)
    actual_pith_crop = _cached_pair(row, "pith_crop_px")
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
        for actual, expected in zip(actual_pith_crop, expected_pith_crop)
    ):
        raise ValueError(f"cached crop {crop_id} has inconsistent pith coordinates")
    if (
        not isinstance(row.get("pith_inside_crop"), bool)
        or row["pith_inside_crop"] != expected_inside
    ):
        raise ValueError(f"cached crop {crop_id} has inconsistent pith_inside_crop")
    scale = float(row.get("normalization_scale_px", float("nan")))
    if not math.isclose(scale, math.hypot(width, height), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"cached crop {crop_id} has inconsistent normalization scale")
    annotation_path = Path(str(row.get("crop_annotation_path", ""))).resolve()
    if not annotation_path.is_relative_to(annotation_root) or not annotation_path.is_file():
        raise ValueError(f"cached crop annotation is unavailable for {crop_id}")
    if image_root is not None:
        image_path = Path(str(row.get("crop_image_path", ""))).resolve()
        if not image_path.is_relative_to(image_root) or not image_path.is_file():
            raise ValueError(f"cached crop image is unavailable for {crop_id}")
    for field in ("crop_image_sha256", "crop_annotation_sha256"):
        if not is_sha256(row.get(field)):
            raise ValueError(f"cached crop {crop_id} has invalid {field}")
    return crop_id


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _load_reusable_crop_cache(
    output: Path,
    *,
    dataset_root: Path,
    section_manifest_path: Path,
    crop_config_path: Path,
) -> dict[str, Any] | None:
    """Return a validated completed local crop cache, or ``None`` to rebuild it."""

    audit_path = output / "crop_generation_audit.json"
    manifest_paths = {
        "crop_manifest_sha256": output / "crop_manifest.jsonl",
        "crop_candidate_manifest_sha256": output / "crop_candidate_manifest.jsonl",
        "full_section_reference_manifest_sha256": (
            output / "full_section_reference_manifest.jsonl"
        ),
    }
    section_audit_path = output / "crop_section_audit.jsonl"
    if (
        not audit_path.is_file()
        or not section_audit_path.is_file()
        or any(not path.is_file() for path in manifest_paths.values())
    ):
        return None
    try:
        audit = read_json_object(audit_path)
        if audit.get("schema_version") != "racpith.crop_generation_audit.v1":
            return None
        expected = {
            "dataset_root": str(dataset_root),
            "section_manifest": str(section_manifest_path),
            "section_manifest_sha256": sha256_file(section_manifest_path),
            "crop_config": str(crop_config_path),
            "crop_config_sha256": sha256_file(crop_config_path),
            "derived_crop_root": str((output / "derived_crops").resolve()),
        }
        if any(audit.get(field) != value for field, value in expected.items()):
            return None
        for hash_field, path in manifest_paths.items():
            if audit.get(hash_field) != sha256_file(path):
                return None
        selected_rows = read_jsonl(manifest_paths["crop_manifest_sha256"])
        reference_rows = read_jsonl(
            manifest_paths["full_section_reference_manifest_sha256"]
        )
        if len(selected_rows) != int(audit["selected_crops"]):
            return None
        if len(reference_rows) != int(audit["full_section_references"]):
            return None
        if _count_jsonl_rows(manifest_paths["crop_candidate_manifest_sha256"]) != int(
            audit["candidates"]
        ):
            return None
        if _count_jsonl_rows(section_audit_path) != int(audit["sections"]):
            return None
        derived_root = (output / "derived_crops").resolve()
        selected_ids = {
            _validate_cached_coordinate_row(
                row, image_root=derived_root, annotation_root=derived_root
            )
            for row in selected_rows
        }
        if len(selected_ids) != len(selected_rows):
            return None
        reference_root = (output / "full_section_references").resolve()
        reference_ids = {
            _validate_cached_coordinate_row(
                row, image_root=None, annotation_root=reference_root
            )
            for row in reference_rows
        }
        if len(reference_ids) != len(reference_rows):
            return None
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic overlapping crops without modifying UruDendro4"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "racpith_v1.json"),
        help="RAC-Pith config containing paths.dataset_root",
    )
    parser.add_argument(
        "--dataset-root",
        help="explicit override; defaults to paths.dataset_root in --config",
    )
    parser.add_argument("--section-manifest", required=True)
    parser.add_argument("--crop-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pith-order", choices=["auto", "xy", "yx"], default="auto")
    parser.add_argument("--allow-nonofficial-subset", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="reuse a validated completed local crop cache, rebuilding it only when stale",
    )
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frozen = load_config(args.config)
    configured = resolve_runtime_paths(
        frozen,
        project_root=Path(__file__).resolve().parents[1],
    )
    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else configured.dataset_root
    )

    section_manifest_path = Path(args.section_manifest).resolve()
    crop_config_path = Path(args.crop_config).resolve()
    section_rows = read_jsonl(section_manifest_path)
    section_by_id = _unique_by_section(section_rows)
    crop_config = read_json_object(crop_config_path)
    if crop_config.get("schema_version") != "racpith.crops.v1":
        raise ValueError("unsupported crop configuration schema")

    output = Path(args.output).expanduser().resolve()
    derived_root = output / "derived_crops"
    if args.resume:
        cached_audit = _load_reusable_crop_cache(
            output,
            dataset_root=dataset_root,
            section_manifest_path=section_manifest_path,
            crop_config_path=crop_config_path,
        )
        if cached_audit is not None:
            print(
                f"reused {cached_audit['selected_crops']} cached crops from "
                f"{cached_audit['sections']} sections; pith coordinates loaded from manifests"
            )
            return

    index = load_urudendro4(
        dataset_root,
        pith_order=args.pith_order,
        require_official_complete=not args.allow_nonofficial_subset,
    )
    sample_by_id = {sample.section_id: sample for sample in index.samples}
    if set(sample_by_id) != set(section_by_id):
        missing = sorted(set(section_by_id) - set(sample_by_id))
        unexpected = sorted(set(sample_by_id) - set(section_by_id))
        raise ValueError(
            "dataset/section-manifest mismatch: "
            f"missing_in_dataset={missing[:10]}, absent_from_manifest={unexpected[:10]}"
        )

    overwrite = args.overwrite or args.resume
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    section_audits: list[dict[str, Any]] = []
    for section_id in sorted(section_by_id):
        section_row = section_by_id[section_id]
        sample = sample_by_id[section_id]
        if str(section_row["tree_id"]) != sample.tree_id:
            raise ValueError(f"tree_id mismatch for section {section_id}")
        source_image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if source_image is None:
            raise FileNotFoundError(f"cannot read source image {sample.image_path}")
        plan = prepare_section_crops(
            sample,
            split=str(section_row["split"]),
            source_image=source_image,
            output_root=derived_root,
            crop_config=crop_config,
        )
        for prepared in plan.selected:
            materialized = materialize_prepared_crop(
                prepared, source_image, overwrite=overwrite
            )
            materialized["crop_image_sha256"] = sha256_file(
                materialized["crop_image_path"]
            )
            materialized["crop_annotation_sha256"] = sha256_file(
                materialized["crop_annotation_path"]
            )
            selected_rows.append(_with_split_lineage(materialized, section_row))
        candidate_rows.extend(
            _with_split_lineage(row, section_row)
            for row in plan.manifest_rows(selected_only=False)
        )
        section_audits.append(plan.audit_summary())
        reference_crop_id = f"{section_id}__full_section_reference"
        reference_annotation_path = (
            output / "full_section_references" / "annotations" / f"{reference_crop_id}.json"
        ).resolve()
        if reference_annotation_path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite full-section reference {reference_annotation_path}"
            )
        reference_annotation = build_crop_annotation(
            sample,
            CropBox(0, 0, sample.width, sample.height),
            crop_id=reference_crop_id,
            split=str(section_row["split"]),
            minimum_fragment_length_px=float(crop_config["minimum_fragment_length_px"]),
            maximum_vertex_spacing_px=float(
                crop_config.get("maximum_vertex_spacing_px", 4.0)
            ),
        )
        atomic_write_json(reference_annotation_path, reference_annotation)
        reference_visible_rings = int(reference_annotation["visible_parent_rings"])
        reference_eligible = reference_visible_rings >= int(
            crop_config["minimum_visible_parent_rings"]
        )
        reference_rows.append(
            {
                "schema_version": "racpith.crop_manifest.v1",
                "crop_id": reference_crop_id,
                "tree_id": sample.tree_id,
                "section_id": sample.section_id,
                "split": section_row["split"],
                "inner_fold": section_row.get("inner_fold"),
                "analysis_role": "TARGET_REFERENCE_ONLY",
                "eligible": reference_eligible,
                "selected": reference_eligible,
                "rejection_reasons": (
                    [] if reference_eligible else ["TOO_FEW_VISIBLE_PARENT_RINGS"]
                ),
                "source_image_path": str(sample.image_path.resolve()),
                "source_annotation_path": str(sample.annotation_path.resolve()),
                "crop_image_path": str(sample.image_path.resolve()),
                "crop_annotation_path": str(reference_annotation_path),
                "crop_image_sha256": sha256_file(sample.image_path),
                "crop_annotation_sha256": sha256_file(reference_annotation_path),
                "crop_box_source_px": [0, 0, sample.width, sample.height],
                "crop_origin_source_px": [0, 0],
                "crop_size_px": [sample.width, sample.height],
                "normalization_scale_px": float(
                    (sample.width**2 + sample.height**2) ** 0.5
                ),
                "foreground_fraction": None,
                "visible_parent_rings": reference_visible_rings,
                "pith_source_px": list(sample.pith_source_px),
                "pith_crop_px": list(sample.pith_source_px),
                "pith_inside_crop": True,
                "distance_value": 0.0,
                "distance_stratum": "full_section_reference",
            }
        )

    selected_rows.sort(key=lambda row: str(row["crop_id"]))
    candidate_rows.sort(key=lambda row: str(row["crop_id"]))
    reference_rows.sort(key=lambda row: str(row["crop_id"]))
    atomic_write_jsonl(output / "crop_manifest.jsonl", selected_rows)
    atomic_write_jsonl(output / "crop_candidate_manifest.jsonl", candidate_rows)
    atomic_write_jsonl(
        output / "full_section_reference_manifest.jsonl", reference_rows
    )
    atomic_write_jsonl(output / "crop_section_audit.jsonl", section_audits)
    audit = {
        "schema_version": "racpith.crop_generation_audit.v1",
        "dataset_root": str(dataset_root),
        "runtime_config_path": str(frozen.source),
        "runtime_config_hash": frozen.sha256,
        "paths_config_hash": stable_hash(dict(frozen.section("paths"))),
        "configured_dataset_root": str(configured.dataset_root),
        "dataset_root_override_used": dataset_root != configured.dataset_root,
        "configured_output_root": str(configured.output_root),
        "configured_prepared_root": str(configured.prepared_root),
        "prepared_root_override_used": output.parent != configured.prepared_root,
        "section_manifest": str(section_manifest_path),
        "section_manifest_sha256": sha256_file(section_manifest_path),
        "crop_config": str(crop_config_path),
        "crop_config_sha256": sha256_file(crop_config_path),
        "crop_manifest_sha256": sha256_file(output / "crop_manifest.jsonl"),
        "crop_candidate_manifest_sha256": sha256_file(
            output / "crop_candidate_manifest.jsonl"
        ),
        "full_section_reference_manifest_sha256": sha256_file(
            output / "full_section_reference_manifest.jsonl"
        ),
        "derived_crop_root": str(derived_root),
        "pith_order": index.pith_order,
        "requested_pith_order": args.pith_order,
        "sections": len(section_rows),
        "trees": len({str(row["tree_id"]) for row in section_rows}),
        "candidates": len(candidate_rows),
        "eligible_candidates": sum(bool(row["eligible"]) for row in candidate_rows),
        "selected_crops": len(selected_rows),
        "full_section_references": len(reference_rows),
        "selected_by_split": {
            split: sum(str(row["split"]) == split for row in selected_rows)
            for split in sorted({str(row["split"]) for row in selected_rows})
        },
        "source_dataset_modified": False,
        "gt_present_in_crop_annotation": False,
        "gt_present_only_in_manifest": True,
        "pith_coordinate_cache": {
            "manifest": str((output / "crop_manifest.jsonl").resolve()),
            "fields": ["pith_source_px", "pith_crop_px", "pith_inside_crop"],
            "coordinate_convention": "[x,y]=[column,row]",
        },
    }
    atomic_write_json(output / "crop_generation_audit.json", audit)
    print(
        f"materialised {len(selected_rows)} selected crops from {len(section_rows)} sections; "
        f"candidate denominator={len(candidate_rows)}"
    )


if __name__ == "__main__":
    main()
