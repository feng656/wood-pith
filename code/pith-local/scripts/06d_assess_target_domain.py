#!/usr/bin/env python3
"""Assess radial-centre versus anatomical-pith alignment after GT-blind fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from racpith.config import load_config
from racpith.evaluation.metrics import (
    load_predictions,
    validate_manifest_crop_content,
)
from racpith.provenance import atomic_write_jsonl, read_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute full-section radial-centre/anatomical-pith target bias"
    )
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    args = parser.parse_args()

    frozen = load_config(args.config)
    tolerance = float(frozen.section("target_domain")["alignment_tolerance_norm"])
    all_rows = read_jsonl(args.reference_manifest)
    reference_by_crop: dict[str, dict[str, object]] = {}
    for row in all_rows:
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported full-section reference-manifest schema")
        crop_id = str(row["crop_id"])
        if crop_id in reference_by_crop:
            raise ValueError(f"duplicate full-section reference crop {crop_id}")
        reference_by_crop[crop_id] = row
    rows = all_rows
    if args.split:
        rows = [row for row in rows if row["split"] in set(args.split)]
    if not rows:
        raise ValueError("selected full-section reference scope is empty")
    for row in rows:
        if row.get("analysis_role") != "TARGET_REFERENCE_ONLY":
            raise ValueError(
                "target-domain assessment requires analysis_role=TARGET_REFERENCE_ONLY"
            )
        validate_manifest_crop_content(row)
    predictions, result_index = load_predictions(
        Path(args.predictions),
        result_index_path=args.result_index,
        expected_config_hash=frozen.sha256,
    )
    orphan_results = sorted(set(result_index) - set(reference_by_crop))
    if orphan_results:
        raise ValueError(f"full-section result-index crops absent from manifest: {orphan_results[:10]}")
    reference_manifest_sha256 = sha256_file(args.reference_manifest)
    reference_result_index_sha256 = sha256_file(args.result_index)
    output_rows = []
    seen_sections: set[str] = set()
    for row in rows:
        section_id = str(row["section_id"])
        if section_id in seen_sections:
            raise ValueError(f"duplicate full-section reference for section {section_id}")
        seen_sections.add(section_id)
        prediction = predictions.get(str(row["crop_id"]))
        index_row = result_index.get(str(row["crop_id"]))
        if index_row is not None and any(
            str(index_row.get(field)) != str(row.get(field))
            for field in ("tree_id", "section_id", "split")
        ):
            raise ValueError(f"full-section result-index lineage mismatch for {section_id}")
        bias_norm = None
        bias_px = None
        reason = None
        target_domain = "TARGET_UNKNOWN"
        if prediction is None:
            reason = (
                f"FULL_SECTION_RESULT_STATUS:{index_row.get('status')}:{index_row.get('reason')}"
                if index_row is not None
                else "MISSING_FULL_SECTION_RESULT_INDEX"
            )
        elif prediction.get("config_hash") != frozen.sha256:
            reason = "FULL_SECTION_CONFIG_HASH_MISMATCH"
        elif (
            prediction.get("state") != "POINT"
            or not prediction.get("search_adequate", False)
            or prediction.get("raw_center_norm") is None
        ):
            reason = f"FULL_SECTION_NOT_RELIABLE_POINT:{prediction.get('state')}"
        else:
            width, height = row["crop_size_px"]
            scale = float(row["normalization_scale_px"])
            gt = (
                np.asarray(row["pith_crop_px"], dtype=np.float64)
                - np.asarray([width / 2.0, height / 2.0])
            ) / scale
            estimate = np.asarray(prediction["raw_center_norm"], dtype=np.float64)
            bias_norm = float(np.linalg.norm(estimate - gt))
            bias_px = bias_norm * scale
            target_domain = "TARGET_ALIGNED" if bias_norm <= tolerance else "ECCENTRIC_GT"
        output_rows.append(
            {
                "schema_version": "racpith.target_domain.v1",
                "tree_id": row["tree_id"],
                "section_id": section_id,
                "split": row["split"],
                "reference_crop_id": row["crop_id"],
                "target_domain": target_domain,
                "target_bias_norm": bias_norm,
                "target_bias_px": bias_px,
                "reference_normalization_scale_px": (
                    float(row["normalization_scale_px"])
                    if row.get("normalization_scale_px") is not None
                    else None
                ),
                "alignment_tolerance_norm": tolerance,
                "reason": reason,
                "config_hash": frozen.sha256,
                "reference_result_status": (
                    str(index_row.get("status")) if index_row is not None else "MISSING"
                ),
                "reference_prediction_sha256": (
                    index_row.get("prediction_sha256") if index_row is not None else None
                ),
                "reference_manifest_sha256": reference_manifest_sha256,
                "reference_result_index_sha256": reference_result_index_sha256,
                "gt_used_only_after_reference_fit": True,
            }
        )
    atomic_write_jsonl(args.output, sorted(output_rows, key=lambda row: row["section_id"]))
    counts: dict[str, int] = {}
    for row in output_rows:
        counts[row["target_domain"]] = counts.get(row["target_domain"], 0) + 1
    print(f"target-domain sections={len(output_rows)}; counts={dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
