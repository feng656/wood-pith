#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from racpith.config import load_config
from racpith.evaluation.contribution import LABELS, load_contribution_records
from racpith.evaluation.metrics import validate_manifest_crop_content
from racpith.provenance import atomic_write_json, is_sha256, read_jsonl, sha256_file
from racpith.visualization import render_contribution_overlay


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_visualization_index(
    output: Path,
    *,
    config_hash: str,
    splits: list[str],
    max_overlays: int,
    source_result_index_hashes: list[str],
    input_hashes: dict[str, str],
    cohort_paths: list[Path],
    overlay_paths: dict[str, Path],
    manifest_crops: int,
    trees: int,
    sections: int,
    status_counts: dict[str, int],
    group_records: int,
    overlay_eligible_crops: int,
) -> None:
    registered = {path.resolve() for path in cohort_paths} | {
        path.resolve() for path in overlay_paths.values()
    }
    on_disk = {path.resolve() for path in output.rglob("*.png")}
    stale = sorted(str(path) for path in on_disk - registered)
    if stale:
        raise ValueError(
            "unregistered/stale contribution visualization artifacts: "
            f"{stale[:10]}"
        )
    for path in registered:
        if not path.is_file():
            raise FileNotFoundError(path)
    atomic_write_json(
        output / "contribution_visualization_index.json",
        {
            "schema_version": "racpith.contribution_visualization_index.v1",
            "config_hash": config_hash,
            "splits": splits,
            "max_overlays": max_overlays,
            "selection_policy": "crop_id order over registered primary-subarc records",
            "manifest_crops": manifest_crops,
            "trees": trees,
            "sections": sections,
            "status_counts": status_counts,
            "crop_index_records": manifest_crops,
            "group_records": group_records,
            "overlay_eligible_crops": overlay_eligible_crops,
            "rendered_overlays": len(overlay_paths),
            "source_result_index_sha256": source_result_index_hashes,
            "input_artifact_hashes": input_hashes,
            "cohort_artifacts": {
                str(path.relative_to(output)): sha256_file(path)
                for path in cohort_paths
            },
            "overlay_artifacts": {
                crop_id: {
                    "path": str(path.relative_to(output)),
                    "sha256": sha256_file(path),
                }
                for crop_id, path in sorted(overlay_paths.items())
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render contribution cohort plots and arc overlays")
    parser.add_argument("--contributions", required=True)
    parser.add_argument("--contribution-index", required=True)
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--max-overlays", type=int, default=80)
    args = parser.parse_args()
    if args.max_overlays < 0:
        raise ValueError("--max-overlays must be non-negative")

    frozen = load_config(args.config)
    records = load_contribution_records(
        args.contributions,
        args.contribution_index,
        expected_config_hash=frozen.sha256,
    )
    all_index = read_jsonl(args.contribution_index)
    manifest: dict[str, dict[str, object]] = {}
    for row in read_jsonl(args.crop_manifest):
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported crop-manifest schema")
        crop_id = str(row["crop_id"])
        if crop_id in manifest:
            raise ValueError(f"duplicate crop-manifest row for {crop_id}")
        manifest[crop_id] = row
    selected_manifest_ids = {
        crop_id
        for crop_id, row in manifest.items()
        if not args.split or str(row["split"]) in set(args.split)
    }
    if not selected_manifest_ids:
        raise ValueError("selected crop-manifest scope is empty")
    for crop_id in sorted(selected_manifest_ids):
        validate_manifest_crop_content(manifest[crop_id])
    selected_index = []
    seen_index: set[str] = set()
    for row in all_index:
        if row.get("schema_version") != "racpith.contribution_index.v1":
            raise ValueError("unsupported contribution-index schema")
        crop_id = str(row["crop_id"])
        source = manifest.get(crop_id)
        if source is None:
            raise ValueError(f"orphan contribution-index crop {crop_id}")
        if crop_id in seen_index:
            raise ValueError(f"duplicate contribution-index crop {crop_id}")
        seen_index.add(crop_id)
        if any(
            str(row.get(field)) != str(source.get(field))
            for field in ("tree_id", "section_id", "split")
        ):
            raise ValueError(f"contribution-index/manifest lineage mismatch for {crop_id}")
        if crop_id in selected_manifest_ids:
            selected_index.append(row)
    missing_index = sorted(
        selected_manifest_ids - {str(row["crop_id"]) for row in selected_index}
    )
    if missing_index:
        raise ValueError(
            f"selected manifest crops lack contribution-index rows: {missing_index[:10]}"
        )
    if not records.empty:
        records = records[
            records["crop_id"].astype(str).isin(selected_manifest_ids)
        ].copy()
    source_hash_values = [
        row.get("source_result_index_sha256") for row in selected_index
    ]
    if any(not is_sha256(value) for value in source_hash_values):
        raise ValueError(
            "selected contribution index must bind every crop to a localization result index"
        )
    source_result_index_hashes = sorted(set(source_hash_values))
    if len(source_result_index_hashes) != 1:
        raise ValueError(
            "selected contribution index must bind exactly one localization result index"
        )
    successful_groups = [
        int(row.get("groups", -1))
        for row in selected_index
        if row.get("status") in {"PASS", "RESUMED"}
    ]
    if any(value < 0 for value in successful_groups) or sum(successful_groups) != len(
        records
    ):
        raise ValueError(
            "selected contribution records do not match the index group denominator"
        )
    index = [
        row for row in selected_index if row["status"] in {"PASS", "RESUMED"}
    ]
    status_counts: dict[str, int] = {}
    for row in selected_index:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest_trees = len(
        {str(manifest[crop_id]["tree_id"]) for crop_id in selected_manifest_ids}
    )
    manifest_sections = len(
        {str(manifest[crop_id]["section_id"]) for crop_id in selected_manifest_ids}
    )
    output = Path(args.output)
    cohort_paths = [
        output / "cohort" / "formal_sign_distribution.png",
        output / "cohort" / "crossfit_vs_exact.png",
    ]
    overlay_paths: dict[str, Path] = {}
    input_hashes = {
        "config_file": sha256_file(args.config),
        "crop_manifest": sha256_file(args.crop_manifest),
        "contribution_index": sha256_file(args.contribution_index),
    }
    partition = "subarc:f={:.8g}:phase={:.8g}".format(
        float(frozen.section("contribution")["primary_subarc_fraction"]), 0.0
    )
    overlay_eligible_crops = sum(
        bool((records[records["crop_id"] == str(row["crop_id"])]["partition_id"] == partition).any())
        for row in index
    )

    if records.empty:
        for path, title in (
            (cohort_paths[0], "No per-group contribution records"),
            (cohort_paths[1], "No cross-fitted/exact contribution pairs"),
        ):
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.text(0.5, 0.5, title, ha="center", va="center")
            axis.axis("off")
            _save(figure, path)
        _write_visualization_index(
            output,
            config_hash=frozen.sha256,
            splits=sorted({str(manifest[crop_id]["split"]) for crop_id in selected_manifest_ids}),
            max_overlays=args.max_overlays,
            source_result_index_hashes=source_result_index_hashes,
            input_hashes=input_hashes,
            cohort_paths=cohort_paths,
            overlay_paths=overlay_paths,
            manifest_crops=len(selected_manifest_ids),
            trees=manifest_trees,
            sections=manifest_sections,
            status_counts=status_counts,
            group_records=len(records),
            overlay_eligible_crops=overlay_eligible_crops,
        )
        print("no per-group contribution records; rendered empty-state cohort figures")
        return

    distribution = (
        records[records["gt_label"].isin((*LABELS, "UNCERTAIN"))]
        .groupby(["level", "gt_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["HARMFUL_GT", "NEUTRAL", "BENEFICIAL_GT", "UNCERTAIN"], fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    if not distribution.empty:
        distribution.plot.bar(stacked=True, ax=axis, color=["#D55E00", "#888888", "#009E73", "#E69F00"])
    axis.set_ylabel("registered deletion groups")
    axis.set_title("Exact contribution labels (formal interval labels only)")
    _save(figure, cohort_paths[0])

    comparable = records[records["contrib_cf"].notna() & records["contrib_gt_norm"].notna()]
    figure, axis = plt.subplots(figsize=(6, 6))
    for level, frame in comparable.groupby("level"):
        axis.scatter(frame["contrib_gt_norm"], frame["contrib_cf"], s=10, alpha=0.45, label=level)
    axis.axhline(0.0, color="black", lw=0.7)
    axis.axvline(0.0, color="black", lw=0.7)
    axis.set_xlabel("exact GT contribution / D_FOV")
    axis.set_ylabel("cross-fitted witness contribution")
    axis.set_title("GT-blind witness signal versus exact contribution")
    if not comparable.empty:
        axis.legend()
    _save(figure, cohort_paths[1])

    rendered = 0
    for row in sorted(index, key=lambda value: str(value["crop_id"])):
        if rendered >= args.max_overlays:
            break
        crop_id = str(row["crop_id"])
        source = manifest.get(crop_id)
        record_path = row.get("record_path")
        if source is None or record_path is None:
            continue
        crop_records = records[records["crop_id"] == crop_id]
        if not (crop_records["partition_id"] == partition).any():
            continue
        overlay_path = output / "overlays" / f"{crop_id}.png"
        render_contribution_overlay(
            source["crop_image_path"],
            source["crop_annotation_path"],
            record_path,
            overlay_path,
            partition,
        )
        overlay_paths[crop_id] = overlay_path
        rendered += 1
    _write_visualization_index(
        output,
        config_hash=frozen.sha256,
        splits=sorted({str(manifest[crop_id]["split"]) for crop_id in selected_manifest_ids}),
        max_overlays=args.max_overlays,
        source_result_index_hashes=source_result_index_hashes,
        input_hashes=input_hashes,
        cohort_paths=cohort_paths,
        overlay_paths=overlay_paths,
        manifest_crops=len(selected_manifest_ids),
        trees=manifest_trees,
        sections=manifest_sections,
        status_counts=status_counts,
        group_records=len(records),
        overlay_eligible_crops=overlay_eligible_crops,
    )
    print(f"rendered contribution cohort plots and {rendered} primary-subarc overlays")


if __name__ == "__main__":
    main()
