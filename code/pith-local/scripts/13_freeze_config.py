#!/usr/bin/env python3
"""Create a new immutable-by-content development-frozen configuration.

This command does not tune thresholds.  It records the already completed
development decision and split identities, and refuses to overwrite the source
configuration.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from racpith.config import load_config, resolve_runtime_paths, stable_hash
from racpith.provenance import (
    atomic_write_json,
    is_sha256,
    read_json_object,
    read_jsonl,
    sha256_file,
    sha256_source_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an already-selected RAC-Pith configuration")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--section-manifest", required=True)
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--crop-audit", required=True)
    parser.add_argument("--decision-record", required=True)
    parser.add_argument("--development-run-id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--development-split", default="train")
    parser.add_argument("--sealed-split", default="sealed_test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--acknowledge-no-sealed-results-used", action="store_true")
    args = parser.parse_args()

    if not args.acknowledge_no_sealed_results_used:
        raise ValueError(
            "freezing requires --acknowledge-no-sealed-results-used after verifying that "
            "no sealed-test result influenced configuration choices"
        )
    development_run_id = args.development_run_id.strip()
    code_revision = args.code_revision.strip()
    development_split = args.development_split.strip()
    sealed_split = args.sealed_split.strip()
    if not development_run_id or not code_revision:
        raise ValueError("development run id and immutable code revision must be non-empty")
    if not development_split or not sealed_split or development_split == sealed_split:
        raise ValueError("development and sealed split names must be non-empty and distinct")
    source = Path(args.input_config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output:
        raise ValueError("refusing to overwrite the provisional/source configuration")
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing frozen configuration: {output}"
        )
    decision = Path(args.decision_record).expanduser().resolve()
    section_manifest = Path(args.section_manifest).expanduser().resolve()
    crop_manifest = Path(args.crop_manifest).expanduser().resolve()
    reference_manifest = Path(args.reference_manifest).expanduser().resolve()
    crop_audit_path = Path(args.crop_audit).expanduser().resolve()
    if not source.is_file() or not all(
        path.is_file()
        for path in (
            decision,
            section_manifest,
            crop_manifest,
            reference_manifest,
            crop_audit_path,
        )
    ):
        raise FileNotFoundError(
            "source config, decision record, section manifest, crop/reference manifests, "
            "and crop audit must already exist"
        )
    if decision.stat().st_size == 0:
        raise ValueError("development decision record must not be empty")
    source_config = load_config(source)
    configured_paths = resolve_runtime_paths(
        source_config,
        project_root=Path(__file__).resolve().parents[1],
    )
    config = dict(source_config.data)
    if config.get("calibration_status") == "DEVELOPMENT_FROZEN":
        raise ValueError("input configuration is already development-frozen")
    if "freeze" in config:
        raise ValueError("input configuration already contains a freeze record")

    rows = read_jsonl(section_manifest)
    crop_audit = read_json_object(crop_audit_path)
    if crop_audit.get("schema_version") != "racpith.crop_generation_audit.v1":
        raise ValueError("unsupported crop audit schema")
    paths_config_hash = stable_hash(dict(source_config.section("paths")))
    if crop_audit.get("paths_config_hash") != paths_config_hash:
        raise ValueError("crop audit was prepared with another paths configuration")
    if crop_audit.get("dataset_root") != str(configured_paths.dataset_root):
        raise ValueError("crop audit dataset root differs from paths.dataset_root")
    if crop_audit.get("configured_dataset_root") != str(configured_paths.dataset_root):
        raise ValueError("crop audit does not bind the configured dataset root")
    if crop_audit.get("dataset_root_override_used") is not False:
        raise ValueError("a dataset-root override cannot be frozen; update paths.dataset_root")
    if crop_audit.get("configured_output_root") != str(configured_paths.output_root):
        raise ValueError("crop audit does not bind the configured output root")
    if crop_audit.get("configured_prepared_root") != str(configured_paths.prepared_root):
        raise ValueError("crop audit does not bind the configured prepared root")
    if crop_audit.get("prepared_root_override_used") is not False:
        raise ValueError("a prepared-root override cannot be frozen; update the paths section")
    if crop_audit.get("section_manifest_sha256") != sha256_file(section_manifest):
        raise ValueError("crop audit and section manifest hashes disagree")
    if crop_audit.get("crop_manifest_sha256") != sha256_file(crop_manifest):
        raise ValueError("crop audit and crop manifest hashes disagree")
    if crop_audit.get("full_section_reference_manifest_sha256") != sha256_file(
        reference_manifest
    ):
        raise ValueError("crop audit and full-section reference manifest hashes disagree")
    if not is_sha256(crop_audit.get("crop_config_sha256")):
        raise ValueError("crop audit lacks a valid crop_config_sha256")
    if crop_audit.get("source_dataset_modified") is not False:
        raise ValueError("crop audit does not certify an unmodified source dataset")
    if crop_audit.get("gt_present_in_crop_annotation") is not False:
        raise ValueError("crop annotations are not certified GT-free")
    if crop_audit.get("gt_present_only_in_manifest") is not True:
        raise ValueError("crop audit does not certify manifest-only GT storage")
    tree_splits: dict[str, set[str]] = {}
    seen_sections: set[str] = set()
    section_by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.get("schema_version") != "racpith.source_sample.v1":
            raise ValueError("unsupported section-manifest schema")
        section_id = str(row["section_id"])
        if section_id in seen_sections:
            raise ValueError(f"duplicate section-manifest row for {section_id}")
        seen_sections.add(section_id)
        section_by_id[section_id] = row
        tree_splits.setdefault(str(row["tree_id"]), set()).add(str(row["split"]))
    if not rows:
        raise ValueError("section manifest is empty")
    leaking = {tree: sorted(splits) for tree, splits in tree_splits.items() if len(splits) != 1}
    if leaking:
        raise ValueError(f"tree leakage prevents freeze: {leaking}")
    development_trees = sorted(
        tree for tree, splits in tree_splits.items() if splits == {development_split}
    )
    sealed_trees = sorted(
        tree for tree, splits in tree_splits.items() if splits == {sealed_split}
    )
    if not development_trees or not sealed_trees:
        raise ValueError("development and sealed splits must both contain biological trees")
    if set(development_trees) & set(sealed_trees):
        raise ValueError("development/sealed tree overlap")

    def validate_derived_manifest(path: Path, *, reference: bool) -> None:
        seen_crops: set[str] = set()
        seen_reference_sections: set[str] = set()
        for item in read_jsonl(path):
            if item.get("schema_version") != "racpith.crop_manifest.v1":
                raise ValueError(f"unsupported crop-manifest schema in {path}")
            crop_id = str(item["crop_id"])
            if crop_id in seen_crops:
                raise ValueError(f"duplicate crop-manifest row for {crop_id}")
            seen_crops.add(crop_id)
            section_id = str(item["section_id"])
            source_row = section_by_id.get(section_id)
            if source_row is None:
                raise ValueError(f"crop {crop_id} refers to an unknown section")
            if (
                str(item["tree_id"]) != str(source_row["tree_id"])
                or str(item["split"]) != str(source_row["split"])
            ):
                raise ValueError(f"crop/section lineage mismatch for {crop_id}")
            if reference and item.get("analysis_role") != "TARGET_REFERENCE_ONLY":
                raise ValueError(
                    f"full-section reference {crop_id} lacks analysis_role=TARGET_REFERENCE_ONLY"
                )
            for path_field, hash_field in (
                ("crop_image_path", "crop_image_sha256"),
                ("crop_annotation_path", "crop_annotation_sha256"),
            ):
                content_path = Path(str(item.get(path_field, ""))).expanduser()
                expected_hash = item.get(hash_field)
                if (
                    not content_path.is_absolute()
                    or not content_path.is_file()
                    or not is_sha256(expected_hash)
                ):
                    raise ValueError(
                        f"crop {crop_id} lacks a valid immutable {path_field}/{hash_field}"
                    )
                if sha256_file(content_path) != expected_hash:
                    raise ValueError(
                        f"crop content changed before freeze: {crop_id}/{path_field}"
                    )
            if reference:
                if section_id in seen_reference_sections:
                    raise ValueError(f"multiple full-section references for {section_id}")
                seen_reference_sections.add(section_id)
        if not seen_crops:
            raise ValueError(f"derived manifest is empty: {path}")
        if reference and seen_reference_sections != seen_sections:
            missing_sections = sorted(seen_sections - seen_reference_sections)
            raise ValueError(
                "full-section reference coverage differs from section manifest: "
                f"missing={missing_sections[:10]}"
            )

    validate_derived_manifest(crop_manifest, reference=False)
    validate_derived_manifest(reference_manifest, reference=True)

    config["calibration_status"] = "DEVELOPMENT_FROZEN"
    config_body_sha256 = stable_hash(config)
    freeze_record = {
        "schema_version": "racpith.freeze.v1",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "development_run_id": development_run_id,
        "code_revision": code_revision,
        "source_tree_sha256": sha256_source_tree(Path(__file__).resolve().parents[1]),
        "decision_record_path": str(decision),
        "decision_record_sha256": sha256_file(decision),
        "section_manifest_path": str(section_manifest),
        "section_manifest_sha256": sha256_file(section_manifest),
        "crop_manifest_path": str(crop_manifest),
        "crop_manifest_sha256": sha256_file(crop_manifest),
        "full_section_reference_manifest_path": str(reference_manifest),
        "full_section_reference_manifest_sha256": sha256_file(reference_manifest),
        "crop_audit_path": str(crop_audit_path),
        "crop_audit_sha256": sha256_file(crop_audit_path),
        "crop_config_sha256": crop_audit["crop_config_sha256"],
        "paths_config_sha256": paths_config_hash,
        "pre_freeze_config_sha256": source_config.sha256,
        "config_body_sha256": config_body_sha256,
        "development_split": development_split,
        "sealed_split": sealed_split,
        "development_tree_ids": development_trees,
        "sealed_tree_ids": sealed_trees,
        "sealed_results_used_for_freeze": False,
    }
    freeze_record["freeze_body_sha256"] = stable_hash(freeze_record)
    config["freeze"] = freeze_record
    atomic_write_json(output, config)
    print(
        f"frozen configuration written to {output}; "
        f"development trees={len(development_trees)}, sealed trees={len(sealed_trees)}"
    )


if __name__ == "__main__":
    main()
