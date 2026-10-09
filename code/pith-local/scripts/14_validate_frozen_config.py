#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from racpith.config import load_config, resolve_runtime_paths, stable_hash
from racpith.provenance import (
    is_sha256,
    read_json_object,
    read_jsonl,
    sha256_file,
    sha256_source_tree,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed unless config and tree split are frozen")
    parser.add_argument("--config", required=True)
    parser.add_argument("--section-manifest", required=True)
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--crop-audit", required=True)
    parser.add_argument("--decision-record", required=True)
    args = parser.parse_args()

    frozen = load_config(args.config)
    configured_paths = resolve_runtime_paths(
        frozen,
        project_root=Path(__file__).resolve().parents[1],
    )
    if frozen.data.get("calibration_status") != "DEVELOPMENT_FROZEN":
        raise ValueError("sealed execution requires calibration_status=DEVELOPMENT_FROZEN")
    record = frozen.section("freeze")
    if record.get("schema_version") != "racpith.freeze.v1":
        raise ValueError("unsupported freeze-record schema")
    required = {
        "development_run_id",
        "code_revision",
        "source_tree_sha256",
        "decision_record_path",
        "decision_record_sha256",
        "section_manifest_sha256",
        "crop_manifest_sha256",
        "full_section_reference_manifest_sha256",
        "crop_audit_sha256",
        "crop_config_sha256",
        "paths_config_sha256",
        "pre_freeze_config_sha256",
        "config_body_sha256",
        "development_split",
        "sealed_split",
        "development_tree_ids",
        "sealed_tree_ids",
        "sealed_results_used_for_freeze",
        "freeze_body_sha256",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"frozen configuration lacks fields: {sorted(missing)}")
    for field in ("development_run_id", "code_revision", "development_split", "sealed_split"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"freeze field {field} must be a non-empty string")
    if record["development_split"] == record["sealed_split"]:
        raise ValueError("development and sealed split names must be distinct")
    for field in ("development_tree_ids", "sealed_tree_ids"):
        if not isinstance(record[field], list) or any(
            not isinstance(value, str) or not value.strip()
            for value in record[field]
        ):
            raise ValueError(
                f"freeze field {field} must be a list of non-empty tree-id strings"
            )
        values = list(record[field])
        if not values or len(values) != len(set(values)):
            raise ValueError(f"freeze field {field} must contain unique biological trees")
    if record["sealed_results_used_for_freeze"] is not False:
        raise ValueError("freeze record does not certify sealed-test isolation")
    if not is_sha256(record["config_body_sha256"]):
        raise ValueError("freeze record has an invalid config_body_sha256")
    if not is_sha256(record["pre_freeze_config_sha256"]):
        raise ValueError("freeze record has an invalid pre_freeze_config_sha256")
    if not is_sha256(record["paths_config_sha256"]):
        raise ValueError("freeze record has an invalid paths_config_sha256")
    if not is_sha256(record["freeze_body_sha256"]):
        raise ValueError("freeze record has an invalid freeze_body_sha256")
    freeze_body = dict(record)
    freeze_body.pop("freeze_body_sha256", None)
    if stable_hash(freeze_body) != record["freeze_body_sha256"]:
        raise ValueError("freeze metadata has changed since configuration freeze")
    config_body = dict(frozen.data)
    config_body.pop("freeze", None)
    if stable_hash(config_body) != record["config_body_sha256"]:
        raise ValueError("frozen configuration body has changed since configuration freeze")
    current_source_hash = sha256_source_tree(Path(__file__).resolve().parents[1])
    if current_source_hash != record["source_tree_sha256"]:
        raise ValueError("executable source bundle has changed since configuration freeze")
    decision_path = Path(args.decision_record).expanduser().resolve()
    if not decision_path.is_file() or decision_path.stat().st_size == 0:
        raise ValueError("development decision record is absent or empty")
    if sha256_file(decision_path) != record["decision_record_sha256"]:
        raise ValueError("development decision record has changed since configuration freeze")
    manifest_path = Path(args.section_manifest).expanduser().resolve()
    if sha256_file(manifest_path) != record["section_manifest_sha256"]:
        raise ValueError("section manifest has changed since configuration freeze")
    crop_manifest_path = Path(args.crop_manifest).expanduser().resolve()
    reference_manifest_path = Path(args.reference_manifest).expanduser().resolve()
    crop_audit_path = Path(args.crop_audit).expanduser().resolve()
    if sha256_file(crop_manifest_path) != record["crop_manifest_sha256"]:
        raise ValueError("crop manifest has changed since configuration freeze")
    if (
        sha256_file(reference_manifest_path)
        != record["full_section_reference_manifest_sha256"]
    ):
        raise ValueError("full-section reference manifest has changed since freeze")
    if sha256_file(crop_audit_path) != record["crop_audit_sha256"]:
        raise ValueError("crop-generation audit has changed since configuration freeze")
    crop_audit = read_json_object(crop_audit_path)
    if crop_audit.get("schema_version") != "racpith.crop_generation_audit.v1":
        raise ValueError("unsupported crop-generation audit schema")
    current_paths_hash = stable_hash(dict(frozen.section("paths")))
    if current_paths_hash != record["paths_config_sha256"]:
        raise ValueError("frozen paths configuration differs from freeze record")
    if crop_audit.get("paths_config_hash") != record["paths_config_sha256"]:
        raise ValueError("crop audit paths configuration differs from freeze record")
    if crop_audit.get("dataset_root") != str(configured_paths.dataset_root):
        raise ValueError("crop audit dataset root differs from frozen paths.dataset_root")
    if crop_audit.get("configured_dataset_root") != str(configured_paths.dataset_root):
        raise ValueError("crop audit does not bind frozen paths.dataset_root")
    if crop_audit.get("dataset_root_override_used") is not False:
        raise ValueError("frozen crop audit used an unregistered dataset-root override")
    if crop_audit.get("configured_output_root") != str(configured_paths.output_root):
        raise ValueError("crop audit does not bind frozen paths.output_root")
    if crop_audit.get("configured_prepared_root") != str(configured_paths.prepared_root):
        raise ValueError("crop audit does not bind the frozen prepared directory")
    if crop_audit.get("prepared_root_override_used") is not False:
        raise ValueError("frozen crop audit used an unregistered prepared-root override")
    if crop_audit.get("section_manifest_sha256") != record["section_manifest_sha256"]:
        raise ValueError("crop audit no longer identifies the frozen section manifest")
    if crop_audit.get("crop_manifest_sha256") != record["crop_manifest_sha256"]:
        raise ValueError("crop audit no longer identifies the frozen crop manifest")
    if (
        crop_audit.get("full_section_reference_manifest_sha256")
        != record["full_section_reference_manifest_sha256"]
    ):
        raise ValueError("crop audit no longer identifies the frozen reference manifest")
    if crop_audit.get("crop_config_sha256") != record["crop_config_sha256"]:
        raise ValueError("crop configuration hash differs from freeze record")
    if not is_sha256(crop_audit.get("crop_config_sha256")):
        raise ValueError("crop audit lacks a valid crop_config_sha256")
    if crop_audit.get("source_dataset_modified") is not False:
        raise ValueError("crop audit does not certify an unmodified source dataset")
    if crop_audit.get("gt_present_in_crop_annotation") is not False:
        raise ValueError("crop annotations are not certified GT-free")
    if crop_audit.get("gt_present_only_in_manifest") is not True:
        raise ValueError("crop audit does not certify manifest-only GT storage")
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError("section manifest is empty")
    section_by_id = {}
    tree_splits: dict[str, set[str]] = {}
    for row in rows:
        if row.get("schema_version") != "racpith.source_sample.v1":
            raise ValueError("unsupported section-manifest schema")
        section_id = str(row["section_id"])
        if section_id in section_by_id:
            raise ValueError(f"duplicate section-manifest row for {section_id}")
        section_by_id[section_id] = row
        tree_splits.setdefault(str(row["tree_id"]), set()).add(str(row["split"]))
    leaking = {
        tree: sorted(splits) for tree, splits in tree_splits.items() if len(splits) != 1
    }
    if leaking:
        raise ValueError(f"biological-tree split leakage in section manifest: {leaking}")

    def validate_derived_manifest(path: Path, *, reference: bool) -> None:
        seen: set[str] = set()
        reference_sections: set[str] = set()
        for item in read_jsonl(path):
            if item.get("schema_version") != "racpith.crop_manifest.v1":
                raise ValueError(f"unsupported crop-manifest schema in {path}")
            crop_id = str(item["crop_id"])
            if crop_id in seen:
                raise ValueError(f"duplicate crop-manifest row for {crop_id}")
            seen.add(crop_id)
            section_id = str(item["section_id"])
            source = section_by_id.get(section_id)
            if source is None:
                raise ValueError(f"crop {crop_id} refers to an unknown section")
            if (
                str(item["tree_id"]) != str(source["tree_id"])
                or str(item["split"]) != str(source["split"])
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
                        f"crop {crop_id} lacks a valid frozen {path_field}/{hash_field}"
                    )
                if sha256_file(content_path) != expected_hash:
                    raise ValueError(
                        f"frozen crop content has changed: {crop_id}/{path_field}"
                    )
            if reference:
                if section_id in reference_sections:
                    raise ValueError(f"multiple full-section references for {section_id}")
                reference_sections.add(section_id)
        if not seen:
            raise ValueError(f"derived manifest is empty: {path}")
        if reference and reference_sections != set(section_by_id):
            missing_sections = sorted(set(section_by_id) - reference_sections)
            raise ValueError(
                f"full-section reference coverage differs from section manifest: "
                f"missing={missing_sections[:10]}"
            )

    validate_derived_manifest(crop_manifest_path, reference=False)
    validate_derived_manifest(reference_manifest_path, reference=True)
    actual_development = sorted(
        {str(row["tree_id"]) for row in rows if row["split"] == record["development_split"]}
    )
    actual_sealed = sorted(
        {str(row["tree_id"]) for row in rows if row["split"] == record["sealed_split"]}
    )
    if actual_development != sorted(str(value) for value in record["development_tree_ids"]):
        raise ValueError("development tree identities differ from freeze record")
    if actual_sealed != sorted(str(value) for value in record["sealed_tree_ids"]):
        raise ValueError("sealed tree identities differ from freeze record")
    if set(actual_development) & set(actual_sealed):
        raise ValueError("development/sealed tree leakage")
    print(
        f"frozen configuration valid: config_hash={frozen.sha256}; "
        f"sealed split={record['sealed_split']} ({len(actual_sealed)} trees)"
    )


if __name__ == "__main__":
    main()
