from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..contracts import EvidenceBundle, LocateResult
from ..provenance import is_sha256, read_json_object, read_jsonl, sha256_file


CROP_CONTENT_HASH_FIELDS = ("crop_image_sha256", "crop_annotation_sha256")


@dataclass(frozen=True)
class ContractFinding:
    severity: str
    check: str
    crop_id: str | None
    message: str


def _strict_json(path: Path) -> dict[str, Any]:
    return read_json_object(path)


def _locate_result_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        LocateResult.from_dict(payload).as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _append(
    findings: list[ContractFinding],
    severity: str,
    check: str,
    message: str,
    crop_id: str | None = None,
) -> None:
    findings.append(ContractFinding(severity, check, crop_id, message))


def _finite_scalar(value: Any) -> bool:
    """Return whether a persisted scalar is a finite numeric value (not bool)."""

    if isinstance(value, bool) or value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def audit_split(rows: list[Mapping[str, Any]]) -> list[ContractFinding]:
    findings: list[ContractFinding] = []
    tree_splits: dict[str, set[str]] = {}
    section_splits: dict[str, set[str]] = {}
    for row in rows:
        tree_splits.setdefault(str(row["tree_id"]), set()).add(str(row["split"]))
        section_splits.setdefault(str(row["section_id"]), set()).add(str(row["split"]))
    for tree, splits in tree_splits.items():
        if len(splits) != 1:
            _append(findings, "ERROR", "TREE_SPLIT_LEAKAGE", f"{tree}: {sorted(splits)}")
    for section, splits in section_splits.items():
        if len(splits) != 1:
            _append(findings, "ERROR", "SECTION_SPLIT_LEAKAGE", f"{section}: {sorted(splits)}")
    return findings


def audit_run(
    crop_manifest_path: str | Path,
    evidence_index_path: str | Path,
    prediction_root: str | Path,
    expected_config_hash: str,
    contribution_root: str | Path | None = None,
    contribution_index_path: str | Path | None = None,
    splits: set[str] | None = None,
    uncertainty_index_path: str | Path | None = None,
    result_index_path: str | Path | None = None,
    target_domain_index_path: str | Path | None = None,
) -> dict[str, Any]:
    all_manifest = read_jsonl(crop_manifest_path)
    findings: list[ContractFinding] = []
    full_manifest_by_crop: dict[str, dict[str, Any]] = {}
    for row in all_manifest:
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            _append(
                findings,
                "ERROR",
                "CROP_MANIFEST_SCHEMA",
                "unsupported crop-manifest row",
                str(row.get("crop_id")) if row.get("crop_id") is not None else None,
            )
            continue
        missing_fields = sorted(
            {
                "crop_id",
                "tree_id",
                "section_id",
                "split",
                "crop_size_px",
                "crop_image_path",
                "crop_annotation_path",
                *CROP_CONTENT_HASH_FIELDS,
            }
            - set(row)
        )
        if missing_fields:
            _append(
                findings,
                "ERROR",
                "CROP_MANIFEST_FIELDS",
                f"missing required fields: {missing_fields}",
                str(row.get("crop_id")) if row.get("crop_id") is not None else None,
            )
            continue
        crop_id = str(row["crop_id"])
        if crop_id in full_manifest_by_crop:
            _append(
                findings,
                "ERROR",
                "DUPLICATE_MANIFEST_CROP",
                "more than one crop-manifest row",
                crop_id,
            )
            continue
        full_manifest_by_crop[crop_id] = row
    findings.extend(audit_split(list(full_manifest_by_crop.values())))
    manifest_by_crop = {
        crop_id: row
        for crop_id, row in full_manifest_by_crop.items()
        if not splits or str(row["split"]) in splits
    }
    manifest = list(manifest_by_crop.values())
    if not manifest:
        _append(
            findings,
            "ERROR",
            "EMPTY_SELECTED_MANIFEST",
            "selected crop-manifest scope is empty",
        )
    selected_manifest_ids = set(manifest_by_crop)
    target_domain_count = 0
    target_domain_index_sha256: str | None = None
    if target_domain_index_path is not None:
        target_domain_index_sha256 = sha256_file(target_domain_index_path)
        expected_reference_manifest_sha256 = sha256_file(crop_manifest_path)
        expected_reference_result_index_sha256 = (
            sha256_file(result_index_path) if result_index_path is not None else None
        )
        seen_target_sections: set[str] = set()
        target_rows = read_jsonl(target_domain_index_path)
        for target in target_rows:
            crop_id = str(target.get("reference_crop_id"))
            section_id = str(target.get("section_id"))
            source = manifest_by_crop.get(crop_id)
            if target.get("schema_version") != "racpith.target_domain.v1":
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_SCHEMA",
                    "unsupported target-domain row",
                    crop_id,
                )
                continue
            target_domain_count += 1
            if crop_id not in selected_manifest_ids or source is None:
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_ORPHAN",
                    "reference crop is not in the selected manifest split",
                    crop_id,
                )
                continue
            if section_id in seen_target_sections:
                _append(
                    findings,
                    "ERROR",
                    "DUPLICATE_TARGET_DOMAIN",
                    "more than one target-domain row for a section",
                    section_id,
                )
            seen_target_sections.add(section_id)
            if source.get("analysis_role") != "TARGET_REFERENCE_ONLY":
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_REFERENCE_ROLE",
                    "target-domain reference crop is not marked TARGET_REFERENCE_ONLY",
                    crop_id,
                )
            if any(
                str(target.get(field)) != str(source.get(field))
                for field in ("tree_id", "section_id", "split")
            ):
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_LINEAGE",
                    "target-domain row differs from reference manifest",
                    crop_id,
                )
            if target.get("config_hash") != expected_config_hash:
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_CONFIG_HASH",
                    "target-domain row uses another configuration",
                    crop_id,
                )
            if target.get("reference_manifest_sha256") != expected_reference_manifest_sha256:
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_REFERENCE_MANIFEST_HASH",
                    "target-domain row does not bind the supplied reference manifest",
                    crop_id,
                )
            if not is_sha256(target.get("reference_manifest_sha256")) or not is_sha256(
                target.get("reference_result_index_sha256")
            ):
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_PROVENANCE",
                    "target-domain row lacks valid reference manifest/result-index hashes",
                    crop_id,
                )
            if (
                expected_reference_result_index_sha256 is not None
                and target.get("reference_result_index_sha256")
                != expected_reference_result_index_sha256
            ):
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_REFERENCE_RESULT_INDEX_HASH",
                    "target-domain row does not bind the supplied reference result index",
                    crop_id,
                )
            if target.get("gt_used_only_after_reference_fit") is not True:
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_GT_ORDER",
                    "target-domain row lacks the post-hoc GT-use certificate",
                    crop_id,
                )
            reference_status = str(target.get("reference_result_status"))
            reference_prediction_hash = target.get("reference_prediction_sha256")
            if reference_status in {"FINISHED", "RESUMED"}:
                if not is_sha256(reference_prediction_hash):
                    _append(
                        findings,
                        "ERROR",
                        "TARGET_DOMAIN_REFERENCE_RESULT",
                        "finished reference result lacks a valid prediction hash",
                        crop_id,
                    )
            elif reference_prediction_hash is not None:
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_REFERENCE_RESULT",
                    "non-finished reference result exposes a prediction hash",
                    crop_id,
                )
            reference_crop_id = target.get("reference_crop_id")
            if not isinstance(reference_crop_id, str) or not reference_crop_id.strip():
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_REFERENCE_ID",
                    "target-domain row lacks a non-empty reference_crop_id",
                    crop_id,
                )
            source_scale = source.get("normalization_scale_px")
            target_scale = target.get("reference_normalization_scale_px")
            if not _finite_scalar(source_scale) or float(source_scale) <= 0.0:
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_SCALE_GATE",
                    "reference crop lacks a finite positive normalization scale",
                    crop_id,
                )
            if not _finite_scalar(target_scale) or float(target_scale) <= 0.0:
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_SCALE_GATE",
                    "target-domain row lacks a finite positive reference scale",
                    crop_id,
                )
            elif _finite_scalar(source_scale) and not np.isclose(
                float(target_scale),
                float(source_scale),
                rtol=0.0,
                atol=1e-10,
            ):
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_SCALE_GATE",
                    "target-domain reference scale differs from the manifest",
                    crop_id,
                )
            label = str(target.get("target_domain"))
            bias = target.get("target_bias_norm")
            bias_px = target.get("target_bias_px")
            if label not in {"TARGET_ALIGNED", "ECCENTRIC_GT", "TARGET_UNKNOWN"}:
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_LABEL",
                    f"unsupported target-domain label={label}",
                    crop_id,
                )
            elif label == "TARGET_UNKNOWN" and (
                bias is not None or bias_px is not None
            ):
                _append(
                    findings,
                    "ERROR",
                    "TARGET_DOMAIN_BIAS_GATE",
                    "TARGET_UNKNOWN exposes a target bias",
                    crop_id,
                )
            elif label != "TARGET_UNKNOWN":
                if not _finite_scalar(bias):
                    _append(
                        findings,
                        "ERROR",
                        "TARGET_DOMAIN_BIAS_GATE",
                        "resolved target-domain row lacks finite target bias",
                        crop_id,
                    )
                elif not _finite_scalar(bias_px):
                    _append(
                        findings,
                        "ERROR",
                        "TARGET_DOMAIN_BIAS_GATE",
                        "resolved target-domain row lacks finite pixel bias",
                        crop_id,
                    )
                else:
                    bias_norm_value = float(bias)
                    bias_px_value = float(bias_px)
                    if bias_norm_value < 0.0 or bias_px_value < 0.0:
                        _append(
                            findings,
                            "ERROR",
                            "TARGET_DOMAIN_BIAS_GATE",
                            "target bias norms must be non-negative",
                            crop_id,
                        )
                    elif _finite_scalar(source_scale) and not np.isclose(
                        bias_px_value,
                        bias_norm_value * float(source_scale),
                        rtol=1e-9,
                        atol=1e-8,
                    ):
                        _append(
                            findings,
                            "ERROR",
                            "TARGET_DOMAIN_BIAS_CONSISTENCY",
                            "pixel and normalized target biases disagree with reference scale",
                            crop_id,
                        )
        expected_target_sections = {str(row["section_id"]) for row in manifest}
        if seen_target_sections != expected_target_sections:
            _append(
                findings,
                "ERROR",
                "TARGET_DOMAIN_COVERAGE",
                "target-domain rows do not cover the selected reference sections exactly",
            )
    for crop_id, row in manifest_by_crop.items():
        for path_field, hash_field in (
            ("crop_image_path", "crop_image_sha256"),
            ("crop_annotation_path", "crop_annotation_sha256"),
        ):
            if not is_sha256(row.get(hash_field)):
                _append(
                    findings,
                    "ERROR",
                    "CROP_MANIFEST_CONTENT_HASH",
                    f"invalid {hash_field}",
                    crop_id,
                )
                continue
            try:
                content_path = Path(str(row[path_field])).expanduser()
                if not content_path.is_absolute() or not content_path.is_file():
                    raise ValueError(
                        f"{path_field} must be an existing absolute file: "
                        f"{content_path}"
                    )
                if sha256_file(content_path) != row[hash_field]:
                    raise ValueError(f"{path_field} content differs from manifest hash")
            except Exception as exc:
                _append(
                    findings,
                    "ERROR",
                    "CROP_INPUT_CONTENT_HASH",
                    str(exc),
                    crop_id,
                )
    evidence_index: dict[str, dict[str, Any]] = {}
    seen_evidence_index: set[str] = set()
    for row in read_jsonl(evidence_index_path):
        crop_id = str(row["crop_id"])
        if crop_id in seen_evidence_index:
            _append(
                findings,
                "ERROR",
                "DUPLICATE_EVIDENCE_INDEX",
                "more than one evidence-index row",
                crop_id,
            )
            continue
        seen_evidence_index.add(crop_id)
        full_source = full_manifest_by_crop.get(crop_id)
        if full_source is not None and any(
            str(row.get(field)) != str(full_source.get(field))
            for field in (
                "tree_id",
                "section_id",
                "split",
                *CROP_CONTENT_HASH_FIELDS,
            )
        ):
            _append(
                findings,
                "ERROR",
                "EVIDENCE_INDEX_LINEAGE",
                "identity, split, or crop-content hashes differ from crop manifest",
                crop_id,
            )
        if splits and full_source is not None and crop_id not in selected_manifest_ids:
            continue
        evidence_index[crop_id] = row
    for crop_id, row in evidence_index.items():
        if row.get("schema_version") != "racpith.evidence_index.v1":
            _append(
                findings,
                "ERROR",
                "EVIDENCE_INDEX_SCHEMA",
                "unsupported evidence-index row",
                crop_id,
            )
        if crop_id not in selected_manifest_ids:
            _append(
                findings,
                "ERROR",
                "ORPHAN_EVIDENCE_INDEX",
                "crop is not in the selected manifest split",
                crop_id,
            )
        status = str(row.get("status"))
        if status not in {"PASS", "FAIL", "MANIFEST_INELIGIBLE"}:
            _append(
                findings,
                "ERROR",
                "EVIDENCE_INDEX_STATUS",
                f"unsupported status={status}",
                crop_id,
            )
        for field in CROP_CONTENT_HASH_FIELDS:
            if not is_sha256(row.get(field)):
                _append(
                    findings,
                    "ERROR",
                    "EVIDENCE_INDEX_CROP_HASH",
                    f"invalid or missing {field}",
                    crop_id,
                )
    prediction_by_crop: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(Path(prediction_root).rglob("*.json")):
        try:
            result = _strict_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _append(findings, "ERROR", "PREDICTION_JSON", f"{path}: {exc}")
            continue
        if result.get("schema_version") != "racpith.locate.v1":
            continue
        try:
            validated = LocateResult.from_dict(result)
        except (KeyError, TypeError, ValueError) as exc:
            _append(findings, "ERROR", "PREDICTION_SCHEMA", f"{path}: {exc}")
            continue
        crop_id = validated.crop_id
        if (
            splits
            and crop_id in full_manifest_by_crop
            and crop_id not in selected_manifest_ids
        ):
            continue
        if crop_id in prediction_by_crop:
            _append(findings, "ERROR", "DUPLICATE_PREDICTION", str(path), crop_id)
        prediction_by_crop[crop_id] = (path, result)

    result_index_by_crop: dict[str, dict[str, Any]] = {}
    result_index_sha256: str | None = None
    if result_index_path is not None:
        result_index_sha256 = sha256_file(result_index_path)
        prediction_root_resolved = Path(prediction_root).expanduser().resolve()
        finished_statuses = {"FINISHED", "RESUMED"}
        allowed_statuses = {
            *finished_statuses,
            "CRASH",
            "UPSTREAM_EVIDENCE_FAIL",
        }
        seen_result_index: set[str] = set()
        for index_row in read_jsonl(result_index_path):
            if index_row.get("schema_version") != "racpith.result_index.v1":
                _append(
                    findings,
                    "ERROR",
                    "RESULT_INDEX_SCHEMA",
                    "unsupported localization result-index row",
                )
                continue
            crop_id = str(index_row["crop_id"])
            if crop_id in seen_result_index:
                _append(
                    findings,
                    "ERROR",
                    "DUPLICATE_RESULT_INDEX",
                    "more than one localization result-index row",
                    crop_id,
                )
                continue
            seen_result_index.add(crop_id)
            full_source = full_manifest_by_crop.get(crop_id)
            if full_source is not None and any(
                str(index_row.get(field)) != str(full_source.get(field))
                for field in ("tree_id", "section_id", "split")
            ):
                _append(
                    findings,
                    "ERROR",
                    "RESULT_INDEX_LINEAGE",
                    "tree/section/split differs from crop manifest",
                    crop_id,
                )
            if splits and full_source is not None and crop_id not in selected_manifest_ids:
                continue
            result_index_by_crop[crop_id] = index_row
            if crop_id not in selected_manifest_ids:
                _append(
                    findings,
                    "ERROR",
                    "ORPHAN_RESULT_INDEX",
                    "crop is not in the selected manifest split",
                    crop_id,
                )
            if index_row.get("config_hash") != expected_config_hash:
                _append(
                    findings,
                    "ERROR",
                    "RESULT_INDEX_CONFIG_HASH",
                    "localization result index uses another configuration",
                    crop_id,
                )
            status = str(index_row.get("status"))
            if status not in allowed_statuses:
                _append(
                    findings,
                    "ERROR",
                    "RESULT_INDEX_STATUS",
                    f"unregistered status={status}",
                    crop_id,
                )
            path_value = index_row.get("prediction_path")
            if status in finished_statuses:
                if path_value is None:
                    _append(
                        findings,
                        "ERROR",
                        "RESULT_INDEX_PATH",
                        "finished row lacks prediction_path",
                        crop_id,
                    )
                    continue
                registered_path = Path(str(path_value)).expanduser().resolve()
                if not registered_path.is_relative_to(prediction_root_resolved):
                    _append(
                        findings,
                        "ERROR",
                        "RESULT_INDEX_PATH_ESCAPE",
                        str(registered_path),
                        crop_id,
                    )
                    continue
                try:
                    if sha256_file(registered_path) != index_row.get("prediction_sha256"):
                        raise ValueError("prediction content hash differs from result index")
                except Exception as exc:
                    _append(
                        findings,
                        "ERROR",
                        "RESULT_INDEX_PREDICTION_HASH",
                        str(exc),
                        crop_id,
                    )
                scanned = prediction_by_crop.get(crop_id)
                if scanned is None or scanned[0].resolve() != registered_path:
                    _append(
                        findings,
                        "ERROR",
                        "RESULT_INDEX_PREDICTION_JOIN",
                        "registered prediction is absent or resolves to another file",
                        crop_id,
                    )
                elif str(scanned[1].get("state")) != str(index_row.get("state")):
                    _append(
                        findings,
                        "ERROR",
                        "RESULT_INDEX_STATE",
                        "prediction state differs from result index",
                        crop_id,
                    )
            elif path_value is not None:
                _append(
                    findings,
                    "ERROR",
                    "FAILED_RESULT_EXPOSES_PREDICTION",
                    "failed localization row must not register a prediction",
                    crop_id,
                )
        for crop_id in sorted(selected_manifest_ids - set(result_index_by_crop)):
            _append(
                findings,
                "ERROR",
                "MISSING_RESULT_INDEX_ROW",
                "manifest crop has no localization status row",
                crop_id,
            )
        for crop_id in sorted(set(prediction_by_crop) - set(result_index_by_crop)):
            _append(
                findings,
                "ERROR",
                "UNREGISTERED_PREDICTION",
                "prediction is not registered by result index",
                crop_id,
            )
    uncertainty_by_crop: dict[str, dict[str, Any]] = {}
    if uncertainty_index_path is not None:
        uncertainty_root = Path(uncertainty_index_path).expanduser().resolve().parent
        registered_uncertainty_files: set[Path] = set()
        allowed_uncertainty_statuses = {
            "FINISHED",
            "RESUMED",
            "FAIL",
            "UPSTREAM_LOCALIZATION_FAIL",
            "UPSTREAM_EVIDENCE_FAIL",
        }
        seen_uncertainty_index: set[str] = set()
        for row in read_jsonl(uncertainty_index_path):
            crop_id = str(row["crop_id"])
            if crop_id in seen_uncertainty_index:
                _append(
                    findings,
                    "ERROR",
                    "DUPLICATE_UNCERTAINTY",
                    "more than one uncertainty index row",
                    crop_id,
                )
                continue
            seen_uncertainty_index.add(crop_id)
            full_source = full_manifest_by_crop.get(crop_id)
            if full_source is not None and any(
                str(row.get(field)) != str(full_source.get(field))
                for field in ("tree_id", "section_id", "split")
            ):
                _append(
                    findings,
                    "ERROR",
                    "UNCERTAINTY_INDEX_LINEAGE",
                    "tree/section/split differs from crop manifest",
                    crop_id,
                )
            if splits and full_source is not None and crop_id not in selected_manifest_ids:
                continue
            uncertainty_by_crop[crop_id] = row
            if row.get("schema_version") != "racpith.uncertainty_index.v1":
                _append(
                    findings,
                    "ERROR",
                    "UNCERTAINTY_INDEX_SCHEMA",
                    "unsupported uncertainty index row",
                    crop_id,
                )
            if crop_id not in selected_manifest_ids:
                _append(
                    findings,
                    "ERROR",
                    "ORPHAN_UNCERTAINTY_INDEX",
                    "crop is not in the selected manifest split",
                    crop_id,
                )
            if row.get("config_hash") != expected_config_hash:
                _append(
                    findings,
                    "ERROR",
                    "UNCERTAINTY_INDEX_CONFIG_HASH",
                    "uncertainty index uses another configuration",
                    crop_id,
                )
            if (
                result_index_sha256 is not None
                and row.get("source_result_index_sha256") != result_index_sha256
            ):
                _append(
                    findings,
                    "ERROR",
                    "UNCERTAINTY_RESULT_INDEX_HASH",
                    "uncertainty row does not bind the supplied localization index",
                    crop_id,
                )
            status = str(row.get("status"))
            if status not in allowed_uncertainty_statuses:
                _append(
                    findings,
                    "ERROR",
                    "UNCERTAINTY_INDEX_STATUS",
                    f"unsupported status={status}",
                    crop_id,
                )
            if status in {"FINISHED", "RESUMED"}:
                sidecar_value = row.get("uncertainty_path")
                if sidecar_value is None:
                    _append(
                        findings,
                        "ERROR",
                        "UNCERTAINTY_INDEX_PATH",
                        "finished row lacks uncertainty_path",
                        crop_id,
                    )
                else:
                    sidecar_file = Path(str(sidecar_value)).expanduser().resolve()
                    if not sidecar_file.is_relative_to(uncertainty_root):
                        _append(
                            findings,
                            "ERROR",
                            "UNCERTAINTY_PATH_ESCAPE",
                            str(sidecar_file),
                            crop_id,
                        )
                    else:
                        registered_uncertainty_files.add(sidecar_file)
                        try:
                            if sha256_file(sidecar_file) != row.get(
                                "uncertainty_sha256"
                            ):
                                raise ValueError("uncertainty sidecar hash differs")
                        except Exception as exc:
                            _append(
                                findings,
                                "ERROR",
                                "UNCERTAINTY_INDEX_ARTIFACT",
                                str(exc),
                                crop_id,
                            )
            elif row.get("uncertainty_path") is not None:
                _append(
                    findings,
                    "ERROR",
                    "UNCERTAINTY_FAILED_ARTIFACT",
                    "non-finished uncertainty row exposes a sidecar",
                    crop_id,
                )
        uncertainty_artifact_root = uncertainty_root / "per_crop"
        if uncertainty_artifact_root.exists():
            for sidecar_file in sorted(uncertainty_artifact_root.rglob("*.json")):
                resolved = sidecar_file.resolve()
                file_crop_id = sidecar_file.stem
                if (
                    resolved not in registered_uncertainty_files
                    and (
                        file_crop_id in selected_manifest_ids
                        or file_crop_id not in full_manifest_by_crop
                    )
                ):
                    _append(
                        findings,
                        "ERROR",
                        "UNREGISTERED_UNCERTAINTY_FILE",
                        f"stale or unindexed sidecar: {resolved}",
                        file_crop_id or None,
                    )
    manifest_ids: set[str] = set()
    for row in manifest:
        crop_id = str(row["crop_id"])
        manifest_ids.add(crop_id)
        evidence_row = evidence_index.get(crop_id)
        if evidence_row is None:
            _append(findings, "ERROR", "MISSING_EVIDENCE_INDEX", "no evidence record", crop_id)
        else:
            if evidence_row.get("config_hash") != expected_config_hash:
                _append(
                    findings,
                    "ERROR",
                    "EVIDENCE_CONFIG_HASH_MISMATCH",
                    f"expected {expected_config_hash}, got {evidence_row.get('config_hash')}",
                    crop_id,
                )
            evidence_status = str(evidence_row.get("status"))
            manifest_eligible = bool(row.get("eligible", True))
            if not manifest_eligible:
                if evidence_status != "MANIFEST_INELIGIBLE":
                    _append(
                        findings,
                        "ERROR",
                        "INELIGIBLE_EVIDENCE_STATUS",
                        f"expected MANIFEST_INELIGIBLE, got {evidence_status}",
                        crop_id,
                    )
                if any(
                    evidence_row.get(field) is not None
                    for field in (
                        "metadata_path",
                        "npz_path",
                        "metadata_sha256",
                        "npz_sha256",
                    )
                ):
                    _append(
                        findings,
                        "ERROR",
                        "INELIGIBLE_EVIDENCE_ARTIFACT",
                        "manifest-ineligible row must not expose evidence artifacts",
                        crop_id,
                    )
            elif evidence_status != "PASS":
                _append(
                    findings,
                    "ERROR",
                    "EVIDENCE_BUILD_FAILURE",
                    f"status={evidence_status}: {evidence_row.get('reason')}",
                    crop_id,
                )
            else:
                try:
                    if sha256_file(evidence_row["metadata_path"]) != evidence_row.get(
                        "metadata_sha256"
                    ):
                        _append(
                            findings,
                            "ERROR",
                            "EVIDENCE_METADATA_CONTENT_HASH",
                            "evidence metadata content changed after indexing",
                            crop_id,
                        )
                    if sha256_file(evidence_row["npz_path"]) != evidence_row.get("npz_sha256"):
                        _append(
                            findings,
                            "ERROR",
                            "EVIDENCE_NPZ_CONTENT_HASH",
                            "evidence array content changed after indexing",
                            crop_id,
                        )
                    bundle = EvidenceBundle.load(evidence_row["metadata_path"])
                    bundle.validate()
                    if (
                        bundle.crop_id != crop_id
                        or bundle.tree_id != str(row["tree_id"])
                        or bundle.section_id != str(row["section_id"])
                        or bundle.crop_size_px
                        != tuple(int(value) for value in row["crop_size_px"])
                    ):
                        raise ValueError("evidence bundle identity differs from crop manifest")
                    if bundle.metadata.get("evidence_config_hash") != expected_config_hash:
                        _append(
                            findings,
                            "ERROR",
                            "EVIDENCE_METADATA_CONFIG_HASH_MISMATCH",
                            "evidence metadata does not identify the frozen configuration",
                            crop_id,
                        )
                    for field in CROP_CONTENT_HASH_FIELDS:
                        if (
                            evidence_row.get(field) != row.get(field)
                            or bundle.metadata.get(field) != evidence_row.get(field)
                        ):
                            _append(
                                findings,
                                "ERROR",
                                "EVIDENCE_CROP_CONTENT_LINEAGE",
                                f"manifest/index/bundle metadata disagree on {field}",
                                crop_id,
                            )
                except Exception as exc:
                    _append(findings, "ERROR", "EVIDENCE_CONTRACT", str(exc), crop_id)
        indexed_result = result_index_by_crop.get(crop_id)
        if indexed_result is not None and evidence_row is not None:
            indexed_status = str(indexed_result.get("status"))
            evidence_status = str(evidence_row.get("status"))
            if indexed_status in {"FINISHED", "RESUMED"} and (
                evidence_status != "PASS"
                or indexed_result.get("evidence_metadata_sha256")
                != evidence_row.get("metadata_sha256")
                or indexed_result.get("evidence_npz_sha256")
                != evidence_row.get("npz_sha256")
            ):
                _append(
                    findings,
                    "ERROR",
                    "RESULT_INDEX_EVIDENCE_LINEAGE",
                    "finished localization row does not bind PASS indexed evidence",
                    crop_id,
                )
            if evidence_status != "PASS" and indexed_status != "UPSTREAM_EVIDENCE_FAIL":
                _append(
                    findings,
                    "ERROR",
                    "UPSTREAM_EVIDENCE_STATUS_PROPAGATION",
                    f"evidence={evidence_status}, localization={indexed_status}",
                    crop_id,
                )
        prediction_entry = prediction_by_crop.get(crop_id)
        if prediction_entry is None:
            if not bool(row.get("eligible", True)):
                indexed = result_index_by_crop.get(crop_id)
                if result_index_path is not None and (
                    indexed is None
                    or str(indexed.get("status")) != "UPSTREAM_EVIDENCE_FAIL"
                ):
                    _append(
                        findings,
                        "ERROR",
                        "INELIGIBLE_RESULT_STATUS",
                        "manifest-ineligible crop must be registered as UPSTREAM_EVIDENCE_FAIL",
                        crop_id,
                    )
                continue
            _append(findings, "ERROR", "MISSING_RESULT", "no prediction JSON", crop_id)
            continue
        path, result = prediction_entry
        if result.get("config_hash") != expected_config_hash:
            _append(
                findings,
                "ERROR",
                "CONFIG_HASH_MISMATCH",
                f"expected {expected_config_hash}, got {result.get('config_hash')}",
                crop_id,
            )
        result_diagnostics = result.get("diagnostics", {})
        if evidence_row is not None and (
            result_diagnostics.get("evidence_metadata_sha256")
            != evidence_row.get("metadata_sha256")
            or result_diagnostics.get("evidence_npz_sha256")
            != evidence_row.get("npz_sha256")
        ):
            _append(
                findings,
                "ERROR",
                "RESULT_EVIDENCE_HASH_MISMATCH",
                "prediction does not identify the indexed evidence content",
                crop_id,
            )
        raw_norm = result.get("raw_center_norm")
        if raw_norm is not None:
            width, height = row["crop_size_px"]
            scale = float(row["normalization_scale_px"])
            origin_x, origin_y = row["crop_origin_source_px"]
            expected_crop = np.asarray(
                [
                    width / 2.0 + scale * float(raw_norm[0]),
                    height / 2.0 + scale * float(raw_norm[1]),
                ]
            )
            expected_source = expected_crop + np.asarray([origin_x, origin_y])
            if not np.allclose(
                expected_crop,
                np.asarray(result.get("raw_center_crop_px"), dtype=float),
                rtol=0.0,
                atol=1e-8,
            ) or not np.allclose(
                expected_source,
                np.asarray(result.get("raw_center_source_px"), dtype=float),
                rtol=0.0,
                atol=1e-8,
            ):
                _append(
                    findings,
                    "ERROR",
                    "RESULT_COORDINATE_ROUNDTRIP",
                    "normalized/crop/source center transforms disagree",
                    crop_id,
                )
        serialized = path.read_text(encoding="utf-8").lower()
        if "pith_gt" in serialized or "ground_truth" in serialized:
            _append(
                findings,
                "ERROR",
                "GT_IN_SOLVER_ARTIFACT",
                "prediction contains a GT-named field",
                crop_id,
            )
        if result.get("state") == "POINT":
            if not result.get("search_adequate", False):
                _append(findings, "ERROR", "FALSE_POINT_SEARCH", "POINT without adequate search", crop_id)
            if result.get("usable_center_norm") is None:
                _append(findings, "ERROR", "POINT_WITHOUT_CENTER", "POINT lacks usable center", crop_id)
            far = result.get("far_scan", {})
            if not far.get("stable", far.get("replay_stable", False)):
                _append(findings, "ERROR", "POINT_WITHOUT_FAR_REPLAY", "far scan replay unstable", crop_id)
        elif result.get("usable_center_norm") is not None:
            _append(
                findings,
                "ERROR",
                "NONPOINT_USABLE_CENTER",
                "non-POINT state exposes a usable finite coordinate",
                crop_id,
            )
        if uncertainty_index_path is not None:
            uncertainty_row = uncertainty_by_crop.get(crop_id)
            if uncertainty_row is None:
                _append(
                    findings,
                    "ERROR",
                    "MISSING_UNCERTAINTY",
                    "no structured uncertainty index row",
                    crop_id,
                )
            elif uncertainty_row.get("status") in {"FINISHED", "RESUMED"}:
                sidecar_path = uncertainty_row.get("uncertainty_path")
                try:
                    sidecar_file = Path(str(sidecar_path)).expanduser().resolve()
                    if not sidecar_file.is_relative_to(uncertainty_root):
                        raise ValueError("uncertainty sidecar escapes its run root")
                    if sha256_file(sidecar_file) != uncertainty_row.get(
                        "uncertainty_sha256"
                    ):
                        raise ValueError("uncertainty sidecar content hash changed")
                    if uncertainty_row.get("config_hash") != expected_config_hash:
                        raise ValueError("uncertainty index uses another configuration")
                    sidecar = _strict_json(sidecar_file)
                    if (
                        sidecar.get("schema_version") != "racpith.uncertainty.v1"
                        or str(sidecar.get("crop_id")) != crop_id
                        or str(sidecar.get("tree_id")) != str(row["tree_id"])
                        or str(sidecar.get("section_id")) != str(row["section_id"])
                        or str(sidecar.get("split")) != str(row["split"])
                        or str(sidecar.get("adjudicated_state"))
                        != str(uncertainty_row.get("adjudicated_state"))
                        or bool(sidecar.get("requires_search_reaudit", False))
                        != bool(
                            uncertainty_row.get("requires_search_reaudit", False)
                        )
                    ):
                        raise ValueError("uncertainty index/sidecar identity differs")
                    if sidecar.get("config_hash") != expected_config_hash:
                        _append(
                            findings,
                            "ERROR",
                            "UNCERTAINTY_CONFIG_HASH_MISMATCH",
                            "uncertainty sidecar uses another configuration",
                            crop_id,
                        )
                    if evidence_row is not None and (
                        sidecar.get("evidence_metadata_sha256")
                        != evidence_row.get("metadata_sha256")
                        or sidecar.get("evidence_npz_sha256")
                        != evidence_row.get("npz_sha256")
                        or uncertainty_row.get("evidence_metadata_sha256")
                        != evidence_row.get("metadata_sha256")
                        or uncertainty_row.get("evidence_npz_sha256")
                        != evidence_row.get("npz_sha256")
                    ):
                        _append(
                            findings,
                            "ERROR",
                            "UNCERTAINTY_EVIDENCE_HASH_MISMATCH",
                            "uncertainty index/sidecar does not bind indexed evidence",
                            crop_id,
                        )
                    if sidecar.get("baseline_result_sha256") != sha256_file(path):
                        _append(
                            findings,
                            "ERROR",
                            "UNCERTAINTY_BASELINE_HASH_MISMATCH",
                            "uncertainty sidecar does not reference immutable baseline",
                            crop_id,
                        )
                    if (
                        result_index_sha256 is not None
                        and sidecar.get("source_result_index_sha256")
                        != result_index_sha256
                    ):
                        _append(
                            findings,
                            "ERROR",
                            "UNCERTAINTY_SIDECAR_RESULT_INDEX_HASH",
                            "uncertainty sidecar does not bind the supplied localization index",
                            crop_id,
                        )
                    if (
                        sidecar.get("adjudicated_state") == "POINT"
                        and (
                            not sidecar.get("ellipse", {}).get("valid", False)
                            or sidecar.get("requires_search_reaudit", False)
                        )
                    ):
                        _append(
                            findings,
                            "ERROR",
                            "POINT_UNCERTAINTY_GATE",
                            "adjudicated POINT lacks a valid stable structured-replay region",
                            crop_id,
                        )
                    if (
                        result.get("production_usable", False)
                        and sidecar.get("adjudicated_state") != "POINT"
                    ):
                        _append(
                            findings,
                            "ERROR",
                            "PRODUCTION_USE_BEFORE_UNCERTAINTY_GATE",
                            "core marked result usable but structured replay downgraded it",
                            crop_id,
                        )
                except Exception as exc:
                    _append(
                        findings,
                        "ERROR",
                        "UNCERTAINTY_SIDECAR",
                        str(exc),
                        crop_id,
                    )
            else:
                _append(
                    findings,
                    "ERROR",
                    "UNCERTAINTY_NOT_FINISHED",
                    f"status={uncertainty_row.get('status')}",
                    crop_id,
                )
        if result.get("production_usable", False) and result.get("model_risk") not in {
            "LOW",
            "LOW_RISK",
        }:
            _append(
                findings,
                "ERROR",
                "MODEL_RISK_USE",
                "production_usable is true while model risk is not low",
                crop_id,
            )

    for crop_id in sorted(set(prediction_by_crop) - manifest_ids):
        _append(findings, "ERROR", "ORPHAN_PREDICTION", "not in crop manifest", crop_id)

    contribution_count = 0
    contribution_index_count = 0
    if contribution_root is not None:
        root = Path(contribution_root).expanduser().resolve()
        if contribution_index_path is None:
            _append(
                findings,
                "ERROR",
                "MISSING_CONTRIBUTION_INDEX",
                "formal contribution audit requires an index",
            )
            indexed_rows: list[dict[str, Any]] = []
        else:
            indexed_rows = read_jsonl(contribution_index_path)
        index_by_crop: dict[str, dict[str, Any]] = {}
        registered_files: dict[Path, dict[str, Any]] = {}
        seen_contribution_index: set[str] = set()
        for index_row in indexed_rows:
            if index_row.get("schema_version") != "racpith.contribution_index.v1":
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_INDEX_SCHEMA",
                    "unsupported contribution-index row",
                )
                continue
            crop_id = str(index_row["crop_id"])
            if crop_id in seen_contribution_index:
                _append(
                    findings,
                    "ERROR",
                    "DUPLICATE_CONTRIBUTION_INDEX",
                    "more than one contribution-index row",
                    crop_id,
                )
                continue
            seen_contribution_index.add(crop_id)
            full_source = full_manifest_by_crop.get(crop_id)
            if full_source is not None and any(
                str(index_row.get(field)) != str(full_source.get(field))
                for field in ("tree_id", "section_id", "split")
            ):
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_INDEX_LINEAGE",
                    "tree/section/split differs from crop manifest",
                    crop_id,
                )
            if splits and full_source is not None and crop_id not in selected_manifest_ids:
                continue
            contribution_index_count += 1
            index_by_crop[crop_id] = index_row
            if crop_id not in manifest_ids:
                _append(
                    findings,
                    "ERROR",
                    "ORPHAN_CONTRIBUTION_INDEX",
                    "crop is not in the selected manifest split",
                    crop_id,
                )
            status = str(index_row.get("status"))
            expected_manifest_ineligible = bool(
                crop_id in manifest_by_crop
                and not manifest_by_crop[crop_id].get("eligible", True)
            )
            accepted_status = status in {
                "PASS",
                "RESUMED",
                "SKIPPED_REGISTERED_SUBSAMPLE",
            } or (status == "MISSING_PREREQUISITE" and expected_manifest_ineligible)
            if not accepted_status:
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_CROP_FAILURE",
                    f"status={status}: {index_row.get('reason')}",
                    crop_id,
                )
            if status not in {"PASS", "RESUMED"}:
                continue
            if index_row.get("config_hash") != expected_config_hash:
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_INDEX_CONFIG_HASH",
                    "contribution index uses another configuration",
                    crop_id,
                )
            if (
                result_index_sha256 is not None
                and index_row.get("source_result_index_sha256")
                != result_index_sha256
            ):
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_RESULT_INDEX_HASH",
                    "contribution index does not bind the supplied localization index",
                    crop_id,
                )
            record_value = index_row.get("record_path")
            if record_value is None:
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_INDEX_RECORD_PATH",
                    "PASS row lacks record_path",
                    crop_id,
                )
                continue
            path = Path(str(record_value)).expanduser().resolve()
            if not path.is_relative_to(root):
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_PATH_ESCAPE",
                    str(path),
                    crop_id,
                )
                continue
            try:
                if sha256_file(path) != index_row.get("record_sha256"):
                    raise ValueError("record content hash differs from contribution index")
            except Exception as exc:
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_RECORD_HASH",
                    str(exc),
                    crop_id,
                )
                continue
            registered_files[path] = index_row

        for crop_id in sorted(manifest_ids - set(index_by_crop)):
            _append(
                findings,
                "ERROR",
                "MISSING_CONTRIBUTION_INDEX_ROW",
                "manifest crop has no contribution status",
                crop_id,
            )
        on_disk = (
            set((root / "per_crop").rglob("*.jsonl"))
            if (root / "per_crop").exists()
            else set()
        )
        scoped_unregistered = []
        for path in on_disk:
            resolved = path.resolve()
            file_crop_id = path.stem
            if (
                resolved not in registered_files
                and (
                    file_crop_id in manifest_ids
                    or file_crop_id not in full_manifest_by_crop
                )
            ):
                scoped_unregistered.append(resolved)
        for path in sorted(scoped_unregistered):
            _append(
                findings,
                "ERROR",
                "UNREGISTERED_CONTRIBUTION_FILE",
                f"stale or unindexed per-group file: {path}",
            )
        registered_minus_files: set[Path] = set()
        seen_group_ids: set[str] = set()
        for path, index_row in sorted(registered_files.items(), key=lambda item: str(item[0])):
            file_count = 0
            for record in read_jsonl(path):
                if record.get("schema_version") != "racpith.contribution.v1":
                    _append(
                        findings,
                        "ERROR",
                        "CONTRIBUTION_RECORD_SCHEMA",
                        str(path),
                        str(index_row.get("crop_id")),
                    )
                    continue
                if splits and str(record.get("split")) not in splits:
                    _append(
                        findings,
                        "ERROR",
                        "CONTRIBUTION_RECORD_SPLIT",
                        "registered file contains another split",
                        str(index_row.get("crop_id")),
                    )
                    continue
                file_count += 1
                contribution_count += 1
                crop_id = str(record["crop_id"])
                if crop_id != str(index_row.get("crop_id")):
                    _append(
                        findings,
                        "ERROR",
                        "CONTRIBUTION_RECORD_INDEX_ID",
                        "record crop_id differs from index",
                        crop_id,
                    )
                group_id = str(record.get("group_id"))
                if group_id in seen_group_ids:
                    _append(
                        findings,
                        "ERROR",
                        "DUPLICATE_CONTRIBUTION_GROUP",
                        group_id,
                        crop_id,
                    )
                seen_group_ids.add(group_id)
                prediction_entry = prediction_by_crop.get(crop_id)
                if prediction_entry is None:
                    _append(findings, "ERROR", "ORPHAN_CONTRIBUTION", str(path), crop_id)
                    continue
                baseline_path, baseline = prediction_entry
                if record.get("baseline_result_sha256") != sha256_file(baseline_path):
                    _append(
                        findings,
                        "ERROR",
                        "BASELINE_HASH_MISMATCH",
                        "contribution does not reference the immutable baseline",
                        crop_id,
                    )
                if record.get("baseline_result_digest") != _locate_result_digest(baseline):
                    _append(
                        findings,
                        "ERROR",
                        "BASELINE_DIGEST_MISMATCH",
                        "contribution baseline model digest differs",
                        crop_id,
                    )
                if record.get("frozen_config_hash") != expected_config_hash:
                    _append(
                        findings,
                        "ERROR",
                        "CONTRIBUTION_CONFIG_HASH_MISMATCH",
                        "per-group record uses another configuration",
                        crop_id,
                    )
                if (
                    result_index_sha256 is not None
                    and record.get("source_result_index_sha256")
                    != result_index_sha256
                ):
                    _append(
                        findings,
                        "ERROR",
                        "CONTRIBUTION_RECORD_RESULT_INDEX_HASH",
                        "per-group record does not bind the supplied localization index",
                        crop_id,
                    )
                evidence_row = evidence_index.get(crop_id)
                if evidence_row is not None and (
                    record.get("evidence_metadata_sha256")
                    != evidence_row.get("metadata_sha256")
                    or record.get("evidence_npz_sha256")
                    != evidence_row.get("npz_sha256")
                ):
                    _append(
                        findings,
                        "ERROR",
                        "CONTRIBUTION_EVIDENCE_HASH_MISMATCH",
                        "per-group record does not bind the indexed evidence",
                        crop_id,
                    )
                gt_label = record.get("gt_label")
                if gt_label is not None and (
                    baseline.get("state") != "POINT" or record.get("minus_state") != "POINT"
                ):
                    _append(
                        findings,
                        "ERROR",
                        "GT_SIGN_STATE_GATE",
                        "signed GT contribution exists outside POINT-to-POINT gate",
                        crop_id,
                    )
                minus_dir = (root / "minus_results" / crop_id).resolve()
                minus_path = (minus_dir / f"{group_id}.json").resolve()
                if not minus_path.is_relative_to(minus_dir):
                    _append(
                        findings,
                        "ERROR",
                        "MINUS_RESULT_PATH_ESCAPE",
                        str(minus_path),
                        crop_id,
                    )
                    continue
                registered_minus_files.add(minus_path)
                try:
                    minus_payload = _strict_json(minus_path)
                    if minus_payload.get("config_hash") != expected_config_hash:
                        raise ValueError("minus result config hash differs")
                    if minus_payload.get("crop_id") != crop_id:
                        raise ValueError("minus result crop_id differs")
                    if record.get("minus_result_digest") != _locate_result_digest(
                        minus_payload
                    ):
                        raise ValueError("minus result digest differs")
                except Exception as exc:
                    _append(
                        findings,
                        "ERROR",
                        "MINUS_RESULT_ARTIFACT",
                        str(exc),
                        crop_id,
                    )
                conflict = record.get("conflict_cost")
                if conflict is not None and float(conflict) < -1e-8:
                    _append(
                        findings,
                        "ERROR",
                        "NEGATIVE_CONFLICT",
                        f"conflict cost {conflict} violates refit optimality",
                        crop_id,
                    )
                direct = record.get("contrib_gt_sq")
                identity = record.get("contrib_gt_sq_identity")
                if direct is not None and identity is not None and not np.isclose(
                    float(direct), float(identity), rtol=1e-7, atol=1e-10
                ):
                    _append(
                        findings,
                        "ERROR",
                        "CONTRIBUTION_IDENTITY",
                        f"direct={direct}, identity={identity}",
                        crop_id,
                    )
            if file_count != int(index_row.get("groups", -1)):
                _append(
                    findings,
                    "ERROR",
                    "CONTRIBUTION_GROUP_COUNT",
                    f"index={index_row.get('groups')}, file={file_count}",
                    str(index_row.get("crop_id")),
                )

        minus_root = root / "minus_results"
        if minus_root.exists():
            for path in sorted(minus_root.rglob("*.json")):
                resolved = path.resolve()
                try:
                    relative = resolved.relative_to(minus_root.resolve())
                except ValueError:
                    _append(
                        findings,
                        "ERROR",
                        "MINUS_RESULT_PATH_ESCAPE",
                        str(resolved),
                    )
                    continue
                file_crop_id = relative.parts[0] if len(relative.parts) >= 2 else ""
                if (
                    resolved not in registered_minus_files
                    and (
                        file_crop_id in manifest_ids
                        or file_crop_id not in full_manifest_by_crop
                    )
                ):
                    _append(
                        findings,
                        "ERROR",
                        "UNREGISTERED_MINUS_RESULT",
                        f"stale or unindexed delete-refit artifact: {resolved}",
                        file_crop_id or None,
                    )

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.check] = counts.get(finding.check, 0) + 1
    return {
        "schema_version": "racpith.audit.v1",
        "status": "FAIL" if any(item.severity == "ERROR" for item in findings) else "PASS",
        "denominator": {
            "manifest_crops": len(manifest),
            "evidence_records": len(evidence_index),
            "predictions": len(prediction_by_crop),
            "result_index_records": len(result_index_by_crop),
            "contribution_records": contribution_count,
            "contribution_index_records": contribution_index_count,
            "uncertainty_records": len(uncertainty_by_crop),
            "target_domain_records": target_domain_count,
            "target_domain_index_sha256": target_domain_index_sha256,
        },
        "finding_counts": counts,
        "findings": [asdict(item) for item in findings],
    }
