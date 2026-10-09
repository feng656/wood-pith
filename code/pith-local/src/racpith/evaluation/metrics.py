from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..contracts import EvidenceBundle, LocateResult
from ..estimator import RacPithEstimator
from ..provenance import is_sha256, read_json_object, read_jsonl, sha256_file


USABLE_STATES = {"POINT", "RANGE", "RAY", "AXIS"}
CROP_CONTENT_HASH_FIELDS = ("crop_image_sha256", "crop_annotation_sha256")


def _strict_json_object(path: Path) -> dict[str, Any]:
    return read_json_object(path)


def _finite_scalar(value: object) -> bool:
    """Return whether a persisted scalar is finite numeric data (not bool)."""

    if isinstance(value, bool) or value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def validate_manifest_crop_content(row: Mapping[str, Any]) -> None:
    """Fail closed unless a manifest row still names the exact crop inputs."""

    crop_id = str(row.get("crop_id"))
    for path_field, hash_field in (
        ("crop_image_path", "crop_image_sha256"),
        ("crop_annotation_path", "crop_annotation_sha256"),
    ):
        if path_field not in row or not is_sha256(row.get(hash_field)):
            raise ValueError(f"manifest crop {crop_id} lacks valid {path_field}/{hash_field}")
        path = Path(str(row[path_field])).expanduser()
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"manifest crop {crop_id} has invalid {path_field}: {path}")
        if sha256_file(path) != row[hash_field]:
            raise ValueError(f"manifest crop input content changed: {crop_id}/{path_field}")


def validate_evidence_bundle_lineage(
    row: Mapping[str, Any],
    bundle: EvidenceBundle,
    *,
    expected_config_hash: str | None = None,
) -> None:
    """Bind an in-memory bundle to its evidence-index crop-content lineage."""

    crop_id = str(row.get("crop_id"))
    if (
        bundle.crop_id != crop_id
        or bundle.tree_id != str(row.get("tree_id"))
        or bundle.section_id != str(row.get("section_id"))
    ):
        raise ValueError(f"evidence bundle identities differ from index row: {crop_id}")
    for field in CROP_CONTENT_HASH_FIELDS:
        expected = row.get(field)
        if not is_sha256(expected):
            raise ValueError(f"evidence index has invalid {field}: {crop_id}")
        if bundle.metadata.get(field) != expected:
            raise ValueError(f"evidence bundle/index {field} mismatch: {crop_id}")
    if (
        expected_config_hash is not None
        and bundle.metadata.get("evidence_config_hash") != expected_config_hash
    ):
        raise ValueError(f"evidence bundle/config hash mismatch: {crop_id}")


def _angle_deg(a: np.ndarray, b: np.ndarray, axis: bool) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    cosine = float(np.dot(a, b) / (na * nb))
    if axis:
        cosine = abs(cosine)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _gt_norm(row: Mapping[str, Any]) -> np.ndarray:
    if row.get("pith_crop_px") is None:
        return np.asarray([np.nan, np.nan])
    pith = np.asarray(row["pith_crop_px"], dtype=np.float64)
    width, height = row["crop_size_px"]
    d_fov = float(row.get("normalization_scale_px", np.hypot(width, height)))
    return (pith - np.asarray([width / 2.0, height / 2.0])) / d_fov


def _outside_distance_norm(row: Mapping[str, Any]) -> float:
    pith = np.asarray(row["pith_crop_px"], dtype=np.float64)
    width, height = row["crop_size_px"]
    dx = max(-pith[0], 0.0, pith[0] - width)
    dy = max(-pith[1], 0.0, pith[1] - height)
    return float(np.hypot(dx, dy) / float(row.get("normalization_scale_px", np.hypot(width, height))))


def load_predictions(
    prediction_root: str | Path,
    *,
    result_index_path: str | Path | None = None,
    expected_config_hash: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load localization artifacts, preferably through their immutable result index."""

    root = Path(prediction_root).expanduser().resolve()
    scanned: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = _strict_json_object(path)
        except (OSError, ValueError):
            continue
        if payload.get("schema_version") != "racpith.locate.v1":
            continue
        try:
            validated = LocateResult.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid localization result artifact: {path}: {exc}") from exc
        crop_id = validated.crop_id
        if crop_id in scanned:
            raise ValueError(f"duplicate prediction for crop {crop_id}")
        scanned[crop_id] = (path.resolve(), payload)
    if result_index_path is None:
        predictions: dict[str, dict[str, Any]] = {}
        for crop_id, (path, payload) in scanned.items():
            if expected_config_hash is not None and payload.get("config_hash") != expected_config_hash:
                raise ValueError(f"prediction/config hash mismatch for crop {crop_id}")
            predictions[crop_id] = {**payload, "_path": str(path)}
        return predictions, {}

    result_index: dict[str, dict[str, Any]] = {}
    predictions = {}
    registered_paths: set[Path] = set()
    allowed_statuses = {"FINISHED", "RESUMED", "CRASH", "UPSTREAM_EVIDENCE_FAIL"}
    for row in read_jsonl(result_index_path):
        if row.get("schema_version") != "racpith.result_index.v1":
            raise ValueError("unsupported result-index schema")
        crop_id = str(row["crop_id"])
        if crop_id in result_index:
            raise ValueError(f"duplicate result-index crop {crop_id}")
        result_index[crop_id] = row
        if expected_config_hash is not None and row.get("config_hash") != expected_config_hash:
            raise ValueError(f"result-index/config hash mismatch for crop {crop_id}")
        status = str(row.get("status"))
        if status not in allowed_statuses:
            raise ValueError(f"unsupported result-index status {status!r} for crop {crop_id}")
        if status not in {"FINISHED", "RESUMED"}:
            if row.get("prediction_path") is not None:
                raise ValueError(f"failed result-index row unexpectedly has a prediction: {crop_id}")
            continue
        path_value = row.get("prediction_path")
        if path_value is None:
            raise ValueError(f"finished result-index row lacks prediction_path: {crop_id}")
        path = Path(str(path_value)).expanduser().resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"registered prediction escapes prediction root: {path}")
        scanned_entry = scanned.get(crop_id)
        if scanned_entry is None or scanned_entry[0] != path:
            raise ValueError(f"registered prediction is missing or crop/path disagrees: {crop_id}")
        if sha256_file(path) != row.get("prediction_sha256"):
            raise ValueError(f"registered prediction hash changed: {crop_id}")
        payload = scanned_entry[1]
        if payload.get("config_hash") != row.get("config_hash"):
            raise ValueError(f"prediction/result-index config mismatch: {crop_id}")
        diagnostics = payload.get("diagnostics", {})
        if (
            diagnostics.get("evidence_metadata_sha256")
            != row.get("evidence_metadata_sha256")
            or diagnostics.get("evidence_npz_sha256") != row.get("evidence_npz_sha256")
        ):
            raise ValueError(f"prediction/result-index evidence lineage mismatch: {crop_id}")
        predictions[crop_id] = {**payload, "_path": str(path)}
        registered_paths.add(path)
    unregistered = sorted(
        str(path)
        for path, _ in scanned.values()
        if path not in registered_paths
    )
    if unregistered:
        raise ValueError(f"unregistered/stale prediction artifacts: {unregistered[:10]}")
    return predictions, result_index


def load_evidence_index(
    evidence_index_path: str | Path,
    *,
    expected_config_hash: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load and content-verify the evidence registry used for post-hoc scoring."""

    root = Path(evidence_index_path).expanduser().resolve().parent
    result: dict[str, dict[str, Any]] = {}
    registered_paths: set[Path] = set()
    for row in read_jsonl(evidence_index_path):
        if row.get("schema_version") != "racpith.evidence_index.v1":
            raise ValueError("unsupported evidence-index schema")
        crop_id = str(row["crop_id"])
        if crop_id in result:
            raise ValueError(f"duplicate evidence-index crop {crop_id}")
        if expected_config_hash is not None and row.get("config_hash") != expected_config_hash:
            raise ValueError(f"evidence/config hash mismatch for crop {crop_id}")
        if row.get("status") not in {"PASS", "FAIL", "MANIFEST_INELIGIBLE"}:
            raise ValueError(
                f"unsupported evidence-index status for crop {crop_id}: {row.get('status')!r}"
            )
        for field in CROP_CONTENT_HASH_FIELDS:
            if not is_sha256(row.get(field)):
                raise ValueError(f"evidence index has invalid {field}: {crop_id}")
        if row.get("status") == "PASS":
            artifact_paths: dict[str, Path] = {}
            for path_field, hash_field in (
                ("metadata_path", "metadata_sha256"),
                ("npz_path", "npz_sha256"),
            ):
                value = row.get(path_field)
                if value is None:
                    raise ValueError(f"PASS evidence row lacks {path_field}: {crop_id}")
                path = Path(str(value)).expanduser().resolve()
                if not path.is_relative_to(root):
                    raise ValueError(f"registered evidence escapes run root: {path}")
                if sha256_file(path) != row.get(hash_field):
                    raise ValueError(f"registered evidence content changed: {crop_id}/{path_field}")
                artifact_paths[path_field] = path
                registered_paths.add(path)
            payload = _strict_json_object(artifact_paths["metadata_path"])
            metadata = payload.get("metadata")
            if (
                payload.get("schema_version") != "racpith.evidence.v1"
                or str(payload.get("crop_id")) != crop_id
                or str(payload.get("tree_id")) != str(row.get("tree_id"))
                or str(payload.get("section_id")) != str(row.get("section_id"))
                or not isinstance(metadata, Mapping)
                or any(
                    metadata.get(field) != row.get(field)
                    for field in CROP_CONTENT_HASH_FIELDS
                )
                or metadata.get("evidence_config_hash") != row.get("config_hash")
            ):
                raise ValueError(f"evidence metadata/index lineage mismatch: {crop_id}")
            npz_name = Path(str(payload.get("npz_path")))
            if (
                npz_name.is_absolute()
                or npz_name.name != str(npz_name)
                or (artifact_paths["metadata_path"].parent / npz_name).resolve()
                != artifact_paths["npz_path"]
            ):
                raise ValueError(f"evidence metadata/index NPZ path mismatch: {crop_id}")
        elif any(
            row.get(field) is not None
            for field in ("metadata_path", "npz_path", "metadata_sha256", "npz_sha256")
        ):
            raise ValueError(f"non-PASS evidence row exposes artifacts: {crop_id}")
        result[crop_id] = row
    per_crop = root / "per_crop"
    on_disk = (
        {
            path.resolve()
            for suffix in ("*.json", "*.npz")
            for path in per_crop.glob(suffix)
        }
        if per_crop.is_dir()
        else set()
    )
    unregistered = sorted(str(path) for path in on_disk - registered_paths)
    if unregistered:
        raise ValueError(f"unregistered/stale evidence artifacts: {unregistered[:10]}")
    return result


def _ellipse_contains_gt(sidecar: Mapping[str, Any], gt_norm: np.ndarray) -> bool | float:
    ellipse = sidecar.get("ellipse", {})
    if not isinstance(ellipse, Mapping) or not bool(ellipse.get("valid", False)):
        return float("nan")
    center = np.asarray(ellipse.get("center_median_norm"), dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        return float("nan")
    major = float(ellipse.get("major_semi_axis_norm", np.nan))
    minor = float(ellipse.get("minor_semi_axis_norm", np.nan))
    angle = float(ellipse.get("major_axis_angle_rad", np.nan))
    if not np.all(np.isfinite([major, minor, angle])) or major < 0.0 or minor < 0.0:
        return float("nan")
    delta = gt_norm - center
    major_axis = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
    minor_axis = np.asarray([-major_axis[1], major_axis[0]], dtype=np.float64)
    projections = np.asarray(
        [float(np.dot(delta, major_axis)), float(np.dot(delta, minor_axis))]
    )
    axes = np.asarray([major, minor], dtype=np.float64)
    zero = axes <= 64.0 * np.finfo(np.float64).eps
    if np.any(zero & (np.abs(projections) > 1e-10)):
        return False
    score = float(np.sum(np.square(projections[~zero] / axes[~zero])))
    return bool(score <= 1.0 + 1e-10)


def build_crop_metrics(
    crop_manifest_path: str | Path,
    prediction_root: str | Path,
    false_point_tolerance_norm: float,
    catastrophic_tolerance_norm: float,
    uncertainty_index_path: str | Path | None = None,
    splits: set[str] | None = None,
    target_domain_index_path: str | Path | None = None,
    expected_config_hash: str | None = None,
    result_index_path: str | Path | None = None,
    evidence_index_path: str | Path | None = None,
    estimator: RacPithEstimator | None = None,
) -> pd.DataFrame:
    all_rows = read_jsonl(crop_manifest_path)
    all_manifest_ids: set[str] = set()
    for row in all_rows:
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported crop-manifest schema")
        crop_id = str(row["crop_id"])
        if crop_id in all_manifest_ids:
            raise ValueError(f"duplicate crop_id in manifest: {crop_id}")
        all_manifest_ids.add(crop_id)
    rows = all_rows
    if splits:
        rows = [row for row in rows if str(row["split"]) in splits]
    if not rows:
        raise ValueError("selected crop-manifest scope is empty")
    for row in rows:
        validate_manifest_crop_content(row)
    predictions, result_index = load_predictions(
        prediction_root,
        result_index_path=result_index_path,
        expected_config_hash=expected_config_hash,
    )
    result_index_sha256 = (
        sha256_file(result_index_path) if result_index_path is not None else None
    )
    if (evidence_index_path is None) != (estimator is None):
        raise ValueError(
            "evidence_index_path and estimator must be supplied together for support scoring"
        )
    evidence_index = (
        load_evidence_index(
            evidence_index_path,
            expected_config_hash=expected_config_hash,
        )
        if evidence_index_path is not None
        else {}
    )
    unexpected_evidence = sorted(set(evidence_index) - all_manifest_ids)
    if unexpected_evidence:
        raise ValueError(f"evidence-index crops not present in manifest: {unexpected_evidence[:10]}")
    uncertainty: dict[str, dict[str, Any]] | None = None
    if uncertainty_index_path is not None:
        uncertainty = {}
        uncertainty_root = Path(uncertainty_index_path).expanduser().resolve().parent
        registered_uncertainty_paths: set[Path] = set()
        for item in read_jsonl(uncertainty_index_path):
            if item.get("schema_version") != "racpith.uncertainty_index.v1":
                raise ValueError("unsupported uncertainty-index schema")
            crop_id = str(item["crop_id"])
            if crop_id in uncertainty:
                raise ValueError(f"duplicate uncertainty record for crop {crop_id}")
            if (
                expected_config_hash is not None
                and item.get("config_hash") != expected_config_hash
            ):
                raise ValueError(f"uncertainty/config hash mismatch for crop {crop_id}")
            if (
                result_index_sha256 is not None
                and item.get("source_result_index_sha256") != result_index_sha256
            ):
                raise ValueError(
                    f"uncertainty/localization result-index hash mismatch for crop {crop_id}"
                )
            if item.get("status") not in {
                "FINISHED",
                "RESUMED",
                "FAIL",
                "UPSTREAM_LOCALIZATION_FAIL",
                "UPSTREAM_EVIDENCE_FAIL",
            }:
                raise ValueError(
                    f"unsupported uncertainty-index status for crop {crop_id}: "
                    f"{item.get('status')!r}"
                )
            if item.get("status") in {"FINISHED", "RESUMED"}:
                sidecar_value = item.get("uncertainty_path")
                if sidecar_value is None:
                    raise ValueError(f"finished uncertainty row lacks sidecar: {crop_id}")
                sidecar_path = Path(str(sidecar_value)).expanduser().resolve()
                if not sidecar_path.is_relative_to(uncertainty_root):
                    raise ValueError(f"uncertainty sidecar escapes its run root: {crop_id}")
                if sha256_file(sidecar_path) != item.get("uncertainty_sha256"):
                    raise ValueError(f"uncertainty sidecar hash changed: {crop_id}")
                registered_uncertainty_paths.add(sidecar_path)
                sidecar = _strict_json_object(sidecar_path)
                if sidecar.get("schema_version") != "racpith.uncertainty.v1":
                    raise ValueError(f"unsupported uncertainty sidecar schema: {crop_id}")
                if (
                    str(sidecar.get("crop_id")) != crop_id
                    or str(sidecar.get("tree_id")) != str(item.get("tree_id"))
                    or str(sidecar.get("section_id")) != str(item.get("section_id"))
                    or sidecar.get("config_hash") != item.get("config_hash")
                    or sidecar.get("baseline_result_sha256")
                    != item.get("baseline_result_sha256")
                    or sidecar.get("source_result_index_sha256")
                    != item.get("source_result_index_sha256")
                    or sidecar.get("evidence_metadata_sha256")
                    != item.get("evidence_metadata_sha256")
                    or sidecar.get("evidence_npz_sha256")
                    != item.get("evidence_npz_sha256")
                    or str(sidecar.get("adjudicated_state"))
                    != str(item.get("adjudicated_state"))
                    or bool(sidecar.get("requires_search_reaudit", False))
                    != bool(item.get("requires_search_reaudit", False))
                ):
                    raise ValueError(f"uncertainty index/sidecar mismatch: {crop_id}")
                item = {**item, "_sidecar": sidecar}
            elif item.get("uncertainty_path") is not None:
                raise ValueError(
                    f"non-finished uncertainty row exposes a sidecar: {crop_id}"
                )
            uncertainty[crop_id] = item
        uncertainty_artifact_root = uncertainty_root / "per_crop"
        on_disk_uncertainty = (
            {
                path.resolve()
                for path in uncertainty_artifact_root.rglob("*.json")
                if path.is_file()
            }
            if uncertainty_artifact_root.is_dir()
            else set()
        )
        stale_uncertainty = sorted(
            str(path)
            for path in on_disk_uncertainty - registered_uncertainty_paths
        )
        if stale_uncertainty:
            raise ValueError(
                f"unregistered/stale uncertainty artifacts: {stale_uncertainty[:10]}"
            )
        unexpected_uncertainty = sorted(set(uncertainty) - all_manifest_ids)
        if unexpected_uncertainty:
            raise ValueError(
                f"uncertainty-index crops not present in manifest: {unexpected_uncertainty[:10]}"
            )
    target_by_section: dict[str, dict[str, Any]] = {}
    if target_domain_index_path is not None:
        target_provenance: set[tuple[str, str]] = set()
        expected_alignment_tolerance = (
            float(estimator.config.section("target_domain")["alignment_tolerance_norm"])
            if estimator is not None
            else None
        )
        for item in read_jsonl(target_domain_index_path):
            if item.get("schema_version") != "racpith.target_domain.v1":
                raise ValueError("unsupported target-domain schema")
            section_id = str(item["section_id"])
            if section_id in target_by_section:
                raise ValueError(f"duplicate target-domain record for section {section_id}")
            if (
                expected_config_hash is not None
                and item.get("config_hash") != expected_config_hash
            ):
                raise ValueError(
                    f"target-domain record uses another configuration: {section_id}"
                )
            if item.get("gt_used_only_after_reference_fit") is not True:
                raise ValueError(
                    f"target-domain record lacks the post-hoc GT-use certificate: {section_id}"
                )
            reference_crop_id = item.get("reference_crop_id")
            if not isinstance(reference_crop_id, str) or not reference_crop_id.strip():
                raise ValueError(
                    f"target-domain record lacks a non-empty reference crop id: {section_id}"
                )
            manifest_hash = item.get("reference_manifest_sha256")
            result_hash = item.get("reference_result_index_sha256")
            if not all(is_sha256(value) for value in (manifest_hash, result_hash)):
                raise ValueError(
                    f"target-domain record has invalid reference hashes: {section_id}"
                )
            target_provenance.add((str(manifest_hash), str(result_hash)))
            reference_status = str(item.get("reference_result_status"))
            reference_prediction_hash = item.get("reference_prediction_sha256")
            if reference_status in {"FINISHED", "RESUMED"}:
                if not is_sha256(reference_prediction_hash):
                    raise ValueError(
                        f"finished target-domain reference lacks prediction hash: {section_id}"
                    )
            elif reference_prediction_hash is not None:
                raise ValueError(
                    f"non-finished target-domain reference exposes prediction hash: {section_id}"
                )
            target_label = str(item.get("target_domain"))
            if target_label not in {
                "TARGET_ALIGNED",
                "ECCENTRIC_GT",
                "TARGET_UNKNOWN",
            }:
                raise ValueError(
                    f"unsupported target-domain label {target_label!r}: {section_id}"
                )
            if expected_alignment_tolerance is not None and not np.isclose(
                float(item.get("alignment_tolerance_norm", np.nan)),
                expected_alignment_tolerance,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"target-domain alignment threshold differs from configuration: {section_id}"
                )
            reference_scale = item.get("reference_normalization_scale_px")
            if not _finite_scalar(reference_scale) or float(reference_scale) <= 0.0:
                raise ValueError(
                    f"target-domain record has invalid reference scale: {section_id}"
                )
            bias = item.get("target_bias_norm")
            bias_px = item.get("target_bias_px")
            if target_label == "TARGET_UNKNOWN":
                if bias is not None or bias_px is not None:
                    raise ValueError(
                        f"TARGET_UNKNOWN unexpectedly exposes a target bias: {section_id}"
                    )
            elif not _finite_scalar(bias) or not _finite_scalar(bias_px):
                raise ValueError(
                    f"resolved target-domain record lacks finite bias: {section_id}"
                )
            elif float(bias) < 0.0 or float(bias_px) < 0.0:
                raise ValueError(
                    f"resolved target-domain record has negative bias: {section_id}"
                )
            elif not np.isclose(
                float(bias_px),
                float(bias) * float(reference_scale),
                rtol=1e-9,
                atol=1e-8,
            ):
                raise ValueError(
                    f"target-domain pixel/normalized bias mismatch: {section_id}"
                )
            target_by_section[section_id] = item
        if len(target_provenance) != 1:
            raise ValueError(
                "target-domain index must bind exactly one reference manifest/result index pair"
            )
        manifest_sections = {str(row["section_id"]) for row in all_rows}
        unexpected_target = sorted(set(target_by_section) - manifest_sections)
        if unexpected_target:
            raise ValueError(
                f"target-domain sections not present in crop manifest: {unexpected_target[:10]}"
            )
        selected_sections = {str(row["section_id"]) for row in rows}
        missing_target = sorted(selected_sections - set(target_by_section))
        if missing_target:
            raise ValueError(
                f"selected sections lack target-domain status rows: {missing_target[:10]}"
            )
    metrics: list[dict[str, Any]] = []
    for row in rows:
        crop_id = str(row["crop_id"])
        gt = _gt_norm(row)
        target = target_by_section.get(str(row["section_id"]), {})
        if target and (
            str(target.get("tree_id")) != str(row["tree_id"])
            or str(target.get("split")) != str(row["split"])
        ):
            raise ValueError(
                f"target-domain/crop-manifest lineage mismatch for section {row['section_id']}"
            )
        result = predictions.get(crop_id)
        index_row = result_index.get(crop_id) if result_index_path is not None else None
        evidence_row = evidence_index.get(crop_id) if evidence_index_path is not None else None
        if evidence_row is not None and any(
            str(evidence_row.get(field)) != str(row.get(field))
            for field in (
                "tree_id",
                "section_id",
                "split",
                *CROP_CONTENT_HASH_FIELDS,
            )
        ):
            raise ValueError(f"evidence-index/manifest lineage mismatch for crop {crop_id}")
        if result is not None and (
            evidence_row is None or evidence_row.get("status") != "PASS"
        ):
            raise ValueError(f"registered prediction lacks PASS evidence for crop {crop_id}")
        if result is not None and index_row is not None and evidence_row is not None and (
            index_row.get("evidence_metadata_sha256")
            != evidence_row.get("metadata_sha256")
            or index_row.get("evidence_npz_sha256") != evidence_row.get("npz_sha256")
        ):
            raise ValueError(f"result/evidence-index content lineage mismatch for crop {crop_id}")
        if index_row is not None and any(
            str(index_row.get(field)) != str(row.get(field))
            for field in ("tree_id", "section_id", "split")
        ):
            raise ValueError(f"result-index/manifest lineage mismatch for crop {crop_id}")
        base = {
            "crop_id": crop_id,
            "tree_id": str(row["tree_id"]),
            "section_id": str(row["section_id"]),
            "split": row["split"],
            "distance_stratum": row.get("distance_stratum"),
            "pith_outside_distance_norm": _outside_distance_norm(row),
            "visible_parent_rings": row.get("visible_parent_rings"),
            "manifest_eligible": bool(row.get("eligible", True)),
            "target_domain": target.get("target_domain", "TARGET_UNKNOWN"),
            "target_bias_norm": target.get("target_bias_norm"),
            "target_bias_px": target.get("target_bias_px"),
            "target_reference_manifest_sha256": target.get(
                "reference_manifest_sha256"
            ),
            "target_reference_result_index_sha256": target.get(
                "reference_result_index_sha256"
            ),
        }
        if result is None:
            uncertainty_row = uncertainty.get(crop_id) if uncertainty is not None else None
            indexed_status = (
                str(index_row.get("status"))
                if index_row is not None
                else ("MISSING_RESULT_INDEX" if result_index_path is not None else "MISSING_RESULT")
            )
            metrics.append(
                {
                    **base,
                    "result_status": indexed_status,
                    "result_index_status": indexed_status,
                    "state": "REJECT",
                    "geometry_state": "REJECT",
                    "uncertainty_status": (
                        str(uncertainty_row.get("status"))
                        if uncertainty_row is not None
                        else "NOT_REQUESTED"
                    ),
                    "uncertainty_reaudit_required": bool(
                        uncertainty_row.get("requires_search_reaudit", True)
                        if uncertainty_row is not None
                        else False
                    ),
                    "reason_codes": str(
                        index_row.get("reason")
                        if index_row is not None and index_row.get("reason")
                        else indexed_status
                    ),
                    "search_adequate": False,
                    "production_usable": False,
                    "point_error_norm": np.nan,
                    "point_error_px": np.nan,
                    "false_point": False,
                    "catastrophic_point": False,
                    "direction_error_deg": np.nan,
                    "range_contains_gt": np.nan,
                    "radial_interval_contains_gt": np.nan,
                    "core_support_contains_gt": np.nan,
                    "support_contains_gt": np.nan,
                    "gt_profiled_objective": np.nan,
                    "support_threshold": np.nan,
                    "stability_ellipse_contains_gt": np.nan,
                    "stability_ellipse_major_norm": np.nan,
                    "stability_ellipse_minor_norm": np.nan,
                    "range_lower_norm": np.nan,
                    "range_upper_norm": np.nan,
                    "range_width_norm": np.nan,
                    "range_unbounded": False,
                    "finite_mode_count": 0,
                    "support_component_count": 0,
                    "far_gap": np.nan,
                    "point_safety_margin": np.nan,
                    "condition_ratio": np.nan,
                    "model_risk": "UNKNOWN",
                    "prediction_path": None,
                }
            )
            continue
        if expected_config_hash is not None and result.get("config_hash") != expected_config_hash:
            raise ValueError(f"prediction/config hash mismatch for crop {crop_id}")
        geometry_state = str(result["state"])
        uncertainty_row = uncertainty.get(crop_id) if uncertainty is not None else None
        if uncertainty is not None and uncertainty_row is None:
            raise ValueError(f"uncertainty index lacks manifest crop {crop_id}")
        uncertainty_status = (
            str(uncertainty_row.get("status")) if uncertainty_row is not None else "NOT_REQUESTED"
        )
        uncertainty_complete = uncertainty_status in {
            "FINISHED",
            "RESUMED",
            "NOT_REQUESTED",
        }
        if (
            uncertainty_row is not None
            and expected_config_hash is not None
            and uncertainty_row.get("config_hash") != expected_config_hash
        ):
            raise ValueError(f"uncertainty/config hash mismatch for crop {crop_id}")
        if uncertainty_row is not None and uncertainty_complete and uncertainty_status != "NOT_REQUESTED":
            if uncertainty_row.get("baseline_result_sha256") != sha256_file(result["_path"]):
                raise ValueError(f"uncertainty/baseline hash mismatch for crop {crop_id}")
        state = (
            str(uncertainty_row.get("adjudicated_state", "REJECT"))
            if uncertainty_row is not None
            else geometry_state
        )
        if not uncertainty_complete:
            state = "REJECT"
        raw = result.get("raw_center_norm")
        raw_center = np.asarray(raw, dtype=np.float64) if raw is not None else np.asarray([np.nan, np.nan])
        if state == "POINT" and raw is None:
            raise ValueError(f"adjudicated POINT has no finite coordinate for crop {crop_id}")
        error_norm = float(np.linalg.norm(raw_center - gt)) if state == "POINT" else float("nan")
        d_fov = float(row.get("normalization_scale_px", np.hypot(*row["crop_size_px"])))
        error_px = error_norm * d_fov if np.isfinite(error_norm) else float("nan")
        direction = result.get("direction_norm")
        direction_error = float("nan")
        if direction is not None:
            direction_error = _angle_deg(
                np.asarray(direction, dtype=np.float64), gt, axis=state == "AXIS"
            )
        interval = result.get("range_interval_norm")
        radial_contains: bool | float = float("nan")
        range_lower = float("nan")
        range_upper = float("nan")
        range_width = float("nan")
        range_unbounded = False
        if interval is not None:
            lower = float(interval[0]) if interval[0] is not None else 0.0
            upper = float(interval[1]) if interval[1] is not None else np.inf
            range_lower = lower
            range_upper = upper
            range_unbounded = not np.isfinite(upper)
            range_width = upper - lower
            radial_contains = bool(lower <= np.linalg.norm(gt) <= upper)

        gt_objective = float("nan")
        support_threshold = float("nan")
        core_support_contains: bool | float = float("nan")
        if evidence_index_path is not None and raw is not None:
            if evidence_row is None or evidence_row.get("status") != "PASS":
                raise ValueError(f"finished prediction lacks PASS evidence row: {crop_id}")
            bundle = EvidenceBundle.load(evidence_row["metadata_path"])
            validate_evidence_bundle_lineage(
                evidence_row,
                bundle,
                expected_config_hash=expected_config_hash,
            )
            if (
                bundle.crop_size_px
                != tuple(int(value) for value in row["crop_size_px"])
            ):
                raise ValueError(f"evidence/manifest lineage mismatch during scoring: {crop_id}")
            if estimator is None:
                raise AssertionError("support scorer is unavailable")
            saved_objective = result.get("objective")
            rescored_center_objective = estimator.objective_at_center(bundle, raw_center)
            if saved_objective is None or not np.isclose(
                rescored_center_objective,
                float(saved_objective),
                rtol=1e-7,
                atol=1e-9,
            ):
                raise ValueError(
                    f"saved/rescored objective mismatch for {crop_id}: "
                    f"saved={saved_objective}, rescored={rescored_center_objective}"
                )
            reference = result.get("diagnostics", {}).get("global_support_reference")
            delta = result.get("diagnostics", {}).get("support_delta")
            if reference is not None and delta is not None:
                support_threshold = float(reference) + float(delta)
                gt_objective = estimator.objective_at_center(bundle, gt)
                tolerance = 1e-10 * max(1.0, abs(support_threshold))
                core_support_contains = bool(gt_objective <= support_threshold + tolerance)

        ellipse_contains: bool | float = float("nan")
        ellipse_major = float("nan")
        ellipse_minor = float("nan")
        if uncertainty_row is not None and isinstance(uncertainty_row.get("_sidecar"), Mapping):
            sidecar = uncertainty_row["_sidecar"]
            ellipse_contains = _ellipse_contains_gt(sidecar, gt)
            ellipse = sidecar.get("ellipse", {})
            if isinstance(ellipse, Mapping):
                ellipse_major = float(ellipse.get("major_semi_axis_norm", np.nan))
                ellipse_minor = float(ellipse.get("minor_semi_axis_norm", np.nan))
        adjudicated_support_contains = (
            core_support_contains
            if state in {"POINT", "RANGE", "RAY", "AXIS", "MULTIMODAL"}
            else float("nan")
        )
        metrics.append(
            {
                **base,
                "result_status": (
                    "FINISHED"
                    if uncertainty_complete
                    else f"UNCERTAINTY_{uncertainty_status}"
                ),
                "result_index_status": (
                    str(index_row.get("status")) if index_row is not None else "NOT_SUPPLIED"
                ),
                "state": state,
                "geometry_state": geometry_state,
                "uncertainty_status": uncertainty_status,
                "uncertainty_reaudit_required": bool(
                    uncertainty_row.get("requires_search_reaudit", False)
                    if uncertainty_row is not None
                    else False
                ),
                "reason_codes": "|".join(
                    [
                        *result.get("reason_codes", []),
                        *(
                            [str(uncertainty_row.get("reason"))]
                            if uncertainty_row is not None and uncertainty_row.get("reason")
                            else []
                        ),
                    ]
                ),
                "search_adequate": bool(result.get("search_adequate", False)),
                "production_usable": bool(
                    result.get("production_usable", False)
                    and state == "POINT"
                    and not (
                        uncertainty_row.get("requires_search_reaudit", False)
                        if uncertainty_row is not None
                        else False
                    )
                ),
                "point_error_norm": error_norm,
                "point_error_px": error_px,
                "false_point": bool(state == "POINT" and error_norm > false_point_tolerance_norm),
                "catastrophic_point": bool(
                    state == "POINT" and error_norm > catastrophic_tolerance_norm
                ),
                "direction_error_deg": direction_error,
                # Kept as a compatibility alias, but now denotes the full
                # profiled-objective support rather than radial distance alone.
                "range_contains_gt": adjudicated_support_contains,
                "radial_interval_contains_gt": radial_contains,
                "core_support_contains_gt": core_support_contains,
                "support_contains_gt": adjudicated_support_contains,
                "gt_profiled_objective": gt_objective,
                "support_threshold": support_threshold,
                "stability_ellipse_contains_gt": ellipse_contains,
                "stability_ellipse_major_norm": ellipse_major,
                "stability_ellipse_minor_norm": ellipse_minor,
                "range_lower_norm": range_lower,
                "range_upper_norm": range_upper,
                "range_width_norm": range_width,
                "range_unbounded": range_unbounded,
                "finite_mode_count": len(result.get("finite_modes", [])),
                "support_component_count": len(
                    (
                        result.get("diagnostics", {})
                        .get("compact_audit", {})
                        .get("components", [])
                    )
                    if isinstance(
                        result.get("diagnostics", {}).get("compact_audit"), Mapping
                    )
                    else []
                ),
                "far_gap": result.get("diagnostics", {}).get("far_gap", np.nan),
                "point_safety_margin": result.get("diagnostics", {}).get(
                    "point_safety_margin", np.nan
                ),
                "condition_ratio": result.get("condition_ratio", np.nan),
                "model_risk": result.get("model_risk", "UNKNOWN"),
                "prediction_path": result.get("_path"),
            }
        )
    unexpected = sorted(set(predictions) - all_manifest_ids)
    if unexpected:
        raise ValueError(f"predictions not present in crop manifest: {unexpected[:10]}")
    unexpected_index = sorted(set(result_index) - all_manifest_ids)
    if unexpected_index:
        raise ValueError(f"result-index crops not present in crop manifest: {unexpected_index[:10]}")
    return pd.DataFrame(metrics)


def _safe_quantile(values: pd.Series, q: float) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    return float(finite.quantile(q)) if len(finite) else float("nan")


def tree_balanced_quantile(
    metrics: pd.DataFrame, value_column: str, quantile: float
) -> float:
    """Weighted quantile giving each represented biological tree equal mass."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    frame = metrics[["tree_id", value_column]].copy()
    finite = np.isfinite(frame[value_column].to_numpy(dtype=float))
    frame = frame.loc[finite]
    if frame.empty:
        return float("nan")
    counts = frame.groupby("tree_id")[value_column].transform("count").to_numpy(dtype=float)
    tree_count = frame["tree_id"].nunique()
    weights = 1.0 / (tree_count * counts)
    order = np.argsort(frame[value_column].to_numpy(dtype=float), kind="mergesort")
    values = frame[value_column].to_numpy(dtype=float)[order]
    cumulative = np.cumsum(weights[order])
    return float(values[min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)])


def summarize_localization(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = metrics[metrics["state"] == "POINT"]
    tree_rows: list[dict[str, Any]] = []
    for tree_id, frame in metrics.groupby("tree_id", sort=True):
        point_tree = frame[frame["state"] == "POINT"]
        supported_tree = frame[frame["support_contains_gt"].notna()]
        point_support_tree = point_tree[point_tree["support_contains_gt"].notna()]
        point_ellipse_tree = point_tree[point_tree["stability_ellipse_contains_gt"].notna()]
        tree_rows.append(
            {
                "tree_id": tree_id,
                "n_manifest": len(frame),
                "finished_rate": float((frame["result_status"] == "FINISHED").mean()),
                "point_coverage": float((frame["state"] == "POINT").mean()),
                "usable_coverage": float(frame["state"].isin(USABLE_STATES).mean()),
                "production_usable_coverage": float(
                    frame["production_usable"].astype(bool).mean()
                ),
                "false_point_risk": float(point_tree["false_point"].mean())
                if len(point_tree)
                else np.nan,
                "catastrophic_point_risk": float(point_tree["catastrophic_point"].mean())
                if len(point_tree)
                else np.nan,
                "point_median_error_norm": _safe_quantile(point_tree["point_error_norm"], 0.50),
                "point_p90_error_norm": _safe_quantile(point_tree["point_error_norm"], 0.90),
                "point_p95_error_norm": _safe_quantile(point_tree["point_error_norm"], 0.95),
                "direction_median_error_deg": _safe_quantile(frame["direction_error_deg"], 0.50),
                "support_coverage": float(supported_tree["support_contains_gt"].astype(bool).mean())
                if len(supported_tree)
                else np.nan,
                "point_support_coverage": float(
                    point_support_tree["support_contains_gt"].astype(bool).mean()
                )
                if len(point_support_tree)
                else np.nan,
                "point_stability_ellipse_coverage": float(
                    point_ellipse_tree["stability_ellipse_contains_gt"].astype(bool).mean()
                )
                if len(point_ellipse_tree)
                else np.nan,
                "range_coverage": float(
                    frame.loc[
                        (frame["state"] == "RANGE") & frame["support_contains_gt"].notna(),
                        "support_contains_gt",
                    ].astype(bool).mean()
                )
                if ((frame["state"] == "RANGE") & frame["support_contains_gt"].notna()).any()
                else np.nan,
            }
        )
    tree = pd.DataFrame(tree_rows)
    dataset = pd.DataFrame(
        [
            {
                "n_trees": metrics["tree_id"].nunique(),
                "n_sections": metrics["section_id"].nunique(),
                "n_crops": len(metrics),
                "n_missing": int((metrics["result_status"] != "FINISHED").sum()),
                "point_coverage_tree_mean": float(tree["point_coverage"].mean()),
                "usable_coverage_tree_mean": float(tree["usable_coverage"].mean()),
                "production_usable_coverage_tree_mean": float(
                    tree["production_usable_coverage"].mean()
                ),
                "false_point_risk_tree_mean": float(tree["false_point_risk"].mean()),
                "catastrophic_point_risk_tree_mean": float(
                    tree["catastrophic_point_risk"].mean()
                ),
                "point_median_error_norm_tree_mean": float(
                    tree["point_median_error_norm"].mean()
                ),
                "point_p90_error_norm_tree_mean": float(tree["point_p90_error_norm"].mean()),
                "point_p95_error_norm_tree_mean": float(tree["point_p95_error_norm"].mean()),
                "support_coverage_tree_mean": float(tree["support_coverage"].mean()),
                "point_support_coverage_tree_mean": float(
                    tree["point_support_coverage"].mean()
                ),
                "point_stability_ellipse_coverage_tree_mean": float(
                    tree["point_stability_ellipse_coverage"].mean()
                ),
                "point_median_error_norm_tree_balanced": tree_balanced_quantile(
                    point, "point_error_norm", 0.50
                ),
                "point_p90_error_norm_tree_balanced": tree_balanced_quantile(
                    point, "point_error_norm", 0.90
                ),
                "point_p95_error_norm_tree_balanced": tree_balanced_quantile(
                    point, "point_error_norm", 0.95
                ),
            }
        ]
    )
    return tree, dataset


def cluster_bootstrap_summary(
    metrics: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    trees = np.asarray(sorted(metrics["tree_id"].unique()), dtype=object)
    if len(trees) < 2:
        return {"estimate": statistic(metrics), "lower": np.nan, "upper": np.nan}
    grouped = {tree: frame for tree, frame in metrics.groupby("tree_id", sort=False)}
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.choice(trees, size=len(trees), replace=True)
        frames = [
            grouped[tree].assign(
                source_tree_id=tree,
                tree_id=f"bootstrap-{slot}:{tree}",
            )
            for slot, tree in enumerate(sample)
        ]
        values[index] = statistic(pd.concat(frames, ignore_index=True))
    alpha = (1.0 - confidence_level) / 2.0
    finite = values[np.isfinite(values)]
    return {
        "estimate": float(statistic(metrics)),
        "lower": float(np.quantile(finite, alpha)) if len(finite) else np.nan,
        "upper": float(np.quantile(finite, 1.0 - alpha)) if len(finite) else np.nan,
    }


def tree_equal_ecdf_weights(metrics: pd.DataFrame) -> np.ndarray:
    tree_counts = metrics.groupby("tree_id")["crop_id"].transform("count").to_numpy(dtype=float)
    n_trees = metrics["tree_id"].nunique()
    return 1.0 / (n_trees * tree_counts)


def risk_coverage_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    n_trees = metrics["tree_id"].nunique()
    total_by_tree = metrics.groupby("tree_id")["crop_id"].size().to_dict()
    frame = metrics[(metrics["state"] == "POINT") & metrics["point_safety_margin"].notna()].copy()
    frame = frame.sort_values("point_safety_margin", ascending=False)
    if frame.empty:
        return pd.DataFrame(columns=["accepted", "point_coverage", "false_point_risk"])
    weights = np.asarray(
        [1.0 / (n_trees * total_by_tree[tree]) for tree in frame["tree_id"]], dtype=float
    )
    frame["weight"] = weights
    frame["weighted_false"] = weights * frame["false_point"].astype(float)
    cumulative_weight = frame["weight"].cumsum()
    cumulative_false = frame["weighted_false"].cumsum()
    return pd.DataFrame(
        {
            "accepted": np.arange(1, len(frame) + 1),
            "threshold": frame["point_safety_margin"].to_numpy(),
            "point_coverage": cumulative_weight.to_numpy(),
            "false_point_risk": (cumulative_false / cumulative_weight).to_numpy(),
        }
    )
