#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from racpith.config import load_config
from racpith.contracts import EvidenceBundle, LocateResult
from racpith.contribution import exact_delete_refit
from racpith.estimator import RacPithEstimator
from racpith.evaluation.metrics import (
    CROP_CONTENT_HASH_FIELDS,
    load_evidence_index,
    load_predictions,
    validate_evidence_bundle_lineage,
    validate_manifest_crop_content,
)
from racpith.groups import build_deletion_groups, validate_partition_mass
from racpith.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json_object,
    read_jsonl,
    sha256_file,
    sha256_source_tree,
)


def _registered_crop_subset(
    crop_ids: list[str],
    manifest: dict[str, dict[str, object]],
    maximum_per_tree: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Deterministic, result-blind, distance-stratum round-robin selection."""

    if maximum_per_tree <= 0:
        return sorted(crop_ids), []
    by_tree: dict[str, dict[str, list[str]]] = {}
    for crop_id in crop_ids:
        row = manifest[crop_id]
        tree_id = str(row["tree_id"])
        stratum = str(row.get("distance_stratum", "unknown"))
        by_tree.setdefault(tree_id, {}).setdefault(stratum, []).append(crop_id)
    selected: set[str] = set()
    for tree_id, by_stratum in sorted(by_tree.items()):
        for values in by_stratum.values():
            values.sort(
                key=lambda crop_id: hashlib.sha256(
                    f"{seed}|{tree_id}|{crop_id}".encode("utf-8")
                ).digest()
            )
        strata = sorted(by_stratum)
        cursor = {name: 0 for name in strata}
        while len([value for value in selected if str(manifest[value]["tree_id"]) == tree_id]) < maximum_per_tree:
            added = False
            for name in strata:
                index = cursor[name]
                if index < len(by_stratum[name]):
                    selected.add(by_stratum[name][index])
                    cursor[name] += 1
                    added = True
                    if len([value for value in selected if str(manifest[value]["tree_id"]) == tree_id]) >= maximum_per_tree:
                        break
            if not added:
                break
    chosen = sorted(selected)
    skipped = sorted(set(crop_ids) - selected)
    return chosen, skipped


def _one_crop(
    crop_id: str,
    evidence_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    prediction_path: str,
    expected_baseline_sha256: str,
    source_result_index_sha256: str,
    config_path: str,
    output_root: str,
    include_audit_partitions: bool,
    replay_count: int | None,
    replay_correlation_fraction: float,
    gt_sigma_px: float | None,
    no_gt: bool,
    skip_crossfit: bool,
    replay_seed: int,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        frozen = load_config(config_path)
        validate_manifest_crop_content(manifest_row)
        metadata_sha = sha256_file(evidence_row["metadata_path"])
        npz_sha = sha256_file(evidence_row["npz_path"])
        if (
            metadata_sha != evidence_row.get("metadata_sha256")
            or npz_sha != evidence_row.get("npz_sha256")
        ):
            raise ValueError("indexed evidence content has changed")
        bundle = EvidenceBundle.load(evidence_row["metadata_path"])
        validate_evidence_bundle_lineage(
            evidence_row,
            bundle,
            expected_config_hash=frozen.sha256,
        )
        baseline_path = Path(prediction_path)
        if sha256_file(baseline_path) != expected_baseline_sha256:
            raise ValueError("baseline changed after result-index validation")
        baseline = LocateResult.from_dict(read_json_object(baseline_path))
        if baseline.config_hash != frozen.sha256:
            raise ValueError("baseline and contribution config hashes differ")
        if baseline.crop_id != crop_id or bundle.crop_id != crop_id:
            raise ValueError("manifest/evidence/baseline crop_id mismatch")
        if (
            baseline.diagnostics.get("evidence_metadata_sha256") != metadata_sha
            or baseline.diagnostics.get("evidence_npz_sha256") != npz_sha
        ):
            raise ValueError("baseline prediction is not bound to the indexed evidence")
        for field, actual in (
            ("tree_id", bundle.tree_id),
            ("section_id", bundle.section_id),
        ):
            if str(evidence_row[field]) != actual or str(manifest_row[field]) != actual:
                raise ValueError(f"manifest/evidence bundle {field} mismatch")
        if str(evidence_row["split"]) != str(manifest_row["split"]):
            raise ValueError("manifest/evidence split mismatch")
        if any(
            evidence_row.get(field) != manifest_row.get(field)
            for field in CROP_CONTENT_HASH_FIELDS
        ):
            raise ValueError("manifest/evidence crop content hash mismatch")
        if tuple(int(value) for value in manifest_row["crop_size_px"]) != bundle.crop_size_px:
            raise ValueError("manifest/evidence crop size mismatch")
        if not np.isclose(
            float(manifest_row["normalization_scale_px"]),
            bundle.normalization_scale_px,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("manifest/evidence normalization scale mismatch")
        groups = build_deletion_groups(
            bundle,
            frozen.section("contribution"),
            include_audit_partitions=include_audit_partitions,
        )
        partition_failures = validate_partition_mass(bundle, groups)
        if partition_failures:
            raise ValueError("; ".join(partition_failures))
        gt_norm = None
        if not no_gt:
            pith = np.asarray(manifest_row["pith_crop_px"], dtype=np.float64)
            width, height = manifest_row["crop_size_px"]
            gt_norm = (
                pith - np.asarray([width / 2.0, height / 2.0])
            ) / bundle.normalization_scale_px
        estimator = RacPithEstimator(frozen, run_id=f"{baseline.run_id}:delete-refit")
        records, minus_results = exact_delete_refit(
            bundle,
            estimator,
            groups,
            baseline,
            baseline_path,
            gt_norm,
            replay_count=replay_count,
            replay_seed=replay_seed,
            replay_correlation_fraction=replay_correlation_fraction,
            gt_sigma_px=gt_sigma_px,
            compute_crossfit=not skip_crossfit,
        )
        split = str(manifest_row["split"])
        for record in records:
            record["split"] = split
            record["inner_fold"] = manifest_row.get("inner_fold")
            record["evidence_metadata_sha256"] = metadata_sha
            record["evidence_npz_sha256"] = npz_sha
            record["source_result_index_sha256"] = source_result_index_sha256
        validate_manifest_crop_content(manifest_row)
        output = Path(output_root)
        record_path = output / "per_crop" / f"{crop_id}.jsonl"
        atomic_write_jsonl(record_path, records)
        minus_dir = output / "minus_results" / crop_id
        for group_id, result in minus_results.items():
            atomic_write_json(minus_dir / f"{group_id}.json", result.as_dict())
        return {
            "schema_version": "racpith.contribution_index.v1",
            "crop_id": crop_id,
            "tree_id": bundle.tree_id,
            "section_id": bundle.section_id,
            "split": split,
            "status": "PASS",
            "config_hash": frozen.sha256,
            "baseline_result_sha256": sha256_file(baseline_path),
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": metadata_sha,
            "evidence_npz_sha256": npz_sha,
            "groups": len(records),
            "signed_evaluable": sum(
                record["gt_label"] in {"BENEFICIAL_GT", "HARMFUL_GT", "NEUTRAL"}
                for record in records
            ),
            "point_signed_evaluable": sum(
                record["gt_point_label"] in {"BENEFICIAL_GT", "HARMFUL_GT", "NEUTRAL"}
                for record in records
            ),
            "record_path": str(record_path.resolve()),
            "record_sha256": sha256_file(record_path),
            "runtime_seconds": time.perf_counter() - started,
            "reason": None,
        }
    except Exception as exc:
        return {
            "schema_version": "racpith.contribution_index.v1",
            "crop_id": crop_id,
            "tree_id": manifest_row["tree_id"],
            "section_id": manifest_row["section_id"],
            "split": manifest_row["split"],
            "status": "FAIL",
            "config_hash": evidence_row.get("config_hash"),
            "baseline_result_sha256": None,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": evidence_row.get("metadata_sha256"),
            "evidence_npz_sha256": evidence_row.get("npz_sha256"),
            "groups": 0,
            "signed_evaluable": 0,
            "point_signed_evaluable": 0,
            "record_path": None,
            "record_sha256": None,
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact ring/arc/subarc delete-refit")
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--include-audit-partitions", action="store_true")
    parser.add_argument("--replays", type=int)
    parser.add_argument("--replay-correlation-fraction", type=float, default=0.05)
    parser.add_argument("--gt-sigma-px", type=float)
    parser.add_argument("--no-gt", action="store_true")
    parser.add_argument("--skip-crossfit", action="store_true")
    parser.add_argument(
        "--max-crops-per-tree",
        type=int,
        help="result-blind registered audit subsample; 0 means all joined crops",
    )
    parser.add_argument(
        "--allow-partial-result-index",
        action="store_true",
        help=(
            "accept a partial result index for development; contribute only on "
            "crops with PASS evidence and a finished/resumed baseline"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fail-on-any-error", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    if args.replays is not None and args.replays < 0:
        raise ValueError("--replays must be non-negative")
    if not 0.0 < args.replay_correlation_fraction <= 1.0:
        raise ValueError("--replay-correlation-fraction must lie in (0,1]")
    if args.gt_sigma_px is not None and args.gt_sigma_px < 0.0:
        raise ValueError("--gt-sigma-px must be non-negative")
    if args.max_crops_per_tree is not None and args.max_crops_per_tree < 0:
        raise ValueError("--max-crops-per-tree must be zero or positive")

    frozen = load_config(args.config)
    evidence_source = list(
        load_evidence_index(
            args.evidence_index,
            expected_config_hash=frozen.sha256,
        ).values()
    )
    manifest_source = read_jsonl(args.crop_manifest)
    seen_evidence: set[str] = set()
    for row in evidence_source:
        if row.get("schema_version") != "racpith.evidence_index.v1":
            raise ValueError("unsupported evidence-index schema")
        crop_id = str(row["crop_id"])
        if crop_id in seen_evidence:
            raise ValueError(f"duplicate evidence-index crop {crop_id}")
        seen_evidence.add(crop_id)
    manifest: dict[str, dict[str, Any]] = {}
    for row in manifest_source:
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported crop-manifest schema")
        crop_id = str(row["crop_id"])
        if crop_id in manifest:
            raise ValueError(f"duplicate crop-manifest row for {crop_id}")
        manifest[crop_id] = row
    orphan_evidence = sorted(seen_evidence - set(manifest))
    if orphan_evidence:
        raise ValueError(f"evidence-index crops absent from manifest: {orphan_evidence[:10]}")
    evidence_by_crop = {str(row["crop_id"]): row for row in evidence_source}
    for crop_id, evidence_row in evidence_by_crop.items():
        if any(
            str(evidence_row.get(field)) != str(manifest[crop_id].get(field))
            for field in ("tree_id", "section_id", "split", *CROP_CONTENT_HASH_FIELDS)
        ):
            raise ValueError(f"evidence/manifest lineage mismatch for {crop_id}")
    evidence_rows = {
        str(row["crop_id"]): row for row in evidence_source if row["status"] == "PASS"
    }
    result_index_path = Path(args.result_index).expanduser().resolve()
    if args.allow_partial_result_index and not result_index_path.is_file():
        prediction_payloads, result_index = load_predictions(
            args.predictions,
            result_index_path=None,
            expected_config_hash=frozen.sha256,
        )
        result_index = {
            crop_id: {
                "schema_version": "racpith.result_index.v1",
                "crop_id": crop_id,
                "tree_id": payload["tree_id"],
                "section_id": payload["section_id"],
                "split": payload["split"],
                "status": "FINISHED",
                "state": payload["state"],
                "prediction_path": payload["_path"],
                "prediction_sha256": sha256_file(payload["_path"]),
                "config_hash": frozen.sha256,
                "evidence_metadata_sha256": payload.get("diagnostics", {}).get(
                    "evidence_metadata_sha256"
                ),
                "evidence_npz_sha256": payload.get("diagnostics", {}).get(
                    "evidence_npz_sha256"
                ),
                "runtime_seconds": None,
                "reason": None,
            }
            for crop_id, payload in prediction_payloads.items()
        }
        source_result_index_sha256 = sha256_source_tree(
            Path(args.config).resolve().parents[1]
        )
    else:
        prediction_payloads, result_index = load_predictions(
            args.predictions,
            result_index_path=result_index_path,
            expected_config_hash=frozen.sha256,
        )
        source_result_index_sha256 = sha256_file(result_index_path)
    predictions = {
        crop_id: Path(str(payload["_path"]))
        for crop_id, payload in prediction_payloads.items()
    }
    for crop_id, row in result_index.items():
        if crop_id not in manifest or crop_id not in evidence_by_crop:
            raise ValueError(f"orphan localization result-index crop {crop_id}")
        if any(
            str(row.get(field)) != str(manifest[crop_id].get(field))
            or str(row.get(field)) != str(evidence_by_crop[crop_id].get(field))
            for field in ("tree_id", "section_id", "split")
        ):
            raise ValueError(f"localization/evidence/manifest lineage mismatch for {crop_id}")
        if any(
            evidence_by_crop[crop_id].get(field) != manifest[crop_id].get(field)
            for field in CROP_CONTENT_HASH_FIELDS
        ):
            raise ValueError(f"evidence/manifest crop content hash mismatch for {crop_id}")
        if row.get("status") in {"FINISHED", "RESUMED"} and (
            evidence_by_crop[crop_id].get("status") != "PASS"
            or row.get("evidence_metadata_sha256")
            != evidence_by_crop[crop_id].get("metadata_sha256")
            or row.get("evidence_npz_sha256")
            != evidence_by_crop[crop_id].get("npz_sha256")
        ):
            raise ValueError(f"localization/evidence content lineage mismatch for {crop_id}")
    scope = sorted(manifest)
    if args.split:
        allowed = set(args.split)
        scope = [crop_id for crop_id in scope if manifest[crop_id]["split"] in allowed]
    if not scope:
        raise ValueError("selected crop-manifest scope is empty")
    for crop_id in scope:
        validate_manifest_crop_content(manifest[crop_id])
    joined = sorted(set(evidence_rows) & set(predictions) & set(manifest))
    if args.split:
        joined = [crop_id for crop_id in joined if manifest[crop_id]["split"] in allowed]
    missing_prerequisite = sorted(set(scope) - set(joined))
    configured_maximum = int(frozen.section("contribution").get("maximum_crops_per_tree", 0))
    maximum = configured_maximum if args.max_crops_per_tree is None else args.max_crops_per_tree
    selected, skipped = _registered_crop_subset(
        joined, manifest, maximum, int(frozen.data["random_seed"])
    )
    output = Path(args.output)
    (output / "per_crop").mkdir(parents=True, exist_ok=True)
    (output / "minus_results").mkdir(parents=True, exist_ok=True)
    index: list[dict[str, object]] = [
        {
            "schema_version": "racpith.contribution_index.v1",
            "crop_id": crop_id,
            "tree_id": manifest[crop_id]["tree_id"],
            "section_id": manifest[crop_id]["section_id"],
            "split": manifest[crop_id]["split"],
            "status": "SKIPPED_REGISTERED_SUBSAMPLE",
            "config_hash": frozen.sha256,
            "baseline_result_sha256": None,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": evidence_rows[crop_id].get("metadata_sha256"),
            "evidence_npz_sha256": evidence_rows[crop_id].get("npz_sha256"),
            "groups": 0,
            "signed_evaluable": 0,
            "point_signed_evaluable": 0,
            "record_path": None,
            "record_sha256": None,
            "runtime_seconds": 0.0,
            "reason": f"maximum_crops_per_tree={maximum}",
        }
        for crop_id in skipped
    ]
    index.extend(
        {
            "schema_version": "racpith.contribution_index.v1",
            "crop_id": crop_id,
            "tree_id": manifest[crop_id]["tree_id"],
            "section_id": manifest[crop_id]["section_id"],
            "split": manifest[crop_id]["split"],
            "status": "MISSING_PREREQUISITE",
            "config_hash": frozen.sha256,
            "baseline_result_sha256": None,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": evidence_rows.get(crop_id, {}).get("metadata_sha256"),
            "evidence_npz_sha256": evidence_rows.get(crop_id, {}).get("npz_sha256"),
            "groups": 0,
            "signed_evaluable": 0,
            "point_signed_evaluable": 0,
            "record_path": None,
            "record_sha256": None,
            "runtime_seconds": 0.0,
            "reason": "evidence or immutable baseline prediction unavailable",
        }
        for crop_id in missing_prerequisite
    )
    jobs = [
        (
            crop_id,
            evidence_rows[crop_id],
            manifest[crop_id],
            str(predictions[crop_id].resolve()),
            str(result_index[crop_id]["prediction_sha256"]),
            source_result_index_sha256,
            str(Path(args.config).resolve()),
            str(output.resolve()),
            args.include_audit_partitions,
            args.replays,
            args.replay_correlation_fraction,
            args.gt_sigma_px,
            args.no_gt,
            args.skip_crossfit,
            int(frozen.data["random_seed"]) + ordinal,
        )
        for ordinal, crop_id in enumerate(selected)
    ]
    if args.workers <= 1:
        index.extend(_one_crop(*job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_one_crop, *job) for job in jobs]
            for future in as_completed(futures):
                index.append(future.result())
    index.sort(key=lambda row: str(row["crop_id"]))
    atomic_write_jsonl(output / "contribution_index.jsonl", index)
    failures = sum(
        row["status"] not in {"PASS", "SKIPPED_REGISTERED_SUBSAMPLE"}
        for row in index
    )
    passed = sum(row["status"] == "PASS" for row in index)
    print(
        f"contributions: {passed} crops PASS, {failures} FAIL, "
        f"{len(skipped)} skipped by registered result-blind subsample, "
        f"{len(missing_prerequisite)} missing prerequisites"
    )
    if failures and args.fail_on_any_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
