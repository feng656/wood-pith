#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from racpith.config import load_config
from racpith.contracts import EvidenceBundle, LocateResult
from racpith.estimator import RacPithEstimator
from racpith.evaluation.metrics import (
    load_evidence_index,
    load_predictions,
    validate_evidence_bundle_lineage,
)
from racpith.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json_object,
    sha256_file,
)
from racpith.uncertainty import run_structured_uncertainty


def _one_impl(
    row: Mapping[str, Any],
    prediction_path: str,
    expected_baseline_sha256: str,
    source_result_index_sha256: str,
    config_path: str,
    output_root: str,
    ring_replays: int | None,
    curve_replays: int | None,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    crop_id = str(row["crop_id"])
    split = str(row["split"])
    destination = Path(output_root) / "per_crop" / split / f"{crop_id}.json"
    baseline_path = Path(prediction_path)
    baseline_sha = sha256_file(baseline_path)
    if baseline_sha != expected_baseline_sha256:
        raise ValueError(f"baseline changed after result-index validation for {crop_id}")
    frozen = load_config(config_path)
    metadata_sha = sha256_file(row["metadata_path"])
    npz_sha = sha256_file(row["npz_path"])
    if metadata_sha != row.get("metadata_sha256") or npz_sha != row.get("npz_sha256"):
        raise ValueError(f"evidence content hash mismatch for {crop_id}")
    baseline_payload = read_json_object(baseline_path)
    baseline = LocateResult.from_dict(baseline_payload)
    if baseline.crop_id != crop_id or baseline.config_hash != frozen.sha256:
        raise ValueError(f"baseline crop/config mismatch for {crop_id}")
    if (
        baseline.diagnostics.get("evidence_metadata_sha256") != metadata_sha
        or baseline.diagnostics.get("evidence_npz_sha256") != npz_sha
    ):
        raise ValueError("baseline prediction is not bound to the indexed evidence")
    section = frozen.section("uncertainty")
    requested_ring_replays = (
        int(section["parent_ring_bootstrap_replays"])
        if ring_replays is None
        else ring_replays
    )
    requested_curve_replays = (
        int(section["correlated_curve_replays"])
        if curve_replays is None
        else curve_replays
    )
    replay_protocol = {
        "random_seed": int(seed),
        "curve_correlation_fraction": float(section["curve_correlation_fraction"]),
        "ellipse_confidence_reference": float(section["ellipse_confidence_reference"]),
        "minimum_point_samples": int(section["minimum_point_samples"]),
        "minimum_point_fraction": float(section["minimum_point_fraction"]),
    }
    if resume and destination.exists():
        existing = read_json_object(destination)
        if (
            existing.get("schema_version") != "racpith.uncertainty.v1"
            or existing.get("crop_id") != crop_id
            or str(existing.get("tree_id")) != str(row["tree_id"])
            or str(existing.get("section_id")) != str(row["section_id"])
            or str(existing.get("split")) != split
            or existing.get("config_hash") != frozen.sha256
            or existing.get("baseline_result_sha256") != baseline_sha
            or existing.get("evidence_metadata_sha256") != metadata_sha
            or existing.get("evidence_npz_sha256") != npz_sha
            or existing.get("source_result_index_sha256")
            != source_result_index_sha256
            or existing.get("requested")
            != {
                "parent_ring_bootstrap": requested_ring_replays,
                "correlated_curve": requested_curve_replays,
            }
            or existing.get("replay_protocol") != replay_protocol
        ):
            raise ValueError(f"refusing to resume {crop_id}: provenance differs")
        return {
            "schema_version": "racpith.uncertainty_index.v1",
            "crop_id": crop_id,
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": split,
            "status": "RESUMED",
            "baseline_state": existing["baseline_state"],
            "adjudicated_state": existing["adjudicated_state"],
            "requires_search_reaudit": existing["requires_search_reaudit"],
            "uncertainty_path": str(destination.resolve()),
            "uncertainty_sha256": sha256_file(destination),
            "config_hash": frozen.sha256,
            "baseline_result_sha256": baseline_sha,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": metadata_sha,
            "evidence_npz_sha256": npz_sha,
            "runtime_seconds": 0.0,
            "reason": None,
        }
    try:
        bundle = EvidenceBundle.load(row["metadata_path"])
        validate_evidence_bundle_lineage(
            row,
            bundle,
            expected_config_hash=frozen.sha256,
        )
        estimator = RacPithEstimator(frozen, run_id=f"{baseline.run_id}:uncertainty")
        result = run_structured_uncertainty(
            bundle,
            estimator,
            baseline,
            ring_replays=requested_ring_replays,
            curve_replays=requested_curve_replays,
            seed=seed,
            curve_correlation_fraction=float(section["curve_correlation_fraction"]),
            confidence_level=float(section["ellipse_confidence_reference"]),
            minimum_point_samples=int(section["minimum_point_samples"]),
            minimum_point_fraction=float(section["minimum_point_fraction"]),
        )
        result["split"] = split
        result["baseline_result_path"] = str(baseline_path.resolve())
        result["baseline_result_sha256"] = baseline_sha
        result["source_result_index_sha256"] = source_result_index_sha256
        result["evidence_metadata_sha256"] = metadata_sha
        result["evidence_npz_sha256"] = npz_sha
        atomic_write_json(destination, result)
        return {
            "schema_version": "racpith.uncertainty_index.v1",
            "crop_id": crop_id,
            "tree_id": bundle.tree_id,
            "section_id": bundle.section_id,
            "split": split,
            "status": "FINISHED",
            "baseline_state": result["baseline_state"],
            "adjudicated_state": result["adjudicated_state"],
            "requires_search_reaudit": result["requires_search_reaudit"],
            "uncertainty_path": str(destination.resolve()),
            "uncertainty_sha256": sha256_file(destination),
            "config_hash": frozen.sha256,
            "baseline_result_sha256": baseline_sha,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": metadata_sha,
            "evidence_npz_sha256": npz_sha,
            "runtime_seconds": time.perf_counter() - started,
            "reason": "|".join(result["gate_reason_codes"]) or None,
        }
    except Exception as exc:
        return {
            "schema_version": "racpith.uncertainty_index.v1",
            "crop_id": crop_id,
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": split,
            "status": "FAIL",
            "baseline_state": None,
            "adjudicated_state": "REJECT",
            "requires_search_reaudit": True,
            "uncertainty_path": None,
            "uncertainty_sha256": None,
            "config_hash": frozen.sha256,
            "baseline_result_sha256": baseline_sha,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": row.get("metadata_sha256"),
            "evidence_npz_sha256": row.get("npz_sha256"),
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _one(
    row: Mapping[str, Any],
    prediction_path: str,
    expected_baseline_sha256: str,
    source_result_index_sha256: str,
    config_path: str,
    output_root: str,
    ring_replays: int | None,
    curve_replays: int | None,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    """Preserve the uncertainty denominator when a crop-local precheck fails."""

    started = time.perf_counter()
    try:
        return _one_impl(
            row,
            prediction_path,
            expected_baseline_sha256,
            source_result_index_sha256,
            config_path,
            output_root,
            ring_replays,
            curve_replays,
            seed,
            resume,
        )
    except Exception as exc:
        return {
            "schema_version": "racpith.uncertainty_index.v1",
            "crop_id": str(row["crop_id"]),
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "FAIL",
            "baseline_state": None,
            "adjudicated_state": "REJECT",
            "requires_search_reaudit": True,
            "uncertainty_path": None,
            "uncertainty_sha256": None,
            "config_hash": row.get("config_hash"),
            "baseline_result_sha256": expected_baseline_sha256,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": row.get("metadata_sha256"),
            "evidence_npz_sha256": row.get("npz_sha256"),
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GT-blind parent-ring bootstrap and correlated-curve replay"
    )
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--ring-replays", type=int)
    parser.add_argument("--curve-replays", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-any-error", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    for name, value in (
        ("--ring-replays", args.ring_replays),
        ("--curve-replays", args.curve_replays),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")

    frozen = load_config(args.config)
    all_evidence_rows = list(
        load_evidence_index(
            args.evidence_index,
            expected_config_hash=frozen.sha256,
        ).values()
    )
    seen_crops: set[str] = set()
    for row in all_evidence_rows:
        if row.get("schema_version") != "racpith.evidence_index.v1":
            raise ValueError("unsupported evidence-index schema")
        crop_id = str(row["crop_id"])
        if crop_id in seen_crops:
            raise ValueError(f"duplicate evidence-index crop {crop_id}")
        seen_crops.add(crop_id)
    evidence_rows = all_evidence_rows
    if args.split:
        allowed = set(args.split)
        evidence_rows = [row for row in evidence_rows if row["split"] in allowed]
    if not evidence_rows:
        raise ValueError("selected evidence-index scope is empty")
    all_evidence_by_crop = {
        str(row["crop_id"]): row for row in all_evidence_rows
    }
    rows = [row for row in evidence_rows if row["status"] == "PASS"]
    prediction_payloads, result_index = load_predictions(
        args.predictions,
        result_index_path=args.result_index,
        expected_config_hash=frozen.sha256,
    )
    predictions = {
        crop_id: Path(str(payload["_path"]))
        for crop_id, payload in prediction_payloads.items()
    }
    source_result_index_sha256 = sha256_file(args.result_index)
    for crop_id, result_row in result_index.items():
        evidence_row = all_evidence_by_crop.get(crop_id)
        if evidence_row is None:
            raise ValueError(f"orphan localization result-index crop {crop_id}")
        if any(
            str(result_row.get(field)) != str(evidence_row.get(field))
            for field in ("tree_id", "section_id", "split")
        ):
            raise ValueError(f"localization/evidence lineage mismatch for {crop_id}")
        if result_row.get("status") in {"FINISHED", "RESUMED"} and (
            evidence_row.get("status") != "PASS"
            or result_row.get("evidence_metadata_sha256")
            != evidence_row.get("metadata_sha256")
            or result_row.get("evidence_npz_sha256")
            != evidence_row.get("npz_sha256")
        ):
            raise ValueError(f"localization/evidence content lineage mismatch for {crop_id}")
    missing_rows = [row for row in rows if str(row["crop_id"]) not in predictions]
    rows = [row for row in rows if str(row["crop_id"]) in predictions]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            row,
            str(predictions[str(row["crop_id"])]),
            str(result_index[str(row["crop_id"])]["prediction_sha256"]),
            source_result_index_sha256,
            str(Path(args.config).resolve()),
            str(output.resolve()),
            args.ring_replays,
            args.curve_replays,
            int(frozen.data["random_seed"]) + ordinal,
            args.resume,
        )
        for ordinal, row in enumerate(sorted(rows, key=lambda value: str(value["crop_id"])))
    ]
    index = [
        {
            "schema_version": "racpith.uncertainty_index.v1",
            "crop_id": row["crop_id"],
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "UPSTREAM_LOCALIZATION_FAIL",
            "baseline_state": None,
            "adjudicated_state": "REJECT",
            "requires_search_reaudit": True,
            "uncertainty_path": None,
            "uncertainty_sha256": None,
            "config_hash": frozen.sha256,
            "baseline_result_sha256": None,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": row.get("metadata_sha256"),
            "evidence_npz_sha256": row.get("npz_sha256"),
            "runtime_seconds": 0.0,
            "reason": "MISSING_BASELINE_PREDICTION",
        }
        for row in missing_rows
    ]
    index.extend(
        {
            "schema_version": "racpith.uncertainty_index.v1",
            "crop_id": row["crop_id"],
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "UPSTREAM_EVIDENCE_FAIL",
            "baseline_state": None,
            "adjudicated_state": "REJECT",
            "requires_search_reaudit": True,
            "uncertainty_path": None,
            "uncertainty_sha256": None,
            "config_hash": frozen.sha256,
            "baseline_result_sha256": None,
            "source_result_index_sha256": source_result_index_sha256,
            "evidence_metadata_sha256": row.get("metadata_sha256"),
            "evidence_npz_sha256": row.get("npz_sha256"),
            "runtime_seconds": 0.0,
            "reason": row.get("reason"),
        }
        for row in evidence_rows
        if row["status"] != "PASS"
    )
    if args.workers <= 1:
        index.extend(_one(*job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_one, *job) for job in jobs]
            for future in as_completed(futures):
                index.append(future.result())
    index.sort(key=lambda row: str(row["crop_id"]))
    atomic_write_jsonl(output / "uncertainty_index.jsonl", index)
    failures = sum(row["status"] not in {"FINISHED", "RESUMED"} for row in index)
    reaudit = sum(bool(row["requires_search_reaudit"]) for row in index)
    print(f"uncertainty: {len(index) - failures} completed, {failures} failed, {reaudit} require S3 re-audit")
    # A re-audit flag is a scientific outcome, not a process crash.  Preserve it
    # for S3 adjudication and allow downstream full-denominator summaries to run.
    if failures and args.fail_on_any_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
