#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from racpith.config import load_config
from racpith.contracts import EvidenceBundle
from racpith.estimator import RacPithEstimator
from racpith.evaluation.metrics import (
    load_evidence_index,
    validate_evidence_bundle_lineage,
)
from racpith.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    read_json_object,
    runtime_provenance,
    sha256_file,
    sha256_source_tree,
)


def _with_coordinate_outputs(result_payload: dict[str, Any], bundle: EvidenceBundle) -> dict[str, Any]:
    width, height = bundle.crop_size_px
    origin_x, origin_y = bundle.crop_origin_source_px
    scale = bundle.normalization_scale_px

    def convert(value: object) -> tuple[list[float] | None, list[float] | None]:
        if value is None:
            return None, None
        point = list(value)  # type: ignore[arg-type]
        crop = [width / 2.0 + scale * float(point[0]), height / 2.0 + scale * float(point[1])]
        source = [origin_x + crop[0], origin_y + crop[1]]
        return crop, source

    raw_crop, raw_source = convert(result_payload.get("raw_center_norm"))
    usable_crop, usable_source = convert(result_payload.get("usable_center_norm"))
    result_payload.update(
        {
            "coordinate_convention": "[x,y]=[column,row]",
            "normalization_scale_px": scale,
            "crop_origin_source_px": [origin_x, origin_y],
            "crop_size_px": [width, height],
            "raw_center_crop_px": raw_crop,
            "raw_center_source_px": raw_source,
            "usable_center_crop_px": usable_crop,
            "usable_center_source_px": usable_source,
            "range_interval_px": (
                [
                    (
                        float(result_payload["range_interval_norm"][0]) * scale
                        if result_payload["range_interval_norm"][0] is not None
                        else None
                    ),
                    (
                        float(result_payload["range_interval_norm"][1]) * scale
                        if result_payload["range_interval_norm"][1] is not None
                        else None
                    ),
                ]
                if result_payload.get("range_interval_norm") is not None
                else None
            ),
        }
    )
    return result_payload


def _fit_one_impl(
    row: Mapping[str, Any], config_path: str, run_id: str, output_root: str, resume: bool
) -> dict[str, Any]:
    started = time.perf_counter()
    crop_id = str(row["crop_id"])
    output = Path(output_root) / "predictions" / str(row["split"]) / f"{crop_id}.json"
    frozen = load_config(config_path)
    metadata_sha = sha256_file(row["metadata_path"])
    npz_sha = sha256_file(row["npz_path"])
    if (
        metadata_sha != row.get("metadata_sha256")
        or npz_sha != row.get("npz_sha256")
    ):
        raise ValueError(f"indexed evidence content changed for {crop_id}")
    if resume and output.exists():
        existing = read_json_object(output)
        if (
            existing.get("schema_version") != "racpith.locate.v1"
            or existing.get("crop_id") != crop_id
            or existing.get("config_hash") != frozen.sha256
        ):
            raise ValueError(
                f"refusing to resume {crop_id}: existing result schema/crop/config differs"
            )
        if existing.get("run_id") != run_id:
            raise ValueError(
                f"refusing to resume {crop_id}: existing run_id="
                f"{existing.get('run_id')!r}, requested={run_id!r}"
            )
        diagnostics = existing.get("diagnostics", {})
        if (
            diagnostics.get("evidence_metadata_sha256") != metadata_sha
            or diagnostics.get("evidence_npz_sha256") != npz_sha
        ):
            raise ValueError(f"refusing to resume {crop_id}: evidence content differs")
        return {
            "schema_version": "racpith.result_index.v1",
            "crop_id": crop_id,
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "RESUMED",
            "state": existing["state"],
            "prediction_path": str(output.resolve()),
            "prediction_sha256": sha256_file(output),
            "config_hash": frozen.sha256,
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
        estimator = RacPithEstimator(frozen, run_id=run_id)
        result = estimator.fit(bundle)
        result.diagnostics["normalization_scale_px"] = bundle.normalization_scale_px
        result.diagnostics["evidence_metadata_sha256"] = metadata_sha
        result.diagnostics["evidence_npz_sha256"] = npz_sha
        atomic_write_json(output, _with_coordinate_outputs(result.as_dict(), bundle))
        return {
            "schema_version": "racpith.result_index.v1",
            "crop_id": crop_id,
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "FINISHED",
            "state": result.state.value,
            "prediction_path": str(output.resolve()),
            "prediction_sha256": sha256_file(output),
            "config_hash": frozen.sha256,
            "evidence_metadata_sha256": metadata_sha,
            "evidence_npz_sha256": npz_sha,
            "runtime_seconds": time.perf_counter() - started,
            "reason": "|".join(result.reason_codes),
        }
    except Exception as exc:
        return {
            "schema_version": "racpith.result_index.v1",
            "crop_id": crop_id,
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "CRASH",
            "state": "REJECT",
            "prediction_path": None,
            "prediction_sha256": None,
            "config_hash": frozen.sha256,
            "evidence_metadata_sha256": metadata_sha,
            "evidence_npz_sha256": npz_sha,
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _fit_one(
    row: Mapping[str, Any], config_path: str, run_id: str, output_root: str, resume: bool
) -> dict[str, Any]:
    """Convert every crop-local exception into a full-denominator index row."""

    started = time.perf_counter()
    try:
        return _fit_one_impl(row, config_path, run_id, output_root, resume)
    except Exception as exc:
        return {
            "schema_version": "racpith.result_index.v1",
            "crop_id": str(row["crop_id"]),
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "CRASH",
            "state": "REJECT",
            "prediction_path": None,
            "prediction_sha256": None,
            "config_hash": row.get("config_hash"),
            "evidence_metadata_sha256": row.get("metadata_sha256"),
            "evidence_npz_sha256": row.get("npz_sha256"),
            "runtime_seconds": time.perf_counter() - started,
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full GT-blind RAC-Pith v2 estimator")
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--fail-on-any-error", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    frozen = load_config(args.config)
    rows = list(
        load_evidence_index(
            args.evidence_index,
            expected_config_hash=frozen.sha256,
        ).values()
    )
    if args.split:
        rows = [row for row in rows if row["split"] in set(args.split)]
    if not rows:
        raise ValueError("selected evidence-index scope is empty")
    seen_crop_ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != "racpith.evidence_index.v1":
            raise ValueError("unsupported evidence-index schema")
        crop_id = str(row["crop_id"])
        if crop_id in seen_crop_ids:
            raise ValueError(f"duplicate evidence-index crop {crop_id}")
        seen_crop_ids.add(crop_id)
    pass_rows = [row for row in rows if row["status"] == "PASS"]
    prefailed = [
        {
            "schema_version": "racpith.result_index.v1",
            "crop_id": row["crop_id"],
            "tree_id": row["tree_id"],
            "section_id": row["section_id"],
            "split": row["split"],
            "status": "UPSTREAM_EVIDENCE_FAIL",
            "state": "REJECT",
            "prediction_path": None,
            "prediction_sha256": None,
            "config_hash": frozen.sha256,
            "evidence_metadata_sha256": row.get("metadata_sha256"),
            "evidence_npz_sha256": row.get("npz_sha256"),
            "runtime_seconds": 0.0,
            "reason": row.get("reason"),
        }
        for row in rows
        if row["status"] != "PASS"
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"racpith-{uuid.uuid4().hex[:12]}"
    provenance_path = output / "provenance.json"
    evidence_index_sha256 = sha256_file(args.evidence_index)
    source_tree_sha256 = sha256_source_tree(Path(__file__).resolve().parents[1])
    provenance = {
        "schema_version": "racpith.run.v1",
        "run_id": run_id,
        "config_path": str(Path(args.config).resolve()),
        "config_hash": frozen.sha256,
        "device": args.device,
        "geometry_dtype": "float64",
        "gt_available_to_estimator": False,
        "evidence_index_path": str(Path(args.evidence_index).resolve()),
        "evidence_index_sha256": evidence_index_sha256,
        "source_tree_sha256": source_tree_sha256,
        "runtime": runtime_provenance(),
    }
    if args.resume and provenance_path.exists():
        existing_provenance = read_json_object(provenance_path)
        expected = {
            "run_id": run_id,
            "config_hash": frozen.sha256,
            "evidence_index_sha256": evidence_index_sha256,
            "device": args.device,
            "geometry_dtype": "float64",
            "source_tree_sha256": source_tree_sha256,
        }
        mismatches = {
            key: {"existing": existing_provenance.get(key), "requested": value}
            for key, value in expected.items()
            if existing_provenance.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"refusing to resume localization with changed provenance: {mismatches}"
            )
    else:
        atomic_write_json(provenance_path, provenance)
    results = list(prefailed)
    fit_started = time.perf_counter()
    total = len(pass_rows)
    progress_step = max(1, min(500, total // 100 or 1))
    if args.workers <= 1:
        for completed, row in enumerate(pass_rows, start=1):
            results.append(
                _fit_one(
                    row,
                    str(Path(args.config).resolve()),
                    run_id,
                    str(output),
                    args.resume,
                )
            )
            if completed % progress_step == 0 or completed == total:
                elapsed = time.perf_counter() - fit_started
                rate = elapsed / completed
                print(
                    f"localization: {completed}/{total} crops; "
                    f"elapsed={elapsed/3600:.2f} h; "
                    f"eta={(total - completed) * rate/3600:.2f} h",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _fit_one,
                    row,
                    str(Path(args.config).resolve()),
                    run_id,
                    str(output),
                    args.resume,
                )
                for row in pass_rows
            ]
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % progress_step == 0 or completed == total:
                    elapsed = time.perf_counter() - fit_started
                    rate = elapsed / completed
                    print(
                        f"localization: {completed}/{total} crops; "
                        f"elapsed={elapsed/3600:.2f} h; "
                        f"eta={(total - completed) * rate/3600:.2f} h",
                        flush=True,
                    )
    results.sort(key=lambda row: row["crop_id"])
    atomic_write_jsonl(output / "result_index.jsonl", results)
    crashes = sum(row["status"] in {"CRASH", "UPSTREAM_EVIDENCE_FAIL"} for row in results)
    print(f"run={run_id}: {len(results) - crashes} completed, {crashes} failed")
    if crashes and args.fail_on_any_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
