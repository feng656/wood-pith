#!/usr/bin/env python3
"""Summarise engineering cost without rerunning any scientific stage.

The command is intentionally index-driven.  It keeps failed, skipped, and
missing crops in the denominator and rejects mixed-configuration artefacts.
Wall-clock values are per-worker elapsed seconds recorded by the producing
stage; they must not be interpreted as end-to-end elapsed time under parallel
execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from racpith.config import load_config
from racpith.provenance import (
    atomic_write_json,
    read_json_object,
    read_jsonl,
    sha256_file,
)


STAGES: dict[str, dict[str, Any]] = {
    "evidence": {
        "schema": "racpith.evidence_index.v1",
        "success": {"PASS"},
        "executed": {"PASS", "FAIL"},
        "allowed": {"PASS", "FAIL", "MANIFEST_INELIGIBLE"},
    },
    "localization": {
        "schema": "racpith.result_index.v1",
        "success": {"FINISHED", "RESUMED"},
        "executed": {"FINISHED", "CRASH"},
        "allowed": {"FINISHED", "RESUMED", "CRASH", "UPSTREAM_EVIDENCE_FAIL"},
    },
    "uncertainty": {
        "schema": "racpith.uncertainty_index.v1",
        "success": {"FINISHED", "RESUMED"},
        "executed": {"FINISHED", "FAIL"},
        "allowed": {
            "FINISHED",
            "RESUMED",
            "FAIL",
            "UPSTREAM_LOCALIZATION_FAIL",
            "UPSTREAM_EVIDENCE_FAIL",
        },
    },
    "contribution": {
        "schema": "racpith.contribution_index.v1",
        "success": {"PASS", "RESUMED"},
        "executed": {"PASS", "FAIL"},
        "allowed": {
            "PASS",
            "RESUMED",
            "FAIL",
            "SKIPPED_REGISTERED_SUBSAMPLE",
            "MISSING_PREREQUISITE",
        },
    },
}


def _finite_nonnegative(value: object, *, context: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{context} runtime_seconds must be finite and non-negative")
    return number


def _load_stage(
    stage: str,
    path: str | Path,
    manifest: Mapping[str, Mapping[str, Any]],
    full_manifest: Mapping[str, Mapping[str, Any]],
    config_hash: str,
) -> dict[str, dict[str, Any]]:
    specification = STAGES[stage]
    indexed: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for source in read_jsonl(path):
        if source.get("schema_version") != specification["schema"]:
            raise ValueError(f"unsupported {stage} index schema")
        crop_id = str(source["crop_id"])
        if crop_id in seen:
            raise ValueError(f"duplicate {stage} index row for {crop_id}")
        seen.add(crop_id)
        manifest_source = full_manifest.get(crop_id)
        if manifest_source is None:
            raise ValueError(f"orphan {stage} index row for {crop_id}")
        if any(
            str(source.get(field)) != str(manifest_source.get(field))
            for field in ("tree_id", "section_id", "split")
        ):
            raise ValueError(f"{stage}/manifest lineage mismatch for {crop_id}")
        if crop_id not in manifest:
            continue
        if source.get("config_hash") != config_hash:
            raise ValueError(f"{stage}/config hash mismatch for {crop_id}")
        status = str(source.get("status"))
        if status not in specification["allowed"]:
            raise ValueError(f"unsupported {stage} status {status!r} for {crop_id}")
        row = dict(source)
        row["runtime_seconds"] = _finite_nonnegative(
            row.get("runtime_seconds", float("nan")),
            context=f"{stage}/{crop_id}",
        )
        indexed[crop_id] = row
    return indexed


def _stage_rows(
    stage: str,
    indexed: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    specification = STAGES[stage]
    output: list[dict[str, Any]] = []
    for crop_id, source in sorted(manifest.items()):
        record = indexed.get(crop_id)
        status = str(record.get("status")) if record is not None else "MISSING_INDEX"
        runtime = float(record["runtime_seconds"]) if record is not None else np.nan
        output.append(
            {
                "stage": stage,
                "crop_id": crop_id,
                "tree_id": str(source["tree_id"]),
                "section_id": str(source["section_id"]),
                "split": str(source["split"]),
                "status": status,
                "success": status in specification["success"],
                "executed_in_this_run": status in specification["executed"],
                "runtime_seconds": runtime,
                "n_nodes": record.get("n_nodes") if record is not None else None,
                "n_parent_rings": (
                    record.get("n_parent_rings") if record is not None else None
                ),
                "n_deletion_groups": record.get("groups") if record is not None else None,
                "reason": record.get("reason") if record is not None else "MISSING_INDEX",
            }
        )
    return output


def _runtime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, group in frame.groupby("stage", sort=False):
        timed = group[
            group["executed_in_this_run"]
            & np.isfinite(group["runtime_seconds"].to_numpy(dtype=float))
        ]
        values = timed["runtime_seconds"]
        rows.append(
            {
                "stage": stage,
                "manifest_crops": len(group),
                "successful_crops": int(group["success"].sum()),
                "failed_or_missing_crops": int((~group["success"]).sum()),
                "success_rate": float(group["success"].mean()),
                "executed_with_timing": len(timed),
                "not_timed_in_this_run": int((~group["executed_in_this_run"]).sum()),
                "worker_seconds_total": float(values.sum()) if len(values) else 0.0,
                "runtime_mean_seconds": float(values.mean()) if len(values) else np.nan,
                "runtime_p50_seconds": float(values.quantile(0.50)) if len(values) else np.nan,
                "runtime_p90_seconds": float(values.quantile(0.90)) if len(values) else np.nan,
                "runtime_p95_seconds": float(values.quantile(0.95)) if len(values) else np.nan,
                "runtime_max_seconds": float(values.max()) if len(values) else np.nan,
                "timing_semantics": "per-worker elapsed; excludes RESUMED and unexecuted rows",
            }
        )
    return pd.DataFrame(rows)


def _localization_diagnostics(
    indexed: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
    config_hash: str,
    prediction_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for crop_id, record in sorted(indexed.items()):
        if record.get("status") not in {"FINISHED", "RESUMED"}:
            continue
        path_value = record.get("prediction_path")
        if path_value is None:
            raise ValueError(f"finished localization index lacks prediction_path: {crop_id}")
        path = Path(str(path_value)).expanduser().resolve()
        if not path.is_relative_to(prediction_root):
            raise ValueError(f"localization prediction escapes run root for {crop_id}")
        indexed_hash = record.get("prediction_sha256")
        if indexed_hash is None or sha256_file(path) != indexed_hash:
            raise ValueError(f"localization prediction hash mismatch for {crop_id}")
        payload = read_json_object(path)
        if payload.get("schema_version") != "racpith.locate.v1":
            raise ValueError(f"unsupported localization result schema for {crop_id}")
        if str(payload.get("crop_id")) != crop_id:
            raise ValueError(f"localization result/index crop mismatch for {crop_id}")
        if payload.get("config_hash") != config_hash:
            raise ValueError(f"localization result/config mismatch for {crop_id}")
        diagnostics = payload.get("diagnostics", {})
        rows.append(
            {
                "crop_id": crop_id,
                "tree_id": str(manifest[crop_id]["tree_id"]),
                "section_id": str(manifest[crop_id]["section_id"]),
                "split": str(manifest[crop_id]["split"]),
                "state": str(payload.get("state", "REJECT")),
                "search_adequate": bool(payload.get("search_adequate", False)),
                "model_risk": str(payload.get("model_risk", "UNKNOWN")),
                "profile_triggered": bool(diagnostics.get("profile_triggered", False)),
                "compact_audit_required": bool(
                    diagnostics.get("compact_audit_required", False)
                ),
                "compact_radius_cap_truncated": bool(
                    diagnostics.get("compact_radius_cap_truncated", False)
                ),
                "finite_trial_count": len(diagnostics.get("finite_trials", [])),
                "finite_converged_count": int(
                    diagnostics.get("finite_converged_count", 0)
                ),
                "profile_count": len(payload.get("profiles", [])),
                "reason_codes": "|".join(str(value) for value in payload.get("reason_codes", [])),
            }
        )
    return pd.DataFrame(rows)


def _diagnostic_summary(frame: pd.DataFrame, manifest_crops: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "manifest_crops": manifest_crops,
                    "finished_predictions": 0,
                    "profile_trigger_rate": np.nan,
                    "compact_audit_rate": np.nan,
                    "search_adequate_rate": np.nan,
                    "compact_radius_cap_truncation_rate": np.nan,
                }
            ]
        )
    tree_rates = (
        frame.groupby("tree_id")[[
            "profile_triggered",
            "compact_audit_required",
            "search_adequate",
            "compact_radius_cap_truncated",
        ]]
        .mean()
    )
    return pd.DataFrame(
        [
            {
                "manifest_crops": manifest_crops,
                "finished_predictions": len(frame),
                "finished_prediction_coverage": len(frame) / max(manifest_crops, 1),
                "profile_trigger_rate_crop": float(frame["profile_triggered"].mean()),
                "profile_trigger_rate_tree_mean": float(
                    tree_rates["profile_triggered"].mean()
                ),
                "compact_audit_rate_crop": float(
                    frame["compact_audit_required"].mean()
                ),
                "compact_audit_rate_tree_mean": float(
                    tree_rates["compact_audit_required"].mean()
                ),
                "search_adequate_rate_crop": float(frame["search_adequate"].mean()),
                "search_adequate_rate_tree_mean": float(
                    tree_rates["search_adequate"].mean()
                ),
                "compact_radius_cap_truncation_rate_crop": float(
                    frame["compact_radius_cap_truncated"].mean()
                ),
                "compact_radius_cap_truncation_rate_tree_mean": float(
                    tree_rates["compact_radius_cap_truncated"].mean()
                ),
                "finite_trial_count_p90": float(
                    frame["finite_trial_count"].quantile(0.90)
                ),
                "profile_count_p90": float(frame["profile_count"].quantile(0.90)),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-denominator runtime and trigger analysis from immutable indexes"
    )
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--result-index", required=True)
    parser.add_argument("--uncertainty-index")
    parser.add_argument("--contribution-index")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    args = parser.parse_args()

    frozen = load_config(args.config)
    splits = set(args.split) if args.split else None
    full_manifest: dict[str, dict[str, Any]] = {}
    tree_splits: dict[str, set[str]] = {}
    for source in read_jsonl(args.crop_manifest):
        if source.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported crop-manifest schema")
        crop_id = str(source["crop_id"])
        if crop_id in full_manifest:
            raise ValueError(f"duplicate manifest crop {crop_id}")
        full_manifest[crop_id] = dict(source)
        tree_splits.setdefault(str(source["tree_id"]), set()).add(
            str(source["split"])
        )
    manifest = {
        crop_id: source
        for crop_id, source in full_manifest.items()
        if not splits or str(source["split"]) in splits
    }
    if not manifest:
        raise ValueError("selected manifest scope is empty")
    leaking = {tree: sorted(value) for tree, value in tree_splits.items() if len(value) != 1}
    if leaking:
        raise ValueError(f"tree split leakage in runtime denominator: {leaking}")
    selected_tree_ids = {str(source["tree_id"]) for source in manifest.values()}

    supplied = {
        "evidence": args.evidence_index,
        "localization": args.result_index,
        "uncertainty": args.uncertainty_index,
        "contribution": args.contribution_index,
    }
    stage_indexes: dict[str, dict[str, dict[str, Any]]] = {}
    stage_crop_rows: list[dict[str, Any]] = []
    for stage, path in supplied.items():
        if path is None:
            continue
        indexed = _load_stage(
            stage,
            path,
            manifest,
            full_manifest,
            frozen.sha256,
        )
        stage_indexes[stage] = indexed
        stage_crop_rows.extend(_stage_rows(stage, indexed, manifest))
    result_index_sha256 = sha256_file(args.result_index)
    for stage in ("uncertainty", "contribution"):
        for crop_id, row in stage_indexes.get(stage, {}).items():
            if row.get("source_result_index_sha256") != result_index_sha256:
                raise ValueError(
                    f"{stage} index does not bind the supplied localization index for {crop_id}"
                )

    crop_runtime = pd.DataFrame(stage_crop_rows)
    crop_runtime["executed_runtime_seconds"] = crop_runtime["runtime_seconds"].where(
        crop_runtime["executed_in_this_run"]
    )
    runtime_summary = _runtime_summary(crop_runtime)
    tree_runtime = (
        crop_runtime.groupby(["stage", "tree_id"], sort=True)
        .agg(
            manifest_crops=("crop_id", "size"),
            successful_crops=("success", "sum"),
            executed_crops=("executed_in_this_run", "sum"),
            worker_seconds_total=("executed_runtime_seconds", "sum"),
            runtime_p90_seconds=(
                "executed_runtime_seconds",
                lambda value: value.dropna().quantile(0.90),
            ),
        )
        .reset_index()
    )
    diagnostics = _localization_diagnostics(
        stage_indexes["localization"],
        manifest,
        frozen.sha256,
        (Path(args.result_index).expanduser().resolve().parent / "predictions").resolve(),
    )
    diagnostic_summary = _diagnostic_summary(diagnostics, len(manifest))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    crop_runtime.to_csv(output / "stage_crop_runtime.csv", index=False)
    tree_runtime.to_csv(output / "stage_tree_runtime.csv", index=False)
    runtime_summary.to_csv(output / "runtime_summary.csv", index=False)
    diagnostics.to_csv(output / "localization_search_diagnostics.csv", index=False)
    diagnostic_summary.to_csv(output / "search_trigger_summary.csv", index=False)
    output_artifacts = {
        "stage_crop_runtime": output / "stage_crop_runtime.csv",
        "stage_tree_runtime": output / "stage_tree_runtime.csv",
        "runtime_summary": output / "runtime_summary.csv",
        "localization_search_diagnostics": output
        / "localization_search_diagnostics.csv",
        "search_trigger_summary": output / "search_trigger_summary.csv",
    }
    selected_splits = sorted(
        {str(source["split"]) for source in manifest.values()}
    )
    stage_denominators = {
        stage: {
            "manifest_crops": len(manifest),
            "index_records": len(indexed),
            "missing_index_records": len(manifest) - len(indexed),
            "status_counts": {
                status: sum(
                    1
                    for crop_id in manifest
                    if (
                        str(indexed[crop_id]["status"])
                        if crop_id in indexed
                        else "MISSING_INDEX"
                    )
                    == status
                )
                for status in sorted(
                    {
                        str(indexed[crop_id]["status"])
                        if crop_id in indexed
                        else "MISSING_INDEX"
                        for crop_id in manifest
                    }
                )
            },
        }
        for stage, indexed in stage_indexes.items()
    }
    input_artifact_hashes = {
        "config_file": sha256_file(args.config),
        "crop_manifest": sha256_file(args.crop_manifest),
        "evidence_index": sha256_file(args.evidence_index),
        "result_index": result_index_sha256,
        "uncertainty_index": (
            sha256_file(args.uncertainty_index) if args.uncertainty_index else None
        ),
        "contribution_index": (
            sha256_file(args.contribution_index) if args.contribution_index else None
        ),
    }
    atomic_write_json(
        output / "runtime_denominator.json",
        {
            "schema_version": "racpith.runtime_denominator.v1",
            "config_hash": frozen.sha256,
            "splits": selected_splits,
            "manifest_crops": len(manifest),
            "trees": len(selected_tree_ids),
            "stages": list(stage_indexes),
            "stage_denominators": stage_denominators,
            "source_result_index_sha256": result_index_sha256,
            "input_artifact_hashes": input_artifact_hashes,
            "input_index_sha256": {
                stage: sha256_file(path)
                for stage, path in supplied.items()
                if path is not None
            },
            "output_artifact_hashes": {
                name: sha256_file(path) for name, path in output_artifacts.items()
            },
            "timing_semantics": (
                "per-worker elapsed seconds recorded by each stage; RESUMED and "
                "unexecuted rows are excluded from latency quantiles"
            ),
        },
    )
    print(runtime_summary.to_string(index=False))


if __name__ == "__main__":
    main()
