#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from racpith.baselines import BASELINE_METHODS
from racpith.config import load_config
from racpith.evaluation.metrics import (
    CROP_CONTENT_HASH_FIELDS,
    load_evidence_index,
    validate_manifest_crop_content,
)
from racpith.provenance import atomic_write_json, read_jsonl, sha256_file


REFERENCE_METHOD = "B5_robust_varpro_far_profile_state"


def _paired_tree_interval(
    tree: pd.DataFrame,
    comparator: str,
    metric: str,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    pivot = tree.pivot(index="tree_id", columns="method", values=metric)
    if REFERENCE_METHOD not in pivot or comparator not in pivot:
        return {
            "paired_trees": 0,
            "difference_b5_minus_comparator": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    paired = pivot[[REFERENCE_METHOD, comparator]].dropna()
    if paired.empty:
        return {
            "paired_trees": 0,
            "difference_b5_minus_comparator": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    differences = (
        paired[REFERENCE_METHOD] - paired[comparator]
    ).to_numpy(dtype=float)
    estimate = float(np.mean(differences))
    if replicates < 1:
        return {
            "paired_trees": len(differences),
            "difference_b5_minus_comparator": estimate,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draw = rng.integers(0, len(differences), size=len(differences))
        samples[index] = float(np.mean(differences[draw]))
    alpha = 1.0 - confidence_level
    return {
        "paired_trees": len(differences),
        "difference_b5_minus_comparator": estimate,
        "ci_low": float(np.quantile(samples, alpha / 2.0)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-balanced B0-B5 comparison")
    parser.add_argument("--baselines", required=True)
    parser.add_argument(
        "--evidence-index",
        required=True,
        help="immutable evidence registry used to verify every finished baseline",
    )
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    args = parser.parse_args()
    frozen = load_config(args.config)
    tolerance = float(frozen.section("state")["false_point_tolerance_norm"])
    all_manifest = {}
    for row in read_jsonl(args.crop_manifest):
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported crop-manifest schema")
        crop_id = str(row["crop_id"])
        if crop_id in all_manifest:
            raise ValueError(f"duplicate crop-manifest row for {crop_id}")
        all_manifest[crop_id] = row
    manifest = {
        crop_id: row
        for crop_id, row in all_manifest.items()
        if not args.split or row["split"] in set(args.split)
    }
    if not manifest:
        raise ValueError("selected crop-manifest scope is empty")
    for row in manifest.values():
        validate_manifest_crop_content(row)
    all_evidence = load_evidence_index(
        args.evidence_index,
        expected_config_hash=frozen.sha256,
    )
    orphan_evidence = sorted(set(all_evidence) - set(all_manifest))
    if orphan_evidence:
        raise ValueError(f"orphan evidence-index crops: {orphan_evidence[:10]}")
    evidence_by_crop = {}
    for crop_id, row in all_evidence.items():
        if crop_id not in manifest:
            continue
        if any(
            str(row.get(field)) != str(manifest[crop_id].get(field))
            for field in (
                "tree_id",
                "section_id",
                "split",
                *CROP_CONTENT_HASH_FIELDS,
            )
        ):
            raise ValueError(f"evidence/manifest lineage mismatch for crop {crop_id}")
        evidence_by_crop[crop_id] = row
    baseline_by_key = {}
    for result in read_jsonl(args.baselines):
        if result.get("schema_version") != "racpith.baseline.v1":
            raise ValueError("unsupported baseline record schema")
        crop_id = str(result["crop_id"])
        method = str(result["method"])
        source = all_manifest.get(crop_id)
        if source is None:
            raise ValueError(f"orphan baseline record for crop {crop_id}")
        if method not in BASELINE_METHODS:
            raise ValueError(f"unregistered baseline method {method!r}")
        if result.get("config_hash") != frozen.sha256:
            raise ValueError(f"baseline/config hash mismatch: {(crop_id, method)}")
        if any(
            str(result.get(field)) != str(source.get(field))
            for field in ("tree_id", "section_id", "split")
        ):
            raise ValueError(f"baseline/manifest lineage mismatch: {(crop_id, method)}")
        if crop_id not in manifest:
            continue
        key = (crop_id, method)
        if key in baseline_by_key:
            raise ValueError(f"duplicate baseline record: {key}")
        status = str(result.get("status"))
        if status not in {"FINISHED", "CRASH", "UPSTREAM_EVIDENCE_FAIL"}:
            raise ValueError(f"unsupported baseline status {status!r}: {key}")
        evidence = evidence_by_crop.get(crop_id)
        if status == "FINISHED":
            if evidence is None or evidence.get("status") != "PASS":
                raise ValueError(f"finished baseline lacks PASS evidence: {key}")
            if (
                result.get("evidence_metadata_sha256")
                != evidence.get("metadata_sha256")
                or result.get("evidence_npz_sha256") != evidence.get("npz_sha256")
            ):
                raise ValueError(f"baseline/evidence hash mismatch: {key}")
        baseline_by_key[key] = result
    rows = []
    for crop_id, source in sorted(manifest.items()):
        width, height = source["crop_size_px"]
        scale = float(source.get("normalization_scale_px", np.hypot(width, height)))
        gt = (np.asarray(source["pith_crop_px"], dtype=float) - [width / 2.0, height / 2.0]) / scale
        for method in BASELINE_METHODS:
            result = baseline_by_key.get((crop_id, method))
            finished = result is not None and result.get("status") == "FINISHED"
            center = result.get("raw_center_norm") if result is not None else None
            state = str(result.get("state", "REJECT")) if result is not None else "REJECT"
            converged = bool(result.get("converged", False)) if result is not None else False
            point = bool(finished and converged and state == "POINT" and center is not None)
            error = float(np.linalg.norm(np.asarray(center) - gt)) if point else np.nan
            rows.append(
                {
                    "method": method,
                    "crop_id": crop_id,
                    "tree_id": source["tree_id"],
                    "section_id": source["section_id"],
                    "split": source["split"],
                    "result_status": result.get("status") if result is not None else "MISSING_RESULT",
                    "state": state if finished else "REJECT",
                    "point": point,
                    "usable": bool(
                        finished
                        and (
                            (state == "POINT" and converged)
                            or state in {"RANGE", "RAY", "AXIS"}
                        )
                    ),
                    "error_norm": error,
                    "false_point": bool(point and error > tolerance),
                    "search_adequate": bool(result.get("search_adequate", False)) if result else False,
                }
            )
    crop = pd.DataFrame(rows)
    tree_rows = []
    for (method, tree_id), frame in crop.groupby(["method", "tree_id"], sort=True):
        point_frame = frame[frame["point"]]
        tree_rows.append(
            {
                "method": method,
                "tree_id": tree_id,
                "n_manifest": len(frame),
                "finished_rate": float((frame["result_status"] == "FINISHED").mean()),
                "point_coverage": float(frame["point"].mean()),
                "usable_coverage": float(frame["usable"].mean()),
                "median_error_norm": float(point_frame["error_norm"].median()) if len(point_frame) else np.nan,
                "p90_error_norm": float(point_frame["error_norm"].quantile(0.90)) if len(point_frame) else np.nan,
                "false_point_risk": float(point_frame["false_point"].mean()) if len(point_frame) else np.nan,
            }
        )
    tree = pd.DataFrame(tree_rows)
    summary = (
        tree.groupby("method")
        .agg(
            trees=("tree_id", "nunique"),
            finished_rate=("finished_rate", "mean"),
            point_coverage=("point_coverage", "mean"),
            usable_coverage=("usable_coverage", "mean"),
            median_error_norm=("median_error_norm", "mean"),
            p90_error_norm=("p90_error_norm", "mean"),
            false_point_risk=("false_point_risk", "mean"),
        )
        .reset_index()
    )
    evaluation = frozen.section("evaluation")
    replicates = int(evaluation["bootstrap_replicates"])
    confidence = float(evaluation["confidence_level"])
    comparison_rows = []
    metric_directions = {
        "finished_rate": True,
        "point_coverage": True,
        "usable_coverage": True,
        "median_error_norm": False,
        "p90_error_norm": False,
        "false_point_risk": False,
    }
    offset = 0
    for comparator in BASELINE_METHODS:
        if comparator == REFERENCE_METHOD:
            continue
        for metric, higher_is_better in metric_directions.items():
            comparison_rows.append(
                {
                    "reference_method": REFERENCE_METHOD,
                    "comparator_method": comparator,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    **_paired_tree_interval(
                        tree,
                        comparator,
                        metric,
                        replicates=replicates,
                        confidence_level=confidence,
                        seed=int(frozen.data["random_seed"]) + offset,
                    ),
                    "cluster_unit": "tree_id",
                    "bootstrap_replicates": replicates,
                    "confidence_level": confidence,
                }
            )
            offset += 1
    paired_comparisons = pd.DataFrame(comparison_rows)
    status_counts: dict[str, dict[str, int]] = {}
    for method, frame in crop.groupby("method", sort=True):
        status_counts[str(method)] = {
            str(status): int(count)
            for status, count in frame["result_status"].value_counts().items()
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    crop.to_csv(output / "baseline_crop_metrics.csv", index=False)
    tree.to_csv(output / "baseline_tree_metrics.csv", index=False)
    summary.to_csv(output / "baseline_summary.csv", index=False)
    paired_comparisons.to_csv(
        output / "baseline_paired_tree_differences.csv", index=False
    )
    output_artifacts = {
        "baseline_crop_metrics": output / "baseline_crop_metrics.csv",
        "baseline_tree_metrics": output / "baseline_tree_metrics.csv",
        "baseline_summary": output / "baseline_summary.csv",
        "baseline_paired_tree_differences": output
        / "baseline_paired_tree_differences.csv",
    }
    atomic_write_json(
        output / "baseline_denominator.json",
        {
            "schema_version": "racpith.baseline_denominator.v1",
            "config_hash": frozen.sha256,
            "manifest_crops": len(manifest),
            "trees": len({str(row["tree_id"]) for row in manifest.values()}),
            "methods": list(BASELINE_METHODS),
            "expected_records": len(manifest) * len(BASELINE_METHODS),
            "registered_records": len(baseline_by_key),
            "status_counts_by_method": status_counts,
            "input_artifact_hashes": {
                "crop_manifest": sha256_file(args.crop_manifest),
                "evidence_index": sha256_file(args.evidence_index),
                "baselines": sha256_file(args.baselines),
            },
            "output_artifact_hashes": {
                name: sha256_file(path) for name, path in output_artifacts.items()
            },
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
