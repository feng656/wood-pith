#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from racpith.config import load_config
from racpith.estimator import RacPithEstimator
from racpith.evaluation.metrics import (
    build_crop_metrics,
    cluster_bootstrap_summary,
    risk_coverage_curve,
    summarize_localization,
    tree_balanced_quantile,
)
from racpith.provenance import atomic_write_json, sha256_file


USABLE_STATES = {"POINT", "RANGE", "RAY", "AXIS"}


def _tree_mean_rate(frame, mask_column: str) -> float:
    return float(frame.assign(_value=frame[mask_column].astype(float)).groupby("tree_id")["_value"].mean().mean())


def _tree_mean_conditional_point_rate(frame, column: str) -> float:
    point = frame[frame["state"] == "POINT"]
    if point.empty:
        return float("nan")
    return float(point.groupby("tree_id")[column].mean().mean())


def _tree_mean_observed_rate(frame, column: str) -> float:
    observed = frame[frame[column].notna()]
    if observed.empty:
        return float("nan")
    return float(observed.groupby("tree_id")[column].mean().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAC-Pith outputs with full denominators")
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--result-index",
        required=True,
        help="immutable result index used to exclude stale or unregistered prediction files",
    )
    parser.add_argument(
        "--evidence-index",
        required=True,
        help="content-addressed evidence used only to score GT support membership, never to refit",
    )
    parser.add_argument(
        "--uncertainty-index",
        help="optional structured-replay index; when supplied its conservative adjudicated_state is final",
    )
    parser.add_argument("--target-domain-index")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    args = parser.parse_args()
    frozen = load_config(args.config)
    support_scorer = RacPithEstimator(frozen, run_id="evaluation-support-score")
    state_cfg = frozen.section("state")
    metrics = build_crop_metrics(
        args.crop_manifest,
        args.predictions,
        float(state_cfg["false_point_tolerance_norm"]),
        float(state_cfg["catastrophic_tolerance_norm"]),
        uncertainty_index_path=args.uncertainty_index,
        splits=set(args.split) if args.split else None,
        target_domain_index_path=args.target_domain_index,
        expected_config_hash=frozen.sha256,
        result_index_path=args.result_index,
        evidence_index_path=args.evidence_index,
        estimator=support_scorer,
    )
    calibration_status = str(frozen.data.get("calibration_status", "UNKNOWN"))
    calibration_frozen = bool(
        "FROZEN" in calibration_status.upper()
        and "PROVISIONAL" not in calibration_status.upper()
    )
    metrics["core_production_usable"] = metrics["production_usable"].astype(bool)
    metrics["production_usable"] = (
        (metrics["state"] == "POINT")
        & metrics["search_adequate"].astype(bool)
        & metrics["model_risk"].isin({"LOW", "LOW_RISK"})
        & (metrics["target_domain"] == "TARGET_ALIGNED")
        & ~metrics["uncertainty_reaudit_required"].fillna(False).astype(bool)
        & calibration_frozen
    )
    metrics["production_usable_adjudication"] = (
        "evaluation-side frozen geometry + structured replay + model risk + target domain"
    )
    tree, dataset = summarize_localization(metrics)
    risk = risk_coverage_curve(metrics)
    evaluation_cfg = frozen.section("evaluation")
    replicates = int(evaluation_cfg["bootstrap_replicates"])
    confidence = float(evaluation_cfg["confidence_level"])
    seed = int(frozen.data["random_seed"])
    statistics = {
        "point_coverage_tree_mean": lambda frame: float(
            frame.assign(_point=frame["state"] == "POINT")
            .groupby("tree_id")["_point"]
            .mean()
            .mean()
        ),
        "usable_coverage_tree_mean": lambda frame: float(
            frame.assign(_usable=frame["state"].isin(USABLE_STATES))
            .groupby("tree_id")["_usable"]
            .mean()
            .mean()
        ),
        "production_usable_coverage_tree_mean": lambda frame: float(
            frame.assign(_usable=frame["production_usable"].astype(bool))
            .groupby("tree_id")["_usable"]
            .mean()
            .mean()
        ),
        "false_point_risk_tree_mean": lambda frame: _tree_mean_conditional_point_rate(
            frame, "false_point"
        ),
        "catastrophic_point_risk_tree_mean": lambda frame: _tree_mean_conditional_point_rate(
            frame, "catastrophic_point"
        ),
        "support_coverage_tree_mean": lambda frame: _tree_mean_observed_rate(
            frame, "support_contains_gt"
        ),
        "point_support_coverage_tree_mean": lambda frame: _tree_mean_conditional_point_rate(
            frame[frame["support_contains_gt"].notna()], "support_contains_gt"
        ),
        "point_stability_ellipse_coverage_tree_mean": lambda frame: _tree_mean_conditional_point_rate(
            frame[frame["stability_ellipse_contains_gt"].notna()],
            "stability_ellipse_contains_gt",
        ),
        "point_p90_error_norm_tree_balanced": lambda frame: tree_balanced_quantile(
            frame[frame["state"] == "POINT"], "point_error_norm", 0.90
        ),
        "point_p95_error_norm_tree_balanced": lambda frame: tree_balanced_quantile(
            frame[frame["state"] == "POINT"], "point_error_norm", 0.95
        ),
    }
    interval_rows = []
    for offset, (name, statistic) in enumerate(statistics.items()):
        interval_rows.append(
            {
                "metric": name,
                **cluster_bootstrap_summary(
                    metrics,
                    statistic,
                    replicates=replicates,
                    confidence_level=confidence,
                    seed=seed + offset,
                ),
                "cluster_unit": "tree_id",
                "replicates": replicates,
                "confidence_level": confidence,
            }
        )
    import pandas as pd

    intervals = pd.DataFrame(interval_rows)
    strata = (
        metrics.assign(
            point=metrics["state"] == "POINT",
            usable=metrics["state"].isin(USABLE_STATES),
        )
        .groupby("distance_stratum", dropna=False)
        .agg(
            crops=("crop_id", "size"),
            trees=("tree_id", "nunique"),
            point_coverage=("point", "mean"),
            usable_coverage=("usable", "mean"),
            point_median_error_norm=("point_error_norm", "median"),
            point_p90_error_norm=("point_error_norm", lambda values: values.quantile(0.90)),
            false_points=("false_point", "sum"),
        )
        .reset_index()
    )
    target_crop_summary = (
        metrics.assign(
            point=metrics["state"] == "POINT",
            false_point_flag=metrics["false_point"].astype(bool),
        )
        .groupby("target_domain", dropna=False)
        .agg(
            crops=("crop_id", "size"),
            trees=("tree_id", "nunique"),
            sections=("section_id", "nunique"),
            point_coverage=("point", "mean"),
            point_median_error_norm=("point_error_norm", "median"),
            point_p90_error_norm=("point_error_norm", lambda values: values.quantile(0.90)),
            false_points=("false_point_flag", "sum"),
            median_target_bias_norm=("target_bias_norm", "median"),
        )
        .reset_index()
    )
    target_tree = (
        metrics.assign(
            point=metrics["state"] == "POINT",
            usable=metrics["state"].isin(USABLE_STATES),
            false_point_value=metrics["false_point"].where(
                metrics["state"] == "POINT"
            ),
        )
        .groupby(["target_domain", "tree_id"], dropna=False)
        .agg(
            crops=("crop_id", "size"),
            sections=("section_id", "nunique"),
            point_coverage=("point", "mean"),
            usable_coverage=("usable", "mean"),
            false_point_risk=("false_point_value", "mean"),
            point_median_error_norm=("point_error_norm", "median"),
            point_p90_error_norm=("point_error_norm", lambda values: values.quantile(0.90)),
            median_target_bias_norm=("target_bias_norm", "median"),
        )
        .reset_index()
    )
    state_tree = (
        metrics.assign(
            support_value=metrics["support_contains_gt"],
            radial_value=metrics["radial_interval_contains_gt"],
        )
        .groupby(["state", "tree_id"], dropna=False)
        .agg(
            crops=("crop_id", "size"),
            sections=("section_id", "nunique"),
            support_coverage=("support_value", "mean"),
            radial_interval_coverage=("radial_value", "mean"),
            median_direction_error_deg=("direction_error_deg", "median"),
            median_point_error_norm=("point_error_norm", "median"),
            p90_point_error_norm=("point_error_norm", lambda values: values.quantile(0.90)),
            median_range_width_norm=("range_width_norm", "median"),
            unbounded_fraction=("range_unbounded", "mean"),
            median_finite_mode_count=("finite_mode_count", "median"),
        )
        .reset_index()
    )
    state_summary = (
        state_tree.groupby("state", dropna=False)
        .agg(
            trees=("tree_id", "nunique"),
            crops=("crops", "sum"),
            support_coverage_tree_mean=("support_coverage", "mean"),
            radial_interval_coverage_tree_mean=("radial_interval_coverage", "mean"),
            median_direction_error_deg_tree_mean=("median_direction_error_deg", "mean"),
            median_point_error_norm_tree_mean=("median_point_error_norm", "mean"),
            p90_point_error_norm_tree_mean=("p90_point_error_norm", "mean"),
            median_range_width_norm_tree_mean=("median_range_width_norm", "mean"),
            unbounded_fraction_tree_mean=("unbounded_fraction", "mean"),
            median_finite_mode_count_tree_mean=("median_finite_mode_count", "mean"),
        )
        .reset_index()
    )
    target_summary = (
        target_tree.groupby("target_domain", dropna=False)
        .agg(
            trees=("tree_id", "nunique"),
            point_coverage_tree_mean=("point_coverage", "mean"),
            usable_coverage_tree_mean=("usable_coverage", "mean"),
            false_point_risk_tree_mean=("false_point_risk", "mean"),
            point_median_error_norm_tree_mean=("point_median_error_norm", "mean"),
            point_p90_error_norm_tree_mean=("point_p90_error_norm", "mean"),
            median_target_bias_norm_tree_mean=("median_target_bias_norm", "mean"),
        )
        .reset_index()
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "crop_metrics.csv", index=False)
    tree.to_csv(output / "tree_metrics.csv", index=False)
    dataset.to_csv(output / "dataset_summary.csv", index=False)
    risk.to_csv(output / "risk_coverage.csv", index=False)
    intervals.to_csv(output / "tree_cluster_bootstrap_intervals.csv", index=False)
    strata.to_csv(output / "distance_stratum_summary.csv", index=False)
    target_summary.to_csv(output / "target_domain_summary.csv", index=False)
    target_tree.to_csv(output / "target_domain_tree_metrics.csv", index=False)
    target_crop_summary.to_csv(output / "target_domain_crop_descriptive.csv", index=False)
    state_tree.to_csv(output / "state_tree_metrics.csv", index=False)
    state_summary.to_csv(output / "state_specific_summary.csv", index=False)
    if "geometry_state" in metrics.columns:
        pd.crosstab(metrics["geometry_state"], metrics["state"], dropna=False).to_csv(
            output / "uncertainty_state_transitions.csv"
        )
    output_artifacts = {
        "crop_metrics": output / "crop_metrics.csv",
        "tree_metrics": output / "tree_metrics.csv",
        "dataset_summary": output / "dataset_summary.csv",
        "risk_coverage": output / "risk_coverage.csv",
        "tree_cluster_bootstrap_intervals": output
        / "tree_cluster_bootstrap_intervals.csv",
        "distance_stratum_summary": output / "distance_stratum_summary.csv",
        "target_domain_summary": output / "target_domain_summary.csv",
        "target_domain_tree_metrics": output / "target_domain_tree_metrics.csv",
        "target_domain_crop_descriptive": output
        / "target_domain_crop_descriptive.csv",
        "state_tree_metrics": output / "state_tree_metrics.csv",
        "state_specific_summary": output / "state_specific_summary.csv",
    }
    transition_path = output / "uncertainty_state_transitions.csv"
    if transition_path.is_file():
        output_artifacts["uncertainty_state_transitions"] = transition_path
    denominator = {
        "schema_version": "racpith.denominator.v1",
        "manifest_crops": len(metrics),
        "trees": int(metrics["tree_id"].nunique()),
        "sections": int(metrics["section_id"].nunique()),
        "missing_results": int((metrics["result_status"] != "FINISHED").sum()),
        "uncertainty_reaudit_required": int(
            metrics.get("uncertainty_reaudit_required", False).fillna(False).sum()
            if "uncertainty_reaudit_required" in metrics
            else 0
        ),
        "production_usable": int(metrics["production_usable"].astype(bool).sum()),
        "core_production_usable": int(
            metrics["core_production_usable"].astype(bool).sum()
        ),
        "target_domain_counts": {
            str(key): int(value)
            for key, value in metrics["target_domain"].value_counts(dropna=False).items()
        },
        "target_domain_reference": {
            "reference_manifest_sha256": sorted(
                {
                    str(value)
                    for value in metrics["target_reference_manifest_sha256"].dropna()
                }
            ),
            "reference_result_index_sha256": sorted(
                {
                    str(value)
                    for value in metrics[
                        "target_reference_result_index_sha256"
                    ].dropna()
                }
            ),
            "gt_used_only_posthoc": True,
        },
        "calibration_status": calibration_status,
        "support_scoring": {
            "definition": "GT profiled objective <= saved global reference + frozen support delta",
            "gt_used_to_refit": False,
            "evaluable": int(metrics["core_support_contains_gt"].notna().sum()),
            "contained": int(metrics["core_support_contains_gt"].fillna(False).astype(bool).sum()),
        },
        "stability_ellipse_evaluable": int(
            metrics["stability_ellipse_contains_gt"].notna().sum()
        ),
        "input_artifact_hashes": {
            "crop_manifest": sha256_file(args.crop_manifest),
            "evidence_index": sha256_file(args.evidence_index),
            "result_index": sha256_file(args.result_index),
            "uncertainty_index": (
                sha256_file(args.uncertainty_index) if args.uncertainty_index else None
            ),
            "target_domain_index": (
                sha256_file(args.target_domain_index) if args.target_domain_index else None
            ),
        },
        "output_artifact_hashes": {
            name: sha256_file(path) for name, path in output_artifacts.items()
        },
        "state_counts": {str(key): int(value) for key, value in metrics["state"].value_counts().items()},
        "config_hash": frozen.sha256,
    }
    atomic_write_json(output / "denominators.json", denominator)
    print(dataset.to_string(index=False))


if __name__ == "__main__":
    main()
