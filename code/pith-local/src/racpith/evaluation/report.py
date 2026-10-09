from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..provenance import read_json_object, sha256_file


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No eligible records._"
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
        rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, rule, *rows])


def _count_true(values: pd.Series) -> int:
    """Count persisted booleans without treating the string ``"False"`` as true."""

    normalised = values.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes"}
    )
    return int(normalised.fillna(False).sum())


def _resolve_registered_path(root: Path, relative_path: object) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute():
        raise ValueError(f"registered artifact path must be relative: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"registered artifact escapes its root: {relative}")
    return path


def _verify_named_artifacts(
    root: Path, registry: object, *, context: str
) -> dict[str, Path]:
    if not isinstance(registry, dict):
        raise ValueError(f"{context} artifact registry is not an object")
    verified: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for name, entry in registry.items():
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise ValueError(f"invalid {context} artifact entry: {name}")
        path = _resolve_registered_path(root, entry["path"])
        if path in seen_paths:
            raise ValueError(f"{context} registers one artifact path more than once: {path}")
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{context} artifact changed after registration: {name}")
        verified[str(name)] = path
        seen_paths.add(path)
    return verified


def _verify_named_csv_hashes(
    root: Path, registry: object, *, context: str
) -> dict[str, Path]:
    if not isinstance(registry, dict):
        raise ValueError(f"{context} output hash registry is not an object")
    verified: dict[str, Path] = {}
    for name, expected_hash in registry.items():
        path = _resolve_registered_path(root, f"{name}.csv")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"{context} artifact changed after registration: {path.name}")
        verified[str(name)] = path
    return verified


def _verify_path_hashes(
    root: Path, registry: object, *, context: str
) -> dict[str, Path]:
    if not isinstance(registry, dict):
        raise ValueError(f"{context} artifact registry is not an object")
    verified: dict[str, Path] = {}
    for relative_path, expected_hash in registry.items():
        path = _resolve_registered_path(root, relative_path)
        if path in verified.values():
            raise ValueError(f"{context} registers one artifact path more than once: {path}")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"{context} artifact changed after registration: {relative_path}")
        verified[str(relative_path)] = path
    return verified


def _reject_unregistered(
    root: Path,
    registered: set[Path],
    *,
    suffixes: set[str],
    context: str,
) -> None:
    on_disk = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in suffixes
    }
    stale = sorted(str(path) for path in on_disk - {path.resolve() for path in registered})
    if stale:
        raise ValueError(f"unregistered/stale {context} artifacts: {stale[:10]}")


def _string_set(value: object, *, context: str) -> set[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{context} splits must be a list of non-empty strings")
    return set(value)


def build_markdown_report(
    metrics_dir: str | Path,
    figure_dir: str | Path,
    contribution_metrics_dir: str | Path | None = None,
    contribution_figure_dir: str | Path | None = None,
    baseline_metrics_dir: str | Path | None = None,
) -> str:
    metrics_root = Path(metrics_dir)
    figures = Path(figure_dir)
    visualization_index_path = figures / "visualization_index.json"
    if not visualization_index_path.is_file():
        raise FileNotFoundError("visualization_index.json is required")
    visualization_index = read_json_object(visualization_index_path)
    if visualization_index.get("schema_version") != "racpith.visualization_index.v1":
        raise ValueError("unsupported visualization-index schema")
    cohort_registry = visualization_index.get("cohort_artifacts", {})
    required_cohort_figures = {
        "cohort/state_by_distance.png",
        "cohort/condition_distance_state.png",
        "cohort/support_containment_by_state.png",
        "cohort/point_error_ecdf.png",
        "cohort/risk_coverage.png",
    }
    if not required_cohort_figures.issubset(cohort_registry):
        raise ValueError("visualization index lacks required cohort figures")
    cohort_figures = _verify_path_hashes(
        figures, cohort_registry, context="cohort visualization"
    )
    sample_figures = _verify_named_artifacts(
        figures,
        visualization_index.get("sample_artifacts", {}),
        context="sample visualization",
    )
    _reject_unregistered(
        figures,
        set(cohort_figures.values()) | set(sample_figures.values()),
        suffixes={".png"},
        context="localization visualization",
    )
    denominator = read_json_object(metrics_root / "denominators.json")
    if denominator.get("schema_version") != "racpith.denominator.v1":
        raise ValueError("unsupported localization denominator schema")
    if visualization_index.get("config_hash") != denominator.get("config_hash"):
        raise ValueError("visualization and localization denominator configurations differ")
    registered_outputs = denominator.get("output_artifact_hashes", {})
    required_localization_outputs = {
        "dataset_summary",
        "crop_metrics",
        "tree_metrics",
        "tree_cluster_bootstrap_intervals",
        "distance_stratum_summary",
        "target_domain_summary",
        "state_specific_summary",
    }
    if not required_localization_outputs.issubset(registered_outputs):
        raise ValueError("localization denominator lacks required output registrations")
    localization_artifacts = _verify_named_csv_hashes(
        metrics_root,
        registered_outputs,
        context="localization evaluation",
    )
    visualization_inputs = visualization_index.get("input_artifact_hashes", {})
    if visualization_inputs.get("crop_metrics") != sha256_file(
        metrics_root / "crop_metrics.csv"
    ) or visualization_inputs.get("denominators") != sha256_file(
        metrics_root / "denominators.json"
    ):
        raise ValueError("visualization index does not bind the supplied metric artifacts")
    denominator_inputs = denominator.get("input_artifact_hashes", {})
    for name in (
        "crop_manifest",
        "evidence_index",
        "result_index",
        "uncertainty_index",
        "target_domain_index",
    ):
        if visualization_inputs.get(name) != denominator_inputs.get(name):
            raise ValueError(f"visualization/localization {name} lineage differs")
    dataset = pd.read_csv(metrics_root / "dataset_summary.csv")
    crops = pd.read_csv(metrics_root / "crop_metrics.csv")
    trees = pd.read_csv(metrics_root / "tree_metrics.csv")
    intervals = (
        pd.read_csv(metrics_root / "tree_cluster_bootstrap_intervals.csv")
        if (metrics_root / "tree_cluster_bootstrap_intervals.csv").exists()
        else pd.DataFrame()
    )
    strata = (
        pd.read_csv(metrics_root / "distance_stratum_summary.csv")
        if (metrics_root / "distance_stratum_summary.csv").exists()
        else pd.DataFrame()
    )
    target_domain = (
        pd.read_csv(metrics_root / "target_domain_summary.csv")
        if (metrics_root / "target_domain_summary.csv").exists()
        else pd.DataFrame()
    )
    state_specific = (
        pd.read_csv(metrics_root / "state_specific_summary.csv")
        if (metrics_root / "state_specific_summary.csv").exists()
        else pd.DataFrame()
    )
    required_crop_columns = {
        "crop_id",
        "tree_id",
        "section_id",
        "split",
        "state",
        "result_status",
    }
    missing_crop_columns = sorted(required_crop_columns - set(crops.columns))
    if missing_crop_columns:
        raise ValueError(f"crop metrics lack denominator columns: {missing_crop_columns}")
    if crops["crop_id"].astype(str).duplicated().any():
        raise ValueError("crop metrics contain duplicate crop_id rows")
    localization_splits = set(crops["split"].astype(str))
    if not localization_splits:
        raise ValueError("localization evaluation split scope is empty")
    if len(crops) != int(denominator.get("manifest_crops", -1)):
        raise ValueError("localization crop denominator disagrees with crop_metrics.csv")
    if crops["tree_id"].astype(str).nunique() != int(denominator.get("trees", -1)):
        raise ValueError("localization tree denominator disagrees with crop_metrics.csv")
    if crops["section_id"].astype(str).nunique() != int(
        denominator.get("sections", -1)
    ):
        raise ValueError("localization section denominator disagrees with crop_metrics.csv")
    if int((crops["result_status"] != "FINISHED").sum()) != int(
        denominator.get("missing_results", -1)
    ):
        raise ValueError("localization missing-result denominator disagrees with crop metrics")
    crop_state_counts = {
        str(key): int(value)
        for key, value in crops["state"].value_counts().items()
    }
    if crop_state_counts != denominator.get("state_counts"):
        raise ValueError("localization state denominator disagrees with crop metrics")
    if "tree_id" not in trees or trees["tree_id"].astype(str).nunique() != int(
        denominator.get("trees", -1)
    ):
        raise ValueError("tree_metrics.csv does not cover the localization tree denominator")
    runtime_path = metrics_root / "runtime_summary.csv"
    search_trigger_path = metrics_root / "search_trigger_summary.csv"
    runtime_denominator_path = metrics_root / "runtime_denominator.json"
    runtime_output_names = {
        "stage_crop_runtime",
        "stage_tree_runtime",
        "runtime_summary",
        "localization_search_diagnostics",
        "search_trigger_summary",
    }
    runtime_files_present = any(
        (metrics_root / f"{name}.csv").exists() for name in runtime_output_names
    )
    if runtime_files_present and not runtime_denominator_path.is_file():
        raise FileNotFoundError(
            "runtime_denominator.json is required when runtime outputs are present"
        )
    runtime_denominator: dict[str, object] | None = None
    runtime_artifacts: dict[str, Path] = {}
    if runtime_denominator_path.is_file():
        runtime_denominator = read_json_object(runtime_denominator_path)
        if runtime_denominator.get("schema_version") != "racpith.runtime_denominator.v1":
            raise ValueError("unsupported runtime denominator schema")
        if runtime_denominator.get("config_hash") != denominator.get("config_hash"):
            raise ValueError("runtime/localization configurations differ")
        if not runtime_output_names.issubset(
            runtime_denominator.get("output_artifact_hashes", {})
        ):
            raise ValueError("runtime denominator lacks required output registrations")
        runtime_artifacts = _verify_named_csv_hashes(
            metrics_root,
            runtime_denominator.get("output_artifact_hashes", {}),
            context="runtime evaluation",
        )
        if _string_set(runtime_denominator.get("splits"), context="runtime") != localization_splits:
            raise ValueError("runtime/localization split scopes differ")
        if int(runtime_denominator.get("manifest_crops", -1)) != len(crops):
            raise ValueError("runtime/localization crop denominators differ")
        if int(runtime_denominator.get("trees", -1)) != int(denominator["trees"]):
            raise ValueError("runtime/localization tree denominators differ")
        runtime_inputs = runtime_denominator.get("input_artifact_hashes", {})
        if not isinstance(runtime_inputs, dict):
            raise ValueError("runtime input-artifact registry is not an object")
        for name in (
            "crop_manifest",
            "evidence_index",
            "result_index",
            "uncertainty_index",
        ):
            if runtime_inputs.get(name) != denominator_inputs.get(name):
                raise ValueError(
                    f"runtime/localization {name} lineage differs"
                )
        if runtime_denominator.get("source_result_index_sha256") != denominator_inputs.get(
            "result_index"
        ):
            raise ValueError("runtime denominator does not bind the localization result index")
        stages = runtime_denominator.get("stages")
        stage_denominators = runtime_denominator.get("stage_denominators")
        if not isinstance(stages, list) or not isinstance(stage_denominators, dict):
            raise ValueError("runtime stage denominator registry is malformed")
        if set(stages) != set(stage_denominators):
            raise ValueError("runtime stage list and stage denominators differ")
        for stage, stage_denominator in stage_denominators.items():
            if not isinstance(stage_denominator, dict):
                raise ValueError(f"runtime {stage} denominator is not an object")
            if int(stage_denominator.get("manifest_crops", -1)) != len(crops):
                raise ValueError(f"runtime {stage} crop denominator is incomplete")
            status_counts = stage_denominator.get("status_counts")
            if not isinstance(status_counts, dict) or sum(
                int(value) for value in status_counts.values()
            ) != len(crops):
                raise ValueError(f"runtime {stage} status denominator is incomplete")
    _reject_unregistered(
        metrics_root,
        set(localization_artifacts.values()) | set(runtime_artifacts.values()),
        suffixes={".csv"},
        context="evaluation",
    )
    runtime = pd.read_csv(runtime_path) if runtime_path.is_file() else pd.DataFrame()
    search_triggers = (
        pd.read_csv(search_trigger_path) if search_trigger_path.is_file() else pd.DataFrame()
    )
    contribution_root = Path(contribution_metrics_dir) if contribution_metrics_dir else None
    contribution_artifacts: dict[str, Path] = {}
    contribution_crop_denominators = pd.DataFrame()
    contribution_evaluation_index: dict[str, object] | None = None
    if contribution_root is not None:
        contribution_evaluation_index_path = (
            contribution_root / "contribution_evaluation_index.json"
        )
        if not contribution_evaluation_index_path.is_file():
            raise FileNotFoundError("contribution_evaluation_index.json is required")
        contribution_evaluation_index = read_json_object(
            contribution_evaluation_index_path
        )
        if (
            contribution_evaluation_index.get("schema_version")
            != "racpith.contribution_evaluation_index.v1"
        ):
            raise ValueError("unsupported contribution-evaluation index schema")
        if contribution_evaluation_index.get("config_hash") != denominator.get(
            "config_hash"
        ):
            raise ValueError("contribution/localization configurations differ")
        expected_result_hash = denominator_inputs.get("result_index")
        if contribution_evaluation_index.get("source_result_index_sha256") != [
            expected_result_hash
        ]:
            raise ValueError(
                "contribution analysis does not bind the localization result index"
            )
        contribution_artifacts = _verify_named_artifacts(
            contribution_root,
            contribution_evaluation_index.get("output_artifacts", {}),
            context="contribution evaluation",
        )
        required_contribution_outputs = {
            "contribution_crop_denominators",
            "contribution_group_denominators",
            "contribution_evaluation_status",
        }
        if not required_contribution_outputs.issubset(contribution_artifacts):
            raise ValueError("contribution evaluation lacks required denominators")
        contribution_status = contribution_evaluation_index.get("status")
        if contribution_status not in {"PASS", "NO_GROUP_RECORDS"}:
            raise ValueError("unsupported contribution-evaluation status")
        if contribution_status == "PASS":
            pass_outputs = {
                "formal_sign_distribution",
                "point_sign_diagnostic_distribution",
                "contribution_tree_metrics",
                "functional_role_counts",
                "contribution_identity_audit",
                "crossfit_exact_association",
                "scale_phase_stability",
                "most_harmful_exact_groups",
                "most_beneficial_exact_groups",
                "contribution_tree_cluster_bootstrap_intervals",
            }
            if not pass_outputs.issubset(contribution_artifacts):
                raise ValueError("PASS contribution evaluation lacks registered outputs")
        contribution_inputs = contribution_evaluation_index.get(
            "input_artifact_hashes", {}
        )
        if not isinstance(contribution_inputs, dict):
            raise ValueError("contribution input-artifact registry is not an object")
        if contribution_inputs.get("crop_manifest") != denominator_inputs.get(
            "crop_manifest"
        ):
            raise ValueError("contribution/localization crop-manifest lineage differs")
        if _string_set(
            contribution_evaluation_index.get("splits"), context="contribution evaluation"
        ) != localization_splits:
            raise ValueError("contribution/localization split scopes differ")
        for field, expected in (
            ("manifest_crops", len(crops)),
            ("trees", int(denominator["trees"])),
            ("sections", int(denominator["sections"])),
            ("crop_index_records", len(crops)),
        ):
            if int(contribution_evaluation_index.get(field, -1)) != expected:
                raise ValueError(f"contribution {field} denominator differs")
        contribution_status_counts = contribution_evaluation_index.get("status_counts")
        if not isinstance(contribution_status_counts, dict) or sum(
            int(value) for value in contribution_status_counts.values()
        ) != len(crops):
            raise ValueError("contribution status denominator is incomplete")
        contribution_crop_denominators = pd.read_csv(
            contribution_artifacts["contribution_crop_denominators"]
        )
        if int(contribution_crop_denominators["crops"].sum()) != len(crops):
            raise ValueError("contribution crop denominator CSV is incomplete")
        if int(contribution_crop_denominators["groups"].sum()) != int(
            contribution_evaluation_index.get("group_records", -1)
        ):
            raise ValueError("contribution group-record denominator disagrees with crop index")
        if runtime_denominator is not None:
            runtime_contribution_hash = runtime_denominator.get(
                "input_artifact_hashes", {}
            ).get("contribution_index")
            contribution_input_hash = contribution_evaluation_index.get(
                "input_artifact_hashes", {}
            ).get("contribution_index")
            if (
                runtime_contribution_hash is not None
                and runtime_contribution_hash != contribution_input_hash
            ):
                raise ValueError(
                    "runtime/contribution evaluation index lineage differs"
                )
        _reject_unregistered(
            contribution_root,
            set(contribution_artifacts.values()) | {contribution_evaluation_index_path},
            suffixes={".csv", ".json"},
            context="contribution evaluation",
        )
    contribution_association = (
        pd.read_csv(contribution_artifacts["crossfit_exact_association"])
        if "crossfit_exact_association" in contribution_artifacts
        else pd.DataFrame()
    )
    contribution_denominators = (
        pd.read_csv(contribution_artifacts["contribution_group_denominators"])
        if "contribution_group_denominators" in contribution_artifacts
        else pd.DataFrame()
    )
    if contribution_evaluation_index is not None:
        expected_group_records = int(
            contribution_evaluation_index.get("group_records", -1)
        )
        observed_group_records = (
            int(contribution_denominators["total_groups"].sum())
            if "total_groups" in contribution_denominators
            else 0
        )
        if observed_group_records != expected_group_records:
            raise ValueError("contribution group denominator CSV is incomplete")
    scale_phase_stability = (
        pd.read_csv(contribution_artifacts["scale_phase_stability"])
        if "scale_phase_stability" in contribution_artifacts
        else pd.DataFrame()
    )
    stability_summary = pd.DataFrame()
    if not scale_phase_stability.empty:
        stability_summary = pd.DataFrame(
            [
                {
                    "anchors": len(scale_phase_stability),
                    "trees": scale_phase_stability["tree_id"].nunique(),
                    "assessed": int(
                        (scale_phase_stability["stability_status"] == "ASSESSED").sum()
                    ),
                    "insufficient_partitions": int(
                        (
                            scale_phase_stability["stability_status"]
                            == "INSUFFICIENT_PARTITIONS"
                        ).sum()
                    ),
                    "scale_unstable": _count_true(
                        scale_phase_stability["scale_unstable"]
                    ),
                    "beneficial_harmful_sign_flips": _count_true(
                        scale_phase_stability["beneficial_harmful_sign_flip"]
                    ),
                }
            ]
        )
    contribution_figures = (
        Path(contribution_figure_dir).resolve() if contribution_figure_dir else None
    )
    if contribution_figures is not None:
        if contribution_evaluation_index is None:
            raise ValueError(
                "contribution figures require a contribution evaluation index"
            )
        contribution_visualization_path = (
            contribution_figures / "contribution_visualization_index.json"
        )
        if not contribution_visualization_path.is_file():
            raise FileNotFoundError(
                "contribution_visualization_index.json is required"
            )
        contribution_visualization = read_json_object(
            contribution_visualization_path
        )
        if (
            contribution_visualization.get("schema_version")
            != "racpith.contribution_visualization_index.v1"
            or contribution_visualization.get("config_hash")
            != contribution_evaluation_index.get("config_hash")
            or contribution_visualization.get("source_result_index_sha256")
            != contribution_evaluation_index.get("source_result_index_sha256")
        ):
            raise ValueError(
                "contribution visualization/evaluation provenance differs"
            )
        contribution_visualization_inputs = contribution_visualization.get(
            "input_artifact_hashes", {}
        )
        contribution_evaluation_inputs = contribution_evaluation_index.get(
            "input_artifact_hashes", {}
        )
        if not isinstance(contribution_visualization_inputs, dict) or not isinstance(
            contribution_evaluation_inputs, dict
        ):
            raise ValueError("contribution visualization input registry is malformed")
        for name in (
            "config_file",
            "crop_manifest",
            "contribution_index",
        ):
            if contribution_visualization_inputs.get(
                name
            ) != contribution_evaluation_inputs.get(name):
                raise ValueError(
                    f"contribution visualization/evaluation {name} lineage differs"
                )
        if _string_set(
            contribution_visualization.get("splits"),
            context="contribution visualization",
        ) != _string_set(
            contribution_evaluation_index.get("splits"),
            context="contribution evaluation",
        ):
            raise ValueError("contribution visualization/evaluation split scopes differ")
        for field in (
            "manifest_crops",
            "trees",
            "sections",
            "crop_index_records",
            "group_records",
        ):
            if contribution_visualization.get(field) != contribution_evaluation_index.get(
                field
            ):
                raise ValueError(
                    f"contribution visualization/evaluation {field} denominators differ"
                )
        if contribution_visualization.get("status_counts") != contribution_evaluation_index.get(
            "status_counts"
        ):
            raise ValueError(
                "contribution visualization/evaluation status denominators differ"
            )
        contribution_cohort_registry = contribution_visualization.get(
            "cohort_artifacts", {}
        )
        required_contribution_figures = {
            "cohort/formal_sign_distribution.png",
            "cohort/crossfit_vs_exact.png",
        }
        if not required_contribution_figures.issubset(
            contribution_cohort_registry
        ):
            raise ValueError(
                "contribution visualization index lacks required cohort figures"
            )
        contribution_cohort_paths = _verify_path_hashes(
            contribution_figures,
            contribution_cohort_registry,
            context="contribution cohort visualization",
        )
        contribution_overlay_paths = _verify_named_artifacts(
            contribution_figures,
            contribution_visualization.get("overlay_artifacts", {}),
            context="contribution overlay",
        )
        if int(contribution_visualization.get("rendered_overlays", -1)) != len(
            contribution_overlay_paths
        ):
            raise ValueError("contribution rendered-overlay denominator is inconsistent")
        _reject_unregistered(
            contribution_figures,
            set(contribution_cohort_paths.values())
            | set(contribution_overlay_paths.values()),
            suffixes={".png"},
            context="contribution visualization",
        )
    baseline_root = Path(baseline_metrics_dir) if baseline_metrics_dir else None
    if baseline_root is not None:
        baseline_denominator_path = baseline_root / "baseline_denominator.json"
        if not baseline_denominator_path.is_file():
            raise FileNotFoundError("baseline_denominator.json is required")
        baseline_denominator = read_json_object(baseline_denominator_path)
        if baseline_denominator.get("schema_version") != "racpith.baseline_denominator.v1":
            raise ValueError("unsupported baseline denominator schema")
        if baseline_denominator.get("config_hash") != denominator.get("config_hash"):
            raise ValueError("baseline/localization configurations differ")
        baseline_hashes = baseline_denominator.get("output_artifact_hashes", {})
        required_baseline_outputs = {
            "baseline_crop_metrics",
            "baseline_tree_metrics",
            "baseline_summary",
            "baseline_paired_tree_differences",
        }
        if not required_baseline_outputs.issubset(baseline_hashes):
            raise ValueError("baseline denominator lacks required output registrations")
        baseline_artifacts = _verify_named_csv_hashes(
            baseline_root,
            baseline_hashes,
            context="baseline evaluation",
        )
        baseline_inputs = baseline_denominator.get("input_artifact_hashes", {})
        if not isinstance(baseline_inputs, dict):
            raise ValueError("baseline input-artifact registry is not an object")
        for name in ("crop_manifest", "evidence_index"):
            if baseline_inputs.get(name) != denominator_inputs.get(name):
                raise ValueError(f"baseline/localization {name} lineage differs")
        baseline_crop_denominator = pd.read_csv(
            baseline_artifacts["baseline_crop_metrics"]
        )
        required_baseline_columns = {
            "crop_id",
            "tree_id",
            "split",
            "method",
            "result_status",
        }
        if not required_baseline_columns.issubset(baseline_crop_denominator.columns):
            raise ValueError("baseline crop metrics lack denominator columns")
        if set(baseline_crop_denominator["split"].astype(str)) != localization_splits:
            raise ValueError("baseline/localization split scopes differ")
        if baseline_crop_denominator["crop_id"].astype(str).nunique() != len(crops):
            raise ValueError("baseline/localization crop denominators differ")
        if baseline_crop_denominator["tree_id"].astype(str).nunique() != int(
            denominator["trees"]
        ):
            raise ValueError("baseline/localization tree denominators differ")
        if len(baseline_crop_denominator) != int(
            baseline_denominator.get("expected_records", -1)
        ):
            raise ValueError("baseline expected-record denominator is incomplete")
        baseline_methods = [
            str(method) for method in baseline_denominator.get("methods", [])
        ]
        if not baseline_methods or len(baseline_methods) != len(set(baseline_methods)):
            raise ValueError("baseline method registry is empty or duplicated")
        observed_methods = set(baseline_crop_denominator["method"].astype(str))
        if observed_methods != set(baseline_methods):
            raise ValueError("baseline crop table and method registry differ")
        if baseline_crop_denominator.duplicated(["crop_id", "method"]).any():
            raise ValueError("baseline crop table has duplicate crop/method records")
        expected_records = len(crops) * len(baseline_methods)
        if int(baseline_denominator.get("expected_records", -1)) != expected_records:
            raise ValueError("baseline manifest×method denominator is inconsistent")
        observed_status_counts: dict[str, dict[str, int]] = {}
        for method, frame in baseline_crop_denominator.groupby("method", sort=True):
            observed_status_counts[str(method)] = {
                str(status): int(count)
                for status, count in frame["result_status"].value_counts().items()
            }
        declared_status_counts = baseline_denominator.get("status_counts_by_method")
        if declared_status_counts != observed_status_counts:
            raise ValueError("baseline status denominators differ from the crop table")
        observed_registered = int(
            (baseline_crop_denominator["result_status"] != "MISSING_RESULT").sum()
        )
        if observed_registered != int(
            baseline_denominator.get("registered_records", -1)
        ):
            raise ValueError("baseline registered-record denominator is inconsistent")
        _reject_unregistered(
            baseline_root,
            set(baseline_artifacts.values()),
            suffixes={".csv"},
            context="baseline evaluation",
        )
    baseline_summary = (
        pd.read_csv(baseline_root / "baseline_summary.csv")
        if baseline_root is not None
        and (baseline_root / "baseline_summary.csv").exists()
        else pd.DataFrame()
    )
    baseline_paired = (
        pd.read_csv(baseline_root / "baseline_paired_tree_differences.csv")
        if baseline_root is not None
        and (baseline_root / "baseline_paired_tree_differences.csv").exists()
        else pd.DataFrame()
    )

    state = (
        crops.groupby("state", dropna=False)
        .agg(
            crops=("crop_id", "size"),
            trees=("tree_id", "nunique"),
            search_adequate=("search_adequate", "mean"),
            production_usable=("production_usable", "mean"),
            median_direction_error_deg=("direction_error_deg", "median"),
        )
        .reset_index()
    )
    failures = (
        crops[crops["reason_codes"].fillna("") != ""]
        .groupby("reason_codes")
        .size()
        .sort_values(ascending=False)
        .head(20)
        .rename("crops")
        .reset_index()
    )
    lines = [
        "# RAC-Pith v2 运行结果报告",
        "",
        "> 本报告由已保存产物生成；绘图和汇总阶段不重新拟合模型。所有 crop 统计均为描述性结果，主结论以 biological tree 为独立单位。",
        "",
        "## 1. 完整分母",
        "",
        f"- Manifest crops: `{denominator['manifest_crops']}`",
        f"- Trees: `{denominator['trees']}`",
        f"- Sections: `{denominator['sections']}`",
        f"- Missing results: `{denominator['missing_results']}`",
        f"- Crops requiring S3 search re-audit after structured replay: `{denominator.get('uncertainty_reaudit_required', 0)}`",
        f"- Evaluation-adjudicated production-usable POINT: `{denominator.get('production_usable', 0)}`",
        f"- Core-only production-usable flag: `{denominator.get('core_production_usable', 0)}`",
        f"- Target-domain counts: `{json.dumps(denominator.get('target_domain_counts', {}), ensure_ascii=False)}`",
        f"- Exact profiled-objective support scoring: `{json.dumps(denominator.get('support_scoring', {}), ensure_ascii=False)}`",
        f"- State counts: `{json.dumps(denominator['state_counts'], ensure_ascii=False)}`",
        "",
        "## 2. 核心定位结果（tree-balanced）",
        "",
        _markdown_table(dataset),
        "",
        "### Tree-cluster bootstrap intervals",
        "",
        _markdown_table(intervals),
        "",
        "## 3. 公平几何基线",
        "",
        _markdown_table(baseline_summary),
        "",
        "B5 相对 B0–B4 的树级配对差异（统一定义为 B5 − comparator；须结合 higher_is_better 解读）：",
        "",
        _markdown_table(baseline_paired),
        "",
        "## 4. 几何状态",
        "",
        _markdown_table(state),
        "",
        "### 状态专用评价（tree-balanced）",
        "",
        _markdown_table(state_specific),
        "",
        f"![State by distance]({(figures / 'cohort' / 'state_by_distance.png').resolve()})",
        "",
        f"![Condition, distance and state]({(figures / 'cohort' / 'condition_distance_state.png').resolve()})",
        "",
        f"![Support containment]({(figures / 'cohort' / 'support_containment_by_state.png').resolve()})",
        "",
        "## 5. POINT 误差与覆盖",
        "",
        f"![POINT error ECDF]({(figures / 'cohort' / 'point_error_ecdf.png').resolve()})",
        "",
        f"![Risk coverage]({(figures / 'cohort' / 'risk_coverage.png').resolve()})",
        "",
        "POINT 误差必须与 POINT coverage、usable coverage 和 False-POINT 风险联合解释；低误差若来自大规模 REJECT 不能视为改进。",
        "",
        "### 按髓心—裁窗距离分层",
        "",
        _markdown_table(strata),
        "",
        "### 生物学髓心与完整截面径向中心的适用域",
        "",
        _markdown_table(target_domain),
        "",
        "## 6. 弧段贡献",
        "",
        "### Crop/index 分母（含失败、跳过与缺失前置条件）",
        "",
        _markdown_table(contribution_crop_denominators),
        "",
        "### 已注册 deletion-group 分母",
        "",
        _markdown_table(contribution_denominators),
        "",
        _markdown_table(contribution_association),
        "",
        "### 子弧尺度/相位稳定性（独立分区审计）",
        "",
        _markdown_table(stability_summary),
        "",
        *(
            [
                f"![Contribution signs]({(contribution_figures / 'cohort' / 'formal_sign_distribution.png').resolve()})",
                "",
                f"![Cross-fitted versus exact contribution]({(contribution_figures / 'cohort' / 'crossfit_vs_exact.png').resolve()})",
                "",
            ]
            if contribution_figures is not None
            else []
        ),
        "正式正/负/中性标签只统计 full 与 minus 均为 POINT、且配对扰动区间跨过冻结死区的组；点估计符号仅是诊断，不能冒充区间稳定标签。",
        "",
        "## 7. 失败与限制原因",
        "",
        _markdown_table(failures),
        "",
        "## 8. 树级明细",
        "",
        _markdown_table(trees),
        "",
        "## 9. 工程耗时与搜索触发",
        "",
        _markdown_table(runtime),
        "",
        _markdown_table(search_triggers),
        "",
        "耗时是各阶段记录的单 worker elapsed time；并行运行时其总和不是端到端墙钟时间，RESUMED 行不进入延迟分位数。",
        "",
        "## 10. 结论填写门",
        "",
        "- 先确认数据、坐标、lineage、预算、搜索和产物审计均通过。",
        "- 分别陈述 POINT 精度，以及 RANGE/RAY/AXIS/MULTIMODAL 的方向、范围或模态质量。",
        "- 同时报告 aligned-domain 与 full-domain；不得删除 ECCENTRIC_GT 或失败样本。",
        "- 若支持区阈值未在独立树上校准，只称稳定域，不声称名义覆盖率。",
        "- 贡献正负只在 full/minus 均为 POINT 且区间越过死区时发布；其余以状态和功能角色为先。",
        "- 重叠 crop 不增加独立树数；测试集揭封后不得再调参数。",
        "",
        f"_Figure root used by this report: `{figures.resolve()}`_",
    ]
    return "\n".join(lines) + "\n"
