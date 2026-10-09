#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from racpith.config import load_config
from racpith.evaluation.metrics import load_predictions, validate_manifest_crop_content
from racpith.provenance import (
    atomic_write_json,
    read_json_object,
    read_jsonl,
    sha256_file,
)
from racpith.visualization import render_cohort_figures, render_sample_card


def _normalise_boolean(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"true", "1", "yes"}
    ).fillna(False)


def _sample_order(metrics: pd.DataFrame, available: set[str]) -> list[str]:
    """Prioritise risky cases, then round-robin across state/distance strata."""

    frame = metrics[metrics["crop_id"].astype(str).isin(available)].copy()
    frame["crop_id"] = frame["crop_id"].astype(str)
    for column in ("catastrophic_point", "false_point", "uncertainty_reaudit_required"):
        frame[column] = _normalise_boolean(frame[column])
    frame["_priority"] = (
        8 * frame["catastrophic_point"].astype(int)
        + 4 * frame["false_point"].astype(int)
        + 2 * frame["uncertainty_reaudit_required"].astype(int)
        + (~_normalise_boolean(frame["search_adequate"])).astype(int)
    )
    frame["_error"] = pd.to_numeric(frame.get("point_error_norm"), errors="coerce").fillna(-1.0)
    ordered: list[str] = []
    priority = frame[frame["_priority"] > 0].sort_values(
        ["_priority", "_error", "crop_id"], ascending=[False, False, True]
    )
    ordered.extend(priority["crop_id"].tolist())
    buckets = {
        (str(state), str(stratum)): group.sort_values("crop_id")["crop_id"].tolist()
        for (state, stratum), group in frame.groupby(
            ["state", "distance_stratum"], dropna=False, sort=True
        )
    }
    cursor = {key: 0 for key in buckets}
    while True:
        added = False
        for key in sorted(buckets):
            index = cursor[key]
            if index < len(buckets[key]):
                crop_id = buckets[key][index]
                if crop_id not in ordered:
                    ordered.append(crop_id)
                cursor[key] += 1
                added = True
        if not added:
            break
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Render saved RAC-Pith results without refitting")
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result-index", required=True)
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--uncertainty-index")
    parser.add_argument("--target-domain-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sample-cards", type=int, default=80)
    args = parser.parse_args()
    if args.max_sample_cards < 0:
        raise ValueError("--max-sample-cards must be non-negative")
    frozen = load_config(args.config)
    output = Path(args.output)
    metrics = pd.read_csv(args.metrics)
    required_metrics = {
        "crop_id",
        "state",
        "distance_stratum",
        "search_adequate",
        "point_error_norm",
        "false_point",
        "catastrophic_point",
        "uncertainty_reaudit_required",
        "support_contains_gt",
    }
    missing_metrics = sorted(required_metrics - set(metrics.columns))
    if missing_metrics:
        raise ValueError(f"crop metrics lack required columns: {missing_metrics}")
    if metrics["crop_id"].astype(str).duplicated().any():
        raise ValueError("crop metrics contain duplicate crop_id rows")
    for column in (
        "search_adequate",
        "false_point",
        "catastrophic_point",
        "uncertainty_reaudit_required",
    ):
        if column in metrics:
            metrics[column] = _normalise_boolean(metrics[column])
    if "support_contains_gt" in metrics:
        metrics["support_contains_gt"] = metrics["support_contains_gt"].map(
            lambda value: float("nan")
            if pd.isna(value)
            else (
                value
                if isinstance(value, bool)
                else str(value).strip().lower() in {"true", "1", "yes"}
            )
        )
    denominator_path = Path(args.metrics).resolve().parent / "denominators.json"
    if not denominator_path.is_file():
        raise FileNotFoundError("crop metrics denominators.json is required for formal visualization")
    denominator = read_json_object(denominator_path)
    if denominator.get("schema_version") != "racpith.denominator.v1":
        raise ValueError("unsupported crop-metrics denominator schema")
    if denominator.get("config_hash") != frozen.sha256:
        raise ValueError("crop metrics denominator uses another configuration")
    registered_inputs = denominator.get("input_artifact_hashes", {})
    result_index_sha256 = sha256_file(args.result_index)
    expected_inputs = {
        "crop_manifest": sha256_file(args.crop_manifest),
        "evidence_index": sha256_file(args.evidence_index),
        "result_index": result_index_sha256,
        "uncertainty_index": (
            sha256_file(args.uncertainty_index) if args.uncertainty_index else None
        ),
        "target_domain_index": sha256_file(args.target_domain_index),
    }
    for name, expected_hash in expected_inputs.items():
        if registered_inputs.get(name) != expected_hash:
            raise ValueError(
                f"crop metrics were produced from another {name.replace('_', ' ')}"
            )
    registered_outputs = denominator.get("output_artifact_hashes", {})
    if registered_outputs.get("crop_metrics") != sha256_file(args.metrics):
        raise ValueError("crop_metrics.csv content changed after denominator registration")
    risk_path = Path(args.metrics).resolve().parent / "risk_coverage.csv"
    if not risk_path.is_file():
        raise FileNotFoundError("risk_coverage.csv is required for formal visualization")
    risk = pd.read_csv(risk_path)
    if registered_outputs.get("risk_coverage") != sha256_file(risk_path):
        raise ValueError("risk_coverage.csv content changed after denominator registration")
    manifest: dict[str, dict[str, object]] = {}
    for row in read_jsonl(args.crop_manifest):
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported crop-manifest schema")
        crop_id = str(row["crop_id"])
        if crop_id in manifest:
            raise ValueError(f"duplicate crop-manifest row for {crop_id}")
        manifest[crop_id] = row
    predictions, result_index = load_predictions(
        args.predictions,
        result_index_path=args.result_index,
        expected_config_hash=frozen.sha256,
    )
    if not set(metrics["crop_id"].astype(str)).issubset(manifest):
        raise ValueError("crop metrics contain crops absent from the manifest")
    for crop_id in metrics["crop_id"].astype(str):
        validate_manifest_crop_content(manifest[crop_id])
    cohort_paths = render_cohort_figures(metrics, output / "cohort", risk)
    uncertainty: dict[str, str] = {}
    if args.uncertainty_index:
        uncertainty_root = Path(args.uncertainty_index).expanduser().resolve().parent
        registered_uncertainty_paths: set[Path] = set()
        seen_uncertainty: set[str] = set()
        for item in read_jsonl(args.uncertainty_index):
            if item.get("schema_version") != "racpith.uncertainty_index.v1":
                raise ValueError("unsupported uncertainty-index schema")
            crop_id = str(item["crop_id"])
            if crop_id in seen_uncertainty:
                raise ValueError(f"duplicate uncertainty-index crop {crop_id}")
            seen_uncertainty.add(crop_id)
            if item.get("config_hash") != frozen.sha256:
                raise ValueError(f"uncertainty/config hash mismatch for {crop_id}")
            if item.get("source_result_index_sha256") != result_index_sha256:
                raise ValueError(
                    f"uncertainty/localization result-index mismatch for {crop_id}"
                )
            source = manifest.get(crop_id)
            if source is None:
                raise ValueError(f"orphan uncertainty-index crop {crop_id}")
            if any(
                str(item.get(field)) != str(source.get(field))
                for field in ("tree_id", "section_id", "split")
            ):
                raise ValueError(f"uncertainty/manifest lineage mismatch for {crop_id}")
            if item.get("status") not in {"FINISHED", "RESUMED"}:
                continue
            value = item.get("uncertainty_path")
            if value is None:
                raise ValueError(f"finished uncertainty row lacks sidecar for {crop_id}")
            sidecar = Path(str(value)).expanduser().resolve()
            if not sidecar.is_relative_to(uncertainty_root):
                raise ValueError(f"uncertainty sidecar escapes run root for {crop_id}")
            if sha256_file(sidecar) != item.get("uncertainty_sha256"):
                raise ValueError(f"uncertainty sidecar content changed for {crop_id}")
            sidecar_payload = read_json_object(sidecar)
            if (
                sidecar_payload.get("schema_version") != "racpith.uncertainty.v1"
                or str(sidecar_payload.get("crop_id")) != crop_id
                or sidecar_payload.get("config_hash") != frozen.sha256
                or str(sidecar_payload.get("adjudicated_state"))
                != str(item.get("adjudicated_state"))
                or sidecar_payload.get("source_result_index_sha256")
                != item.get("source_result_index_sha256")
            ):
                raise ValueError(f"uncertainty index/sidecar mismatch for {crop_id}")
            prediction_row = result_index.get(crop_id)
            if (
                prediction_row is None
                or item.get("baseline_result_sha256")
                != prediction_row.get("prediction_sha256")
            ):
                raise ValueError(f"uncertainty/baseline lineage mismatch for {crop_id}")
            uncertainty[crop_id] = str(sidecar)
            registered_uncertainty_paths.add(sidecar)
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
    rendered = 0
    sample_paths: dict[str, Path] = {}
    for crop_id in _sample_order(metrics, set(predictions))[: args.max_sample_cards]:
        result = predictions[crop_id]
        result_path = result["_path"]
        row = manifest.get(crop_id)
        if row is None:
            raise ValueError(f"registered prediction is absent from manifest: {crop_id}")
        destination = output / "samples" / f"{crop_id}.png"
        render_sample_card(
            row["crop_image_path"],
            row["crop_annotation_path"],
            result_path,
            destination,
            pith_gt_crop_px=row.get("pith_crop_px"),
            uncertainty_path=uncertainty.get(crop_id),
        )
        sample_paths[crop_id] = destination
        rendered += 1
    registered_pngs = {path.resolve() for path in cohort_paths} | {
        path.resolve() for path in sample_paths.values()
    }
    stale_pngs = sorted(
        str(path.resolve())
        for path in output.rglob("*.png")
        if path.resolve() not in registered_pngs
    )
    if stale_pngs:
        raise ValueError(
            f"unregistered/stale localization visualization artifacts: {stale_pngs[:10]}"
        )
    for path in registered_pngs:
        if not path.is_file():
            raise FileNotFoundError(path)
    atomic_write_json(
        output / "visualization_index.json",
        {
            "schema_version": "racpith.visualization_index.v1",
            "config_hash": frozen.sha256,
            "selection_policy": (
                "risk-priority followed by deterministic round-robin over state/distance"
            ),
            "max_sample_cards": args.max_sample_cards,
            "input_artifact_hashes": {
                "crop_manifest": sha256_file(args.crop_manifest),
                "evidence_index": sha256_file(args.evidence_index),
                "result_index": sha256_file(args.result_index),
                "crop_metrics": sha256_file(args.metrics),
                "uncertainty_index": (
                    sha256_file(args.uncertainty_index)
                    if args.uncertainty_index
                    else None
                ),
                "target_domain_index": sha256_file(args.target_domain_index),
                "denominators": sha256_file(denominator_path),
                "risk_coverage": sha256_file(risk_path),
            },
            "cohort_artifacts": {
                str(path.relative_to(output)): sha256_file(path)
                for path in cohort_paths
            },
            "sample_artifacts": {
                crop_id: {
                    "path": str(path.relative_to(output)),
                    "sha256": sha256_file(path),
                }
                for crop_id, path in sorted(sample_paths.items())
            },
        },
    )
    print(f"rendered {rendered} sample cards and cohort figures")


if __name__ == "__main__":
    main()
