#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd

from racpith.calibration import CLASS_ORDER, ContributionCalibrator
from racpith.config import load_config
from racpith.evaluation.contribution import evaluate_sign_calibrator, load_contribution_records
from racpith.provenance import read_json_object, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an already-frozen no-GT contribution calibrator to a disjoint split"
    )
    parser.add_argument("--contributions", required=True)
    parser.add_argument("--contribution-index", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frozen = load_config(args.config)
    metadata = read_json_object(args.metadata)
    if metadata.get("schema_version") != "racpith.contribution_calibrator.v1":
        raise ValueError("unsupported calibrator metadata schema")
    expected_hash = metadata.get("provenance", {}).get("config_hash")
    if expected_hash != frozen.sha256:
        raise ValueError(
            f"calibrator/config mismatch: metadata={expected_hash}, config={frozen.sha256}"
        )
    if sha256_file(args.model) != metadata.get("model_sha256"):
        raise ValueError("calibrator model content does not match its metadata hash")
    if metadata.get("class_order") != CLASS_ORDER.tolist():
        raise ValueError("calibrator class order differs from the registered class order")
    metadata_threshold = metadata.get("minimum_probability")
    if (
        isinstance(metadata_threshold, bool)
        or not isinstance(metadata_threshold, (int, float))
        or not math.isfinite(float(metadata_threshold))
        or not 0.0 <= float(metadata_threshold) <= 1.0
    ):
        raise ValueError("calibrator metadata has an invalid minimum_probability")
    calibrator = joblib.load(args.model)
    if not isinstance(calibrator, ContributionCalibrator):
        raise TypeError("model file does not contain a ContributionCalibrator")
    if tuple(calibrator.feature_names) != tuple(metadata.get("feature_names", [])):
        raise ValueError("calibrator feature registry differs from metadata")
    if not math.isclose(
        float(calibrator.minimum_probability),
        float(metadata_threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("calibrator threshold differs from metadata")
    records = load_contribution_records(
        args.contributions,
        args.contribution_index,
        expected_config_hash=frozen.sha256,
    )
    test = records[records["split"] == args.split].copy()
    if test.empty:
        raise ValueError(f"no contribution records found for split {args.split!r}")
    training_trees = set(str(value) for value in metadata.get("training_tree_ids", []))
    evaluation_trees = set(test["tree_id"].astype(str))
    overlap = sorted(training_trees & evaluation_trees)
    if overlap:
        raise ValueError(f"calibrator training/evaluation tree leakage: {overlap}")

    probabilities = calibrator.predict(test)
    for column in probabilities.columns:
        test[column] = probabilities[column]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    test.to_csv(output / f"{args.split}_contribution_predictions.csv", index=False)
    metrics, confusion = evaluate_sign_calibrator(test)
    metrics.to_csv(output / f"{args.split}_calibrator_metrics.csv", index=False)
    pd.DataFrame(
        confusion,
        index=["true_harmful", "true_neutral", "true_beneficial"],
        columns=["pred_harmful", "pred_neutral", "pred_beneficial"],
    ).to_csv(output / f"{args.split}_confusion_matrix.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
