#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from racpith.calibration import (
    DEFAULT_FEATURES,
    fit_group_crossfitted_calibrator,
    save_calibrator,
)
from racpith.config import load_config
from racpith.evaluation.contribution import (
    evaluate_sign_calibrator,
    load_contribution_records,
    scale_phase_stability,
    summarize_contribution_denominators,
)
from racpith.provenance import is_sha256, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the small tree-group-cross-fitted no-GT contribution calibrator"
    )
    parser.add_argument("--contributions", required=True)
    parser.add_argument("--contribution-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--test-split",
        help="optional one-time evaluation split; omit while fitting/finalising the frozen model",
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--minimum-probability", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2 for tree-group cross-fitting")
    if not math.isfinite(args.minimum_probability) or not 0.0 <= args.minimum_probability <= 1.0:
        raise ValueError("--minimum-probability must be finite and lie in [0, 1]")
    frozen = load_config(args.config)
    records = load_contribution_records(
        args.contributions,
        args.contribution_index,
        expected_config_hash=frozen.sha256,
    )
    if records.empty:
        raise ValueError("no contribution records found")
    if "split" not in records.columns:
        raise ValueError("contribution records must carry their source split")
    if "source_result_index_sha256" not in records.columns:
        raise ValueError("contribution records lack localization-result lineage")
    source_result_hashes = sorted(
        set(records["source_result_index_sha256"].astype(str))
    )
    if len(source_result_hashes) != 1 or not is_sha256(source_result_hashes[0]):
        raise ValueError(
            "calibration input must bind exactly one canonical localization result index"
        )
    train = records[records["split"] == args.train_split].copy()
    if train.empty:
        raise ValueError(f"no contribution records found for --train-split={args.train_split}")
    if "inner_fold" not in train.columns or train["inner_fold"].isna().any():
        raise ValueError(
            "formal calibration requires the pre-registered tree-level inner_fold lineage"
        )
    test = (
        records[records["split"] == args.test_split].copy()
        if args.test_split is not None
        else pd.DataFrame()
    )
    if args.test_split is not None:
        overlap = set(train["tree_id"].astype(str)) & set(test["tree_id"].astype(str))
        if overlap:
            raise ValueError(f"tree leakage between train and test: {sorted(overlap)}")
    calibrator, oof = fit_group_crossfitted_calibrator(
        train,
        DEFAULT_FEATURES,
        folds=args.folds,
        seed=args.seed,
        minimum_probability=args.minimum_probability,
        fold_column="inner_fold",
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_calibrator(
        calibrator,
        output / "contribution_calibrator.joblib",
        output / "contribution_calibrator.json",
        train["tree_id"].astype(str),
        provenance={
            "config_hash": frozen.sha256,
            "config_file_sha256": sha256_file(args.config),
            "contribution_index_sha256": sha256_file(args.contribution_index),
            "source_result_index_sha256": source_result_hashes[0],
            "training_split": args.train_split,
            "training_group_records": len(train),
            "training_signed_records": int(train["gt_label"].isin(
                ("HARMFUL_GT", "NEUTRAL", "BENEFICIAL_GT")
            ).sum()),
            "folds": args.folds,
            "fold_source": "section_manifest.inner_fold",
            "seed": args.seed,
            "test_split_consumed_during_fit_command": args.test_split,
        },
    )
    oof.to_csv(output / "train_oof_probabilities.csv", index=False)
    if args.test_split is not None:
        if test.empty:
            raise ValueError(f"no contribution records found for --test-split={args.test_split}")
        predicted = test.copy()
        probability = calibrator.predict(test)
        for column in probability.columns:
            predicted[column] = probability[column]
        predicted.to_csv(output / f"{args.test_split}_contribution_predictions.csv", index=False)
        metrics, confusion = evaluate_sign_calibrator(predicted)
        metrics.to_csv(output / f"{args.test_split}_calibrator_metrics.csv", index=False)
        pd.DataFrame(
            confusion,
            index=["true_harmful", "true_neutral", "true_beneficial"],
            columns=["pred_harmful", "pred_neutral", "pred_beneficial"],
        ).to_csv(output / f"{args.test_split}_confusion_matrix.csv")
        print(metrics.to_string(index=False))
    summarize_contribution_denominators(records).to_csv(
        output / "contribution_denominators.csv", index=False
    )
    primary_partition = "subarc:f={:.8g}:phase=0".format(
        float(frozen.section("contribution")["primary_subarc_fraction"])
    )
    scale_phase_stability(records, primary_partition).to_csv(
        output / "scale_phase_stability.csv", index=False
    )
    if args.test_split is None:
        print(
            f"fitted contribution calibrator on split={args.train_split}; "
            "no sealed/evaluation split was read"
        )


if __name__ == "__main__":
    main()
