from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .provenance import atomic_write_json, sha256_file


CLASS_ORDER = np.asarray(["HARMFUL_GT", "NEUTRAL", "BENEFICIAL_GT"], dtype=object)
DEFAULT_FEATURES = (
    "contrib_cf",
    "conflict_cost",
    "shift_norm",
    "directional_info_gain",
    "direction_width_log_ratio",
    "range_width_log_ratio",
    "image_quality_mean",
    "removed_mass",
)


@dataclass
class ContributionCalibrator:
    feature_names: tuple[str, ...]
    base: Pipeline
    probability_calibrator: LogisticRegression
    minimum_probability: float

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        threshold = float(self.minimum_probability)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("calibrator minimum_probability must be finite and lie in [0, 1]")
        missing = set(self.feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"missing contribution features: {sorted(missing)}")
        base_probability = _ordered_probability(self.base, frame[list(self.feature_names)])
        calibrated = self.probability_calibrator.predict_proba(np.log(np.clip(base_probability, 1e-12, 1.0)))
        calibrated = _reorder_columns(
            calibrated, self.probability_calibrator.classes_, np.arange(len(CLASS_ORDER))
        )
        index = calibrated.argmax(axis=1)
        confidence = calibrated.max(axis=1)
        labels = CLASS_ORDER[index].astype(object)
        labels[confidence < threshold] = "ABSTAIN"
        return pd.DataFrame(
            {
                "predicted_label": labels,
                "p_harmful": calibrated[:, 0],
                "p_neutral": calibrated[:, 1],
                "p_beneficial": calibrated[:, 2],
                "calibrator_confidence": confidence,
            },
            index=frame.index,
        )


def _make_base(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    multi_class="multinomial",
                ),
            ),
        ]
    )


def _reorder_columns(
    probability: np.ndarray, observed_classes: Sequence[Any], desired_classes: Sequence[Any]
) -> np.ndarray:
    result = np.zeros((len(probability), len(desired_classes)), dtype=np.float64)
    lookup = {value: index for index, value in enumerate(observed_classes)}
    for target_index, value in enumerate(desired_classes):
        if value in lookup:
            result[:, target_index] = probability[:, lookup[value]]
    row_sum = result.sum(axis=1)
    if np.any(row_sum <= 0):
        raise ValueError("classifier omitted every registered class for at least one row")
    return result / row_sum[:, None]


def _ordered_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(features)
    classes = model.named_steps["classifier"].classes_
    desired = np.arange(len(CLASS_ORDER))
    return _reorder_columns(probability, classes, desired)


def fit_group_crossfitted_calibrator(
    frame: pd.DataFrame,
    feature_names: Iterable[str] = DEFAULT_FEATURES,
    folds: int = 6,
    seed: int = 20260906,
    minimum_probability: float = 0.60,
    fold_column: str | None = None,
) -> tuple[ContributionCalibrator, pd.DataFrame]:
    minimum_probability = float(minimum_probability)
    if not np.isfinite(minimum_probability) or not 0.0 <= minimum_probability <= 1.0:
        raise ValueError("minimum_probability must be finite and lie in [0, 1]")
    features = tuple(feature_names)
    required = {*features, "tree_id", "gt_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"calibration data lacks columns: {sorted(missing)}")
    eligible = frame[frame["gt_label"].isin(CLASS_ORDER)].copy()
    class_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    y = eligible["gt_label"].map(class_to_index).to_numpy(dtype=int)
    groups = eligible["tree_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds:
        raise ValueError(f"need at least {folds} independent trees, got {len(unique_groups)}")
    if len(np.unique(y)) != len(CLASS_ORDER):
        raise ValueError("all three GT contribution classes are required")

    if folds < 2:
        raise ValueError("cross-fitting requires at least two folds")
    split_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    if fold_column is not None:
        if fold_column not in eligible.columns or eligible[fold_column].isna().any():
            raise ValueError(
                f"registered fold column {fold_column!r} is missing or incomplete"
            )
        fold_by_tree = eligible.groupby("tree_id")[fold_column].nunique(dropna=False)
        if (fold_by_tree != 1).any():
            raise ValueError("one biological tree appears in more than one registered inner fold")
        registered = eligible[fold_column].astype(str).to_numpy()
        fold_names = sorted(np.unique(registered).tolist())
        if len(fold_names) != folds:
            raise ValueError(
                f"expected {folds} registered folds, found {len(fold_names)}: {fold_names}"
            )
        for fold_name in fold_names:
            validation_index = np.flatnonzero(registered == fold_name)
            train_index = np.flatnonzero(registered != fold_name)
            split_rows.append((fold_name, train_index, validation_index))
        fold_source = "pre_registered_tree_fold"
    else:
        splitter = GroupKFold(n_splits=folds)
        split_rows = [
            (f"groupkfold_{fold:02d}", train_index, validation_index)
            for fold, (train_index, validation_index) in enumerate(
                splitter.split(eligible[list(features)], y, groups)
            )
        ]
        fold_source = "derived_groupkfold"
    oof = np.full((len(eligible), len(CLASS_ORDER)), np.nan, dtype=np.float64)
    oof_fold = np.full(len(eligible), "", dtype=object)
    for fold, (fold_name, train_index, validation_index) in enumerate(split_rows):
        if len(np.unique(groups[validation_index])) == 0:
            raise ValueError(f"registered validation fold {fold_name!r} has no tree")
        if set(groups[train_index]) & set(groups[validation_index]):
            raise ValueError(f"tree leakage in cross-fit fold {fold_name!r}")
        if len(np.unique(y[train_index])) != len(CLASS_ORDER):
            raise ValueError(
                f"training partition {fold_name!r} does not contain all "
                f"{len(CLASS_ORDER)} registered classes"
            )
        model = _make_base(seed + fold)
        model.fit(eligible.iloc[train_index][list(features)], y[train_index])
        oof[validation_index] = _ordered_probability(
            model, eligible.iloc[validation_index][list(features)]
        )
        oof_fold[validation_index] = fold_name
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("cross-fitting failed to populate all rows")
    log_probability = np.log(np.clip(oof, 1e-12, 1.0))
    probability_calibrator = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        multi_class="multinomial",
    )
    probability_calibrator.fit(log_probability, y)
    base = _make_base(seed)
    base.fit(eligible[list(features)], y)
    calibrator = ContributionCalibrator(
        feature_names=features,
        base=base,
        probability_calibrator=probability_calibrator,
        minimum_probability=float(minimum_probability),
    )
    oof_frame = eligible[["tree_id", "crop_id", "group_id", "gt_label"]].copy()
    oof_frame[["p_harmful", "p_neutral", "p_beneficial"]] = oof
    oof_frame["predicted_label"] = CLASS_ORDER[oof.argmax(axis=1)]
    oof_frame["prediction_source"] = "group_cross_fitted"
    oof_frame["fold"] = oof_fold
    oof_frame["fold_source"] = fold_source
    return calibrator, oof_frame


def save_calibrator(
    calibrator: ContributionCalibrator,
    model_path: str | Path,
    metadata_path: str | Path,
    training_tree_ids: Iterable[str],
    *,
    provenance: dict[str, Any] | None = None,
) -> None:
    threshold = float(calibrator.minimum_probability)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("calibrator minimum_probability must be finite and lie in [0, 1]")
    model_target = Path(model_path)
    model_target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, model_target)
    metadata = {
        "schema_version": "racpith.contribution_calibrator.v1",
        "feature_names": list(calibrator.feature_names),
        "class_order": CLASS_ORDER.tolist(),
        "minimum_probability": threshold,
        "training_tree_ids": sorted(set(str(value) for value in training_tree_ids)),
        "model_path": str(model_target.resolve()),
        "model_sha256": sha256_file(model_target),
        "provenance": dict(provenance or {}),
    }
    atomic_write_json(metadata_path, metadata)
