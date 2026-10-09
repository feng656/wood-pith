from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
)

from ..provenance import read_jsonl, sha256_file


LABELS = ("HARMFUL_GT", "NEUTRAL", "BENEFICIAL_GT")
STABILITY_LABELS = (*LABELS, "UNCERTAIN")


def load_contribution_records(
    root: str | Path,
    contribution_index_path: str | Path | None = None,
    expected_config_hash: str | None = None,
) -> pd.DataFrame:
    """Load only registered per-group files when an index is supplied.

    Index-driven loading prevents stale JSONL files from an older scope or
    interrupted run from silently entering a formal analysis.
    """

    rows: list[dict[str, Any]] = []
    root_path = Path(root).expanduser().resolve()
    registered: list[tuple[Path, dict[str, Any]]] = []
    if contribution_index_path is not None:
        seen_crops: set[str] = set()
        allowed_statuses = {
            "PASS",
            "RESUMED",
            "FAIL",
            "SKIPPED_REGISTERED_SUBSAMPLE",
            "MISSING_PREREQUISITE",
        }
        for item in read_jsonl(contribution_index_path):
            if item.get("schema_version") != "racpith.contribution_index.v1":
                raise ValueError("unsupported contribution-index schema")
            crop_id = str(item["crop_id"])
            if crop_id in seen_crops:
                raise ValueError(f"duplicate contribution-index crop {crop_id}")
            seen_crops.add(crop_id)
            if (
                expected_config_hash is not None
                and item.get("config_hash") != expected_config_hash
            ):
                raise ValueError(
                    f"contribution-index/config hash mismatch for {crop_id}"
                )
            status = str(item.get("status"))
            if status not in allowed_statuses:
                raise ValueError(
                    f"unsupported contribution-index status {status!r} for {crop_id}"
                )
            if status not in {"PASS", "RESUMED"}:
                if any(
                    item.get(field) is not None
                    for field in ("record_path", "record_sha256")
                ):
                    raise ValueError(
                        f"non-PASS contribution row exposes a record file: {crop_id}"
                    )
                continue
            record_value = item.get("record_path")
            if record_value is None:
                raise ValueError(f"PASS contribution index row lacks record_path: {crop_id}")
            path = Path(str(record_value)).expanduser().resolve()
            if not path.is_relative_to(root_path):
                raise ValueError(f"registered contribution file escapes contribution root: {path}")
            if sha256_file(path) != item.get("record_sha256"):
                raise ValueError(f"registered contribution file hash changed: {crop_id}")
            registered.append((path, item))
        registered_paths = {path for path, _ in registered}
        per_crop = root_path / "per_crop"
        on_disk = (
            {path.resolve() for path in per_crop.rglob("*.jsonl")}
            if per_crop.is_dir()
            else set()
        )
        unregistered = sorted(str(path) for path in on_disk - registered_paths)
        if unregistered:
            raise ValueError(
                f"unregistered/stale contribution artifacts: {unregistered[:10]}"
            )
    else:
        registered = [(path.resolve(), {}) for path in sorted(root_path.rglob("*.jsonl"))]

    seen_groups: set[str] = set()
    for path, index_row in registered:
        file_rows = []
        for source_row in read_jsonl(path):
            if source_row.get("schema_version") != "racpith.contribution.v1":
                raise ValueError(f"unsupported per-group contribution record in {path}")
            row = dict(source_row)
            if index_row:
                for field in ("crop_id", "tree_id", "section_id", "split"):
                    if str(row.get(field)) != str(index_row.get(field)):
                        raise ValueError(
                            f"contribution record/index {field} mismatch in {path}"
                        )
                for field in (
                    "baseline_result_sha256",
                    "source_result_index_sha256",
                    "evidence_metadata_sha256",
                    "evidence_npz_sha256",
                ):
                    if row.get(field) != index_row.get(field):
                        raise ValueError(
                            f"contribution record/index {field} mismatch in {path}"
                        )
            if (
                expected_config_hash is not None
                and row.get("frozen_config_hash") != expected_config_hash
            ):
                raise ValueError(
                    f"contribution record/config hash mismatch in {path}"
                )
            group_id = str(row["group_id"])
            if group_id in seen_groups:
                raise ValueError(f"duplicate registered contribution group_id {group_id}")
            seen_groups.add(group_id)
            row["record_path"] = str(path.resolve())
            rows.append(row)
            file_rows.append(row)
        if index_row and len(file_rows) != int(index_row.get("groups", -1)):
            raise ValueError(
                f"contribution group count disagrees with index for {index_row.get('crop_id')}"
            )
    return pd.DataFrame(rows)


def summarize_contribution_denominators(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    frame = records.copy()
    frame["signed_evaluable"] = frame["gt_label"].isin(LABELS)
    frame["state_only"] = frame["gt_label"].isna() & frame["roles"].apply(
        lambda value: any(
            role in set(value if isinstance(value, (list, tuple, set)) else [])
            for role in (
                "IDENTIFIABILITY_CRITICAL",
                "DIRECTION_CRITICAL",
                "RANGE_CRITICAL",
                "MODE_EXCLUSION",
            )
        )
    )
    summary = (
        frame.groupby(["level", "partition_id"], dropna=False)
        .agg(
            total_groups=("group_id", "size"),
            signed_evaluable=("signed_evaluable", "sum"),
            state_only=("state_only", "sum"),
            beneficial=("gt_label", lambda x: int((x == "BENEFICIAL_GT").sum())),
            neutral=("gt_label", lambda x: int((x == "NEUTRAL").sum())),
            harmful=("gt_label", lambda x: int((x == "HARMFUL_GT").sum())),
            uncertain=("gt_label", lambda x: int((x == "UNCERTAIN").sum())),
            abstained=("abstain_reason", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )
    return summary


def expected_calibration_error(
    true_index: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == true_index
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def evaluate_sign_calibrator(records: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"gt_label", "predicted_label", "p_harmful", "p_neutral", "p_beneficial"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"contribution records lack calibrator columns: {sorted(missing)}")
    eligible = records[records["gt_label"].isin(LABELS) & records["predicted_label"].isin(LABELS)].copy()
    if eligible.empty:
        return pd.DataFrame(), np.zeros((3, 3), dtype=int)
    mapping = {label: index for index, label in enumerate(LABELS)}
    y_true = eligible["gt_label"].map(mapping).to_numpy(dtype=int)
    y_pred = eligible["predicted_label"].map(mapping).to_numpy(dtype=int)
    probabilities = eligible[["p_harmful", "p_neutral", "p_beneficial"]].to_numpy(dtype=float)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("contribution probabilities do not sum to one")
    brier = np.mean(
        [brier_score_loss((y_true == index).astype(int), probabilities[:, index]) for index in range(3)]
    )
    harmful = (y_true == mapping["HARMFUL_GT"]).astype(int)
    harmful_ap = (
        average_precision_score(harmful, probabilities[:, mapping["HARMFUL_GT"]])
        if 0 < harmful.sum() < len(harmful)
        else np.nan
    )
    metrics = pd.DataFrame(
        [
            {
                "n": len(eligible),
                "n_trees": eligible["tree_id"].nunique(),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "negative_pr_auc": harmful_ap,
                "multiclass_brier": brier,
                "ece": expected_calibration_error(y_true, probabilities),
                "abstention_rate": 1.0 - len(eligible) / max(len(records), 1),
            }
        ]
    )
    return metrics, confusion_matrix(y_true, y_pred, labels=np.arange(3))


def scale_phase_stability(
    records: pd.DataFrame,
    primary_partition_id: str | None = None,
) -> pd.DataFrame:
    """Compare labels at the same arc location across registered partitions.

    Different subarcs on one arc are not replicates.  Each primary subarc is
    therefore matched to the interval containing its midpoint in every other
    scale/phase partition before agreement is calculated.
    """

    output_columns = [
        "tree_id",
        "crop_id",
        "arc_id",
        "primary_group_id",
        "primary_partition_id",
        "midpoint_fraction",
        "partitions_available",
        "formal_partitions",
        "signed_partitions",
        "uncertain_partitions",
        "label_agreement",
        "beneficial_harmful_sign_flip",
        "stability_status",
        "scale_unstable",
    ]
    subarc = records[records["level"] == "subarc"].copy()
    if subarc.empty:
        return pd.DataFrame(columns=output_columns)
    if primary_partition_id is None:
        candidates = subarc[
            np.isclose(subarc["phase"].astype(float), 0.0, rtol=0.0, atol=1e-12)
        ]
        if candidates.empty:
            primary_partition_id = str(sorted(subarc["partition_id"].astype(str).unique())[0])
        else:
            fractions = candidates["scale_fraction"].astype(float)
            target = float(np.median(fractions))
            primary_partition_id = str(
                candidates.iloc[int(np.argmin(np.abs(fractions.to_numpy() - target)))][
                    "partition_id"
                ]
            )
    rows: list[dict[str, Any]] = []
    for (tree_id, crop_id, arc_id), frame in subarc.groupby(
        ["tree_id", "crop_id", "arc_id"], sort=True
    ):
        anchors = frame[frame["partition_id"].astype(str) == primary_partition_id]
        for _, anchor in anchors.iterrows():
            interval = anchor.get("interval_fraction")
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                continue
            midpoint = 0.5 * (float(interval[0]) + float(interval[1]))
            matched: list[pd.Series] = []
            for _, partition in frame.groupby("partition_id", sort=True):
                candidates: list[tuple[float, pd.Series]] = []
                for _, candidate in partition.iterrows():
                    candidate_interval = candidate.get("interval_fraction")
                    if not isinstance(candidate_interval, (list, tuple)) or len(candidate_interval) != 2:
                        continue
                    lo, hi = float(candidate_interval[0]), float(candidate_interval[1])
                    if lo - 1e-12 <= midpoint <= hi + 1e-12:
                        overlap = max(
                            0.0,
                            min(float(interval[1]), hi) - max(float(interval[0]), lo),
                        )
                        candidates.append((overlap, candidate))
                if candidates:
                    matched.append(max(candidates, key=lambda value: value[0])[1])
            formal_labels = [
                str(value.get("gt_label"))
                for value in matched
                if value.get("gt_label") in STABILITY_LABELS
            ]
            counts = pd.Series(formal_labels, dtype=object).value_counts()
            agreement = float(counts.max() / counts.sum()) if len(counts) else float("nan")
            sign_flip = "HARMFUL_GT" in formal_labels and "BENEFICIAL_GT" in formal_labels
            enough = len(formal_labels) >= 2
            rows.append(
                {
                    "tree_id": tree_id,
                    "crop_id": crop_id,
                    "arc_id": arc_id,
                    "primary_group_id": anchor["group_id"],
                    "primary_partition_id": primary_partition_id,
                    "midpoint_fraction": midpoint,
                    "partitions_available": len(matched),
                    "formal_partitions": len(formal_labels),
                    "signed_partitions": sum(value in LABELS for value in formal_labels),
                    "uncertain_partitions": formal_labels.count("UNCERTAIN"),
                    "label_agreement": agreement,
                    "beneficial_harmful_sign_flip": sign_flip,
                    "stability_status": "ASSESSED" if enough else "INSUFFICIENT_PARTITIONS",
                    "scale_unstable": bool(enough and (agreement < 0.75 or sign_flip)),
                }
            )
    return pd.DataFrame(rows, columns=output_columns)
