#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from racpith.config import load_config
from racpith.evaluation.contribution import (
    LABELS,
    load_contribution_records,
    scale_phase_stability,
    summarize_contribution_denominators,
)
from racpith.evaluation.metrics import cluster_bootstrap_summary
from racpith.provenance import atomic_write_json, is_sha256, read_jsonl, sha256_file


def _conditional_label_rate(frame: pd.DataFrame, label: str) -> float:
    eligible = frame[frame["gt_label"].isin(LABELS)]
    if eligible.empty:
        return float("nan")
    per_tree = eligible.assign(_hit=eligible["gt_label"] == label).groupby("tree_id")["_hit"].mean()
    return float(per_tree.mean())


def _select_complete_index_scope(
    rows: list[dict[str, object]],
    manifest: dict[str, dict[str, object]],
    selected_crop_ids: set[str],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("schema_version") != "racpith.contribution_index.v1":
            raise ValueError("unsupported contribution-index schema")
        crop_id = str(row["crop_id"])
        if crop_id in seen:
            raise ValueError(f"duplicate contribution-index crop {crop_id}")
        seen.add(crop_id)
        source = manifest.get(crop_id)
        if source is None:
            raise ValueError(f"orphan contribution-index crop {crop_id}")
        if any(
            str(row.get(field)) != str(source.get(field))
            for field in ("tree_id", "section_id", "split")
        ):
            raise ValueError(f"contribution-index/manifest lineage mismatch for {crop_id}")
        if crop_id in selected_crop_ids:
            selected.append(row)
    missing = sorted(selected_crop_ids - {str(row["crop_id"]) for row in selected})
    if missing:
        raise ValueError(
            f"selected manifest crops lack contribution-index rows: {missing[:10]}"
        )
    return selected


def _single_required_hash(
    rows: list[dict[str, object]], field: str, *, context: str
) -> str:
    values = [row.get(field) for row in rows]
    if not values or any(not is_sha256(value) for value in values):
        raise ValueError(f"{context} must provide {field} for every selected crop")
    unique = sorted(set(values))
    if len(unique) != 1:
        raise ValueError(f"{context} must bind exactly one {field}")
    return unique[0]


def _write_evaluation_index(
    output: Path,
    *,
    status: str,
    config_hash: str,
    splits: list[str],
    input_hashes: dict[str, str | None],
    source_result_index_hashes: list[str],
    artifacts: dict[str, Path],
    manifest_crops: int,
    trees: int,
    sections: int,
    status_counts: dict[str, int],
    crop_index_records: int,
    group_records: int,
) -> None:
    index_path = output / "contribution_evaluation_index.json"
    registered = {path.resolve() for path in artifacts.values()}
    stale = sorted(
        str(path.resolve())
        for path in output.iterdir()
        if path.is_file()
        and path.resolve() != index_path.resolve()
        and path.suffix in {".csv", ".json"}
        and path.resolve() not in registered
    )
    if stale:
        raise ValueError(
            "unregistered/stale contribution-analysis artifacts in output directory: "
            f"{stale[:10]}"
        )
    for name, path in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"contribution analysis did not create {name}: {path}")
    atomic_write_json(
        index_path,
        {
            "schema_version": "racpith.contribution_evaluation_index.v1",
            "status": status,
            "config_hash": config_hash,
            "splits": splits,
            "manifest_crops": manifest_crops,
            "trees": trees,
            "sections": sections,
            "status_counts": status_counts,
            "crop_index_records": crop_index_records,
            "group_records": group_records,
            "source_result_index_sha256": source_result_index_hashes,
            "input_artifact_hashes": input_hashes,
            "output_artifacts": {
                name: {
                    "path": str(path.resolve().relative_to(output.resolve())),
                    "sha256": sha256_file(path),
                }
                for name, path in sorted(artifacts.items())
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate exact and cross-fitted ring/arc/subarc contributions"
    )
    parser.add_argument("--contributions", required=True)
    parser.add_argument("--contribution-index", required=True)
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--partition-audit-contributions")
    parser.add_argument("--partition-audit-index")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    args = parser.parse_args()
    if bool(args.partition_audit_contributions) != bool(args.partition_audit_index):
        raise ValueError(
            "--partition-audit-contributions and --partition-audit-index must be supplied together"
        )

    frozen = load_config(args.config)
    records = load_contribution_records(
        args.contributions,
        args.contribution_index,
        expected_config_hash=frozen.sha256,
    )
    manifest: dict[str, dict[str, object]] = {}
    for row in read_jsonl(args.crop_manifest):
        if row.get("schema_version") != "racpith.crop_manifest.v1":
            raise ValueError("unsupported crop-manifest schema")
        crop_id = str(row["crop_id"])
        if crop_id in manifest:
            raise ValueError(f"duplicate crop-manifest row for {crop_id}")
        manifest[crop_id] = row
    selected_crop_ids = {
        crop_id
        for crop_id, row in manifest.items()
        if not args.split or str(row["split"]) in set(args.split)
    }
    if not selected_crop_ids:
        raise ValueError("selected crop-manifest scope is empty")
    selected_index_rows = _select_complete_index_scope(
        read_jsonl(args.contribution_index), manifest, selected_crop_ids
    )
    index = pd.DataFrame(selected_index_rows)
    if not records.empty:
        records = records[
            records["crop_id"].astype(str).isin(selected_crop_ids)
        ].copy()
    if index.empty:
        raise ValueError("selected contribution-index scope is empty")
    source_result_index_hashes = [
        _single_required_hash(
            selected_index_rows,
            "source_result_index_sha256",
            context="selected contribution index",
        )
    ]
    successful_groups = sum(
        int(row.get("groups", -1))
        for row in selected_index_rows
        if row.get("status") in {"PASS", "RESUMED"}
    )
    if successful_groups < 0 or successful_groups != len(records):
        raise ValueError(
            "selected contribution records do not match the index group denominator"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    input_hashes = {
        "config_file": sha256_file(args.config),
        "crop_manifest": sha256_file(args.crop_manifest),
        "contribution_index": sha256_file(args.contribution_index),
        "partition_audit_index": (
            sha256_file(args.partition_audit_index)
            if args.partition_audit_index is not None
            else None
        ),
    }
    crop_denominators = (
        index.groupby(["split", "status"], dropna=False)
        .agg(crops=("crop_id", "size"), trees=("tree_id", "nunique"), groups=("groups", "sum"))
        .reset_index()
    )
    crop_denominator_path = output / "contribution_crop_denominators.csv"
    crop_denominators.to_csv(crop_denominator_path, index=False)
    artifacts["contribution_crop_denominators"] = crop_denominator_path
    status_counts = {
        str(status): int(count)
        for status, count in index["status"].value_counts(dropna=False).items()
    }
    manifest_trees = len({str(manifest[crop_id]["tree_id"]) for crop_id in selected_crop_ids})
    manifest_sections = len(
        {str(manifest[crop_id]["section_id"]) for crop_id in selected_crop_ids}
    )
    if records.empty:
        group_denominator_path = output / "contribution_group_denominators.csv"
        pd.DataFrame(
            columns=["level", "partition_id", "total_groups", "signed_evaluable"]
        ).to_csv(group_denominator_path, index=False)
        artifacts["contribution_group_denominators"] = group_denominator_path
        status_path = output / "contribution_evaluation_status.json"
        atomic_write_json(
            status_path,
            {
                "schema_version": "racpith.contribution_evaluation_status.v1",
                "status": "NO_GROUP_RECORDS",
                "crop_index_records": len(index),
            },
        )
        artifacts["contribution_evaluation_status"] = status_path
        _write_evaluation_index(
            output,
            status="NO_GROUP_RECORDS",
            config_hash=frozen.sha256,
            splits=sorted(set(index["split"].astype(str))),
            input_hashes=input_hashes,
            source_result_index_hashes=source_result_index_hashes,
            artifacts=artifacts,
            manifest_crops=len(selected_crop_ids),
            trees=manifest_trees,
            sections=manifest_sections,
            status_counts=status_counts,
            crop_index_records=len(index),
            group_records=0,
        )
        print("no per-group contribution records; crop-level failure denominator was preserved")
        return

    group_denominator_path = output / "contribution_group_denominators.csv"
    summarize_contribution_denominators(records).to_csv(group_denominator_path, index=False)
    artifacts["contribution_group_denominators"] = group_denominator_path

    formal = records.assign(formal_signed=records["gt_label"].isin(LABELS))
    sign_distribution = (
        formal.groupby(["level", "partition_id", "gt_label"], dropna=False)
        .agg(groups=("group_id", "size"), trees=("tree_id", "nunique"), crops=("crop_id", "nunique"))
        .reset_index()
    )
    sign_distribution["fraction_within_partition"] = sign_distribution["groups"] / sign_distribution.groupby(
        ["level", "partition_id"], dropna=False
    )["groups"].transform("sum")
    sign_distribution.to_csv(output / "formal_sign_distribution.csv", index=False)

    point_sign_distribution = (
        records.groupby(["level", "partition_id", "gt_point_label"], dropna=False)
        .agg(groups=("group_id", "size"), trees=("tree_id", "nunique"))
        .reset_index()
    )
    point_sign_distribution.to_csv(output / "point_sign_diagnostic_distribution.csv", index=False)

    tree = (
        formal.assign(
            beneficial=formal["gt_label"] == "BENEFICIAL_GT",
            harmful=formal["gt_label"] == "HARMFUL_GT",
            neutral=formal["gt_label"] == "NEUTRAL",
            uncertain=formal["gt_label"].eq("UNCERTAIN") | formal["gt_label"].isna(),
        )
        .groupby(["level", "tree_id"])
        .agg(
            groups=("group_id", "size"),
            signed_evaluable=("formal_signed", "sum"),
            beneficial_fraction=("beneficial", "mean"),
            harmful_fraction=("harmful", "mean"),
            neutral_fraction=("neutral", "mean"),
            uncertain_fraction=("uncertain", "mean"),
            median_contribution_norm=("contrib_gt_norm", "median"),
            p90_absolute_shift_norm=("shift_norm", lambda value: value.abs().quantile(0.90)),
        )
        .reset_index()
    )
    tree.to_csv(output / "contribution_tree_metrics.csv", index=False)

    roles = records[["tree_id", "crop_id", "group_id", "level", "roles"]].explode("roles")
    roles = roles[roles["roles"].notna() & (roles["roles"].astype(str) != "")]
    (
        roles.groupby(["level", "roles"])
        .agg(groups=("group_id", "size"), trees=("tree_id", "nunique"), crops=("crop_id", "nunique"))
        .reset_index()
        .sort_values(["level", "groups"], ascending=[True, False])
        .to_csv(output / "functional_role_counts.csv", index=False)
    )

    identity = records[
        records["contrib_gt_sq"].notna() & records["contrib_gt_sq_identity"].notna()
    ].copy()
    identity["absolute_identity_error"] = (
        identity["contrib_gt_sq"] - identity["contrib_gt_sq_identity"]
    ).abs()
    identity[["tree_id", "crop_id", "group_id", "absolute_identity_error"]].to_csv(
        output / "contribution_identity_audit.csv", index=False
    )

    association_rows = []
    for (level, partition_id), partition in records.groupby(
        ["level", "partition_id"], dropna=False, sort=True
    ):
        comparable = partition[
            partition["contrib_cf"].notna() & partition["contrib_gt_norm"].notna()
        ]
        association_rows.append(
            {
                "level": level,
                "partition_id": partition_id,
                "groups": len(comparable),
                "trees": comparable["tree_id"].nunique(),
                "spearman_cf_vs_exact": (
                    float(
                        comparable["contrib_cf"].corr(
                            comparable["contrib_gt_norm"], method="spearman"
                        )
                    )
                    if len(comparable) >= 3
                    else np.nan
                ),
                "cf_abstention_rate": float(partition["contrib_cf"].isna().mean()),
                "formal_gt_sign_coverage": float(partition["gt_label"].isin(LABELS).mean()),
            }
        )
    association = pd.DataFrame(association_rows)
    association.to_csv(output / "crossfit_exact_association.csv", index=False)
    primary_partition = "subarc:f={:.8g}:phase=0".format(
        float(frozen.section("contribution")["primary_subarc_fraction"])
    )
    stability_records = records
    if args.partition_audit_contributions is not None:
        stability_records = load_contribution_records(
            args.partition_audit_contributions,
            args.partition_audit_index,
            expected_config_hash=frozen.sha256,
        )
        audit_index = pd.DataFrame(
            _select_complete_index_scope(
                read_jsonl(args.partition_audit_index),
                manifest,
                selected_crop_ids,
            )
        )
        if not stability_records.empty:
            stability_records = stability_records[
                stability_records["crop_id"].astype(str).isin(selected_crop_ids)
            ].copy()
        audit_successful_groups = sum(
            int(row.get("groups", -1))
            for row in audit_index.to_dict(orient="records")
            if row.get("status") in {"PASS", "RESUMED"}
        )
        if audit_successful_groups < 0 or audit_successful_groups != len(
            stability_records
        ):
            raise ValueError(
                "partition-audit records do not match their index group denominator"
            )
        audit_source_hashes = [
            _single_required_hash(
                audit_index.to_dict(orient="records"),
                "source_result_index_sha256",
                context="selected partition-audit contribution index",
            )
        ]
        if audit_source_hashes != source_result_index_hashes:
            raise ValueError(
                "primary and partition-audit contributions bind different localization indexes"
            )
        (
            audit_index.groupby(["split", "status"], dropna=False)
            .agg(
                crops=("crop_id", "size"),
                trees=("tree_id", "nunique"),
                groups=("groups", "sum"),
            )
            .reset_index()
            .to_csv(output / "partition_audit_crop_denominators.csv", index=False)
        )
    scale_phase_stability(stability_records, primary_partition).to_csv(
        output / "scale_phase_stability.csv", index=False
    )

    records.sort_values("contrib_gt_norm", ascending=True).head(200).to_csv(
        output / "most_harmful_exact_groups.csv", index=False
    )
    records.sort_values("contrib_gt_norm", ascending=False).head(200).to_csv(
        output / "most_beneficial_exact_groups.csv", index=False
    )

    evaluation = frozen.section("evaluation")
    bootstrap_rows = []
    offset = 0
    for (level, partition_id), partition in records.groupby(
        ["level", "partition_id"], dropna=False, sort=True
    ):
        for label in ("HARMFUL_GT", "NEUTRAL", "BENEFICIAL_GT"):
            interval = cluster_bootstrap_summary(
                partition,
                lambda frame, target=label: _conditional_label_rate(frame, target),
                replicates=int(evaluation["bootstrap_replicates"]),
                confidence_level=float(evaluation["confidence_level"]),
                seed=int(frozen.data["random_seed"]) + offset,
            )
            bootstrap_rows.append(
                {
                    "level": level,
                    "partition_id": partition_id,
                    "metric": f"tree_balanced_conditional_fraction_{label.lower()}",
                    **interval,
                    "cluster_unit": "tree_id",
                }
            )
            offset += 1
    pd.DataFrame(bootstrap_rows).to_csv(
        output / "contribution_tree_cluster_bootstrap_intervals.csv", index=False
    )
    for name, filename in (
        ("formal_sign_distribution", "formal_sign_distribution.csv"),
        (
            "point_sign_diagnostic_distribution",
            "point_sign_diagnostic_distribution.csv",
        ),
        ("contribution_tree_metrics", "contribution_tree_metrics.csv"),
        ("functional_role_counts", "functional_role_counts.csv"),
        ("contribution_identity_audit", "contribution_identity_audit.csv"),
        ("crossfit_exact_association", "crossfit_exact_association.csv"),
        ("scale_phase_stability", "scale_phase_stability.csv"),
        ("most_harmful_exact_groups", "most_harmful_exact_groups.csv"),
        ("most_beneficial_exact_groups", "most_beneficial_exact_groups.csv"),
        (
            "contribution_tree_cluster_bootstrap_intervals",
            "contribution_tree_cluster_bootstrap_intervals.csv",
        ),
    ):
        artifacts[name] = output / filename
    if args.partition_audit_index is not None:
        artifacts["partition_audit_crop_denominators"] = (
            output / "partition_audit_crop_denominators.csv"
        )
    status_path = output / "contribution_evaluation_status.json"
    atomic_write_json(
        status_path,
        {
            "schema_version": "racpith.contribution_evaluation_status.v1",
            "status": "PASS",
            "crop_index_records": len(index),
            "group_records": len(records),
        },
    )
    artifacts["contribution_evaluation_status"] = status_path
    _write_evaluation_index(
        output,
        status="PASS",
        config_hash=frozen.sha256,
        splits=sorted(set(index["split"].astype(str))),
        input_hashes=input_hashes,
        source_result_index_hashes=source_result_index_hashes,
        artifacts=artifacts,
        manifest_crops=len(selected_crop_ids),
        trees=manifest_trees,
        sections=manifest_sections,
        status_counts=status_counts,
        crop_index_records=len(index),
        group_records=len(records),
    )
    print(association.to_string(index=False))


if __name__ == "__main__":
    main()
