#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(values):
    return [float(value) for value in values if value is not None and np.isfinite(value)]


def median(values):
    values = finite(values)
    return float(np.median(values)) if values else None


def correlation(rows, left, right):
    pairs = [(row.get(left), row.get(right)) for row in rows]
    pairs = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return None
    value = spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic
    return float(value) if np.isfinite(value) else None


def grouped_sample_summary(samples, key):
    groups = defaultdict(list)
    for sample in samples:
        groups[str(sample[key])].append(sample)
    output = {}
    for value, group in sorted(groups.items()):
        output[value] = {
            "samples": len(group),
            "median_parent_rings": median([sample["parent_rings"] for sample in group]),
            "median_max_projective_shift_rad": median([sample["max_projective_shift_rad"] for sample in group]),
            "median_max_point_shift_norm": median([sample["max_point_shift_norm"] for sample in group]),
            "median_max_info_logdet_drop": median([sample["max_info_logdet_drop"] for sample in group]),
            "median_gt_conflict_fraction": median([sample["gt_conflict_fraction"] for sample in group]),
            "samples_with_high_leverage_conflict_fraction": float(np.mean([sample["high_leverage_conflict_count"] > 0 for sample in group])),
            "hessian_valid_fraction": float(np.mean([sample["hessian_valid"] for sample in group])),
        }
    return output


def compact(row):
    return {key: row.get(key) for key in (
        "sample_id", "tree_id", "group_id", "support_budget", "support_length_norm",
        "projective_shift_rad", "point_shift_norm", "info_logdet_drop",
        "gt_error_increase_when_deleted", "gt_conflict",
    )}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contribution", required=True)
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(Path(args.contribution))
    sentinel = {row["sample_id"]: row for row in read_jsonl(Path(args.sentinel))}
    by_sample = defaultdict(list)
    for row in rows:
        if row["sample_id"] not in sentinel:
            raise ValueError(f"contribution sample absent from sentinel: {row['sample_id']}")
        by_sample[row["sample_id"]].append(row)
    if set(by_sample) != set(sentinel):
        raise ValueError("contribution output does not cover every sentinel sample")
    sample_summaries = []
    for sample_id, group in sorted(by_sample.items()):
        source = sentinel[sample_id]
        projective = finite([row.get("projective_shift_rad") for row in group])
        leverage_q75 = float(np.quantile(projective, 0.75)) if projective else float("inf")
        high_conflicts = [row for row in group if row.get("gt_conflict") and row.get("projective_shift_rad") is not None and row["projective_shift_rad"] >= leverage_q75]
        high_significant_conflicts = [row for row in high_conflicts if row.get("gt_error_increase_when_deleted") is not None and row["gt_error_increase_when_deleted"] <= -0.005]
        info = finite([row.get("info_logdet_drop") for row in group])
        sample_summaries.append({
            "sample_id": sample_id,
            "tree_id": source["tree_id"],
            "crop_size_px": int(source["image_size"][0]),
            "pith_in_patch": bool(source.get("metadata", {}).get("pith_in_patch")),
            "parent_rings": len(group),
            "median_support_budget": median([row.get("support_budget") for row in group]),
            "max_projective_shift_rad": max(projective) if projective else None,
            "max_point_shift_norm": max(finite([row.get("point_shift_norm") for row in group]), default=None),
            "max_info_logdet_drop": max(info) if info else None,
            "gt_help_fraction": float(np.mean([(row.get("gt_error_increase_when_deleted") or 0) > 0 for row in group])),
            "gt_conflict_fraction": float(np.mean([row.get("gt_conflict") is True for row in group])),
            "high_leverage_conflict_count": len(high_conflicts),
            "high_leverage_significant_conflict_count": len(high_significant_conflicts),
            "hessian_valid": bool(all(row.get("full_hessian_min_eigenvalue", -1) > 0 and row.get("deleted_hessian_min_eigenvalue", -1) > 0 and row.get("info_logdet_drop") is not None for row in group)),
            "support_info_spearman": correlation(group, "support_budget", "info_logdet_drop"),
            "support_leverage_spearman": correlation(group, "support_budget", "projective_shift_rad"),
        })
    valid_info = [row for row in rows if row.get("info_logdet_drop") is not None]
    gt_effects = finite([row.get("gt_error_increase_when_deleted") for row in rows])
    summary = {
        "sentinel_samples": len(sample_summaries),
        "parent_ring_deletions": len(rows),
        "information_rows_with_positive_definite_hessian": len(valid_info),
        "information_rows_without_valid_logdet": len(rows) - len(valid_info),
        "gt_conflict_rows": sum(row.get("gt_conflict") is True for row in rows),
        "gt_conflict_fraction": float(np.mean([row.get("gt_conflict") is True for row in rows])),
        "gt_effect_quantiles": dict(zip(["min", "q10", "q25", "median", "q75", "q90", "max"], map(float, np.quantile(gt_effects, [0, .1, .25, .5, .75, .9, 1])))),
        "significant_gt_conflict_fraction_le_minus_0.005": float(np.mean([value <= -0.005 for value in gt_effects])),
        "strong_gt_conflict_fraction_le_minus_0.01": float(np.mean([value <= -0.01 for value in gt_effects])),
        "significant_gt_help_fraction_ge_0.005": float(np.mean([value >= 0.005 for value in gt_effects])),
        "strong_gt_help_fraction_ge_0.01": float(np.mean([value >= 0.01 for value in gt_effects])),
        "samples_with_high_leverage_conflict": sum(sample["high_leverage_conflict_count"] > 0 for sample in sample_summaries),
        "samples_with_high_leverage_significant_conflict": sum(sample["high_leverage_significant_conflict_count"] > 0 for sample in sample_summaries),
        "median_within_sample_support_info_spearman": median([sample["support_info_spearman"] for sample in sample_summaries]),
        "median_within_sample_support_leverage_spearman": median([sample["support_leverage_spearman"] for sample in sample_summaries]),
        "by_tree": grouped_sample_summary(sample_summaries, "tree_id"),
        "by_crop_size_px": grouped_sample_summary(sample_summaries, "crop_size_px"),
        "by_pith_in_patch": grouped_sample_summary(sample_summaries, "pith_in_patch"),
        "top_information_drop": [compact(row) for row in sorted(valid_info, key=lambda row: row["info_logdet_drop"], reverse=True)[:10]],
        "top_projective_leverage": [compact(row) for row in sorted(rows, key=lambda row: row.get("projective_shift_rad") or -1, reverse=True)[:10]],
        "strongest_gt_conflicts": [compact(row) for row in sorted([row for row in rows if row.get("gt_error_increase_when_deleted") is not None], key=lambda row: row["gt_error_increase_when_deleted"])[:10]],
        "per_sample": sample_summaries,
        "interpretation_contract": {
            "support": "frozen parent-ring budget and normalized visible length",
            "information": "fixed-coordinate Hessian logdet drop, reported only when both ridge-adjusted Hessians are positive definite",
            "leverage": "projective and finite-point solution movement after deleting the parent ring",
            "gt_help": "positive error increase after deletion means the ring helped the full solution",
            "conflict": "negative GT error increase after deletion means removing the ring improved GT error; significant conflict is <=-0.005 norm and high leverage uses the within-sample top quartile of projective shift",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"per_sample", "top_information_drop", "top_projective_leverage", "strongest_gt_conflicts"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
