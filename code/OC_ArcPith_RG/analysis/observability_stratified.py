#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STATES = ("POINT", "AXIS", "MULTIMODAL", "REJECT")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distance_stratum(row: dict) -> str:
    if row["pith_in_patch"]:
        return "inside"
    distance = row.get("true_pith_distance_norm")
    if distance is None:
        return "unknown"
    if distance <= 2.0:
        return "near_outside"
    if distance <= 4.0:
        return "mid_outside"
    return "far_outside"


def ring_count_stratum(count: int) -> str:
    if count <= 4:
        return "01-04"
    if count <= 8:
        return "05-08"
    if count <= 16:
        return "09-16"
    return "17_plus"


def diversity_stratum(value: float) -> str:
    if value < 0.40:
        return "low_lt_0.40"
    if value < 0.70:
        return "medium_0.40_0.70"
    return "high_ge_0.70"


def group_summary(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result = {}
    for value, group in sorted(grouped.items()):
        counts = Counter(row["state"] for row in group)
        result[value] = {
            "n": len(group),
            "state_counts": {state: counts.get(state, 0) for state in STATES},
            "state_fractions": {state: counts.get(state, 0) / len(group) for state in STATES},
            "median_phi_width_deg": float(np.median([row["phi_width_deg"] for row in group])),
            "median_visible_parent_rings": float(np.median([row["visible_parent_ring_count"] for row in group])),
            "median_tangent_axis_entropy": float(np.median([row["tangent_axis_entropy"] for row in group])),
            "gt_within_profile_support_fraction": float(np.mean([row.get("gt_within_profile_support") is True for row in group])),
            "median_gt_loss_gap": float(np.median([row["gt_loss_gap"] for row in group if row.get("gt_loss_gap") is not None])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_jsonl(Path(args.states))
    manifest = read_jsonl(Path(args.manifest))
    state_ids = [row["sample_id"] for row in rows]
    manifest_ids = [row["sample_id"] for row in manifest]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("duplicate sample IDs in observability states")
    if set(state_ids) != set(manifest_ids):
        raise ValueError("observability states do not match manifest")
    for row in rows:
        row["distance_stratum"] = distance_stratum(row)
        row["ring_count_stratum"] = ring_count_stratum(int(row["visible_parent_ring_count"]))
        row["diversity_stratum"] = diversity_stratum(float(row["tangent_axis_entropy"]))
    counts = Counter(row["state"] for row in rows)
    summary = {
        "samples": len(rows),
        "state_counts": {state: counts.get(state, 0) for state in STATES},
        "state_fractions": {state: counts.get(state, 0) / len(rows) for state in STATES},
        "gt_within_profile_support_fraction": float(np.mean([row.get("gt_within_profile_support") is True for row in rows])),
        "strata_definitions": {
            "distance": "inside uses crop bounds; outside crops are near when true distance/half-FOV <=2, mid when <=4, otherwise far",
            "visible_parent_rings": "01-04, 05-08, 09-16, 17_plus",
            "tangent_axis_entropy": "normalized 18-bin axial tangent entropy: low <0.40, medium <0.70, high >=0.70",
        },
        "by_tree": group_summary(rows, "tree_id"),
        "by_crop_size_px": group_summary(rows, "crop_size_px"),
        "by_true_pith_distance": group_summary(rows, "distance_stratum"),
        "by_visible_parent_ring_count": group_summary(rows, "ring_count_stratum"),
        "by_angular_diversity": group_summary(rows, "diversity_stratum"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
