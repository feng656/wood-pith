#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def grouped_summary(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result = {}
    for value, group in sorted(groups.items()):
        result[value] = {
            "n": len(group),
            "median_m0_bias_norm": float(np.median([r["m0_bias_norm"] for r in group])),
            "median_m1_bias_norm": float(np.median([r["m1_bias_norm"] for r in group])),
            "median_bias_reduction": float(np.median([r["reduction"] for r in group])),
            "fraction_reduction_ge_20pct": float(np.mean([r["reduction"] >= 0.20 for r in group])),
            "median_m0_abs_radial_bias": float(np.median([abs(r["m0_radial"]) for r in group])),
            "median_m0_abs_tangential_bias": float(np.median([abs(r["m0_tangential"]) for r in group])),
            "median_m0_loss_gain": float(np.median([r["m0_loss_gain"] for r in group])),
            "median_m1_loss_gain": float(np.median([r["m1_loss_gain"] for r in group])),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bias-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = {row["sample_id"]: row for row in read_jsonl(Path(args.manifest))}
    m0 = {row["sample_id"]: row for row in read_jsonl(Path(args.bias_dir) / "m0_bias.jsonl")}
    m1 = {row["sample_id"]: row for row in read_jsonl(Path(args.bias_dir) / "m1rpc_bias.jsonl")}
    if set(manifest) != set(m0) or set(m0) != set(m1):
        raise ValueError("manifest, M0, and M1-RPC sample IDs do not match")
    rows = []
    for sample_id, base in m0.items():
        repair = m1[sample_id]
        source = manifest[sample_id]
        m0_bias = float(base["bias_norm"])
        m1_bias = float(repair["bias_norm"])
        rows.append({
            "sample_id": sample_id,
            "tree_id": base["tree_id"],
            "size": int(source["image_size"][0]),
            "pith_in_patch": bool(source.get("metadata", {}).get("pith_in_patch")),
            "pith_distance_norm": float(np.linalg.norm(base["gt"])),
            "m0_bias_norm": m0_bias,
            "m1_bias_norm": m1_bias,
            "reduction": 1.0 - m1_bias / (m0_bias + 1e-12),
            "m0_radial": float(base["radial_bias_norm"]),
            "m0_tangential": float(base["tangential_bias_norm"]),
            "m0_loss_gain": float(base["loss_gain"]),
            "m1_loss_gain": float(repair["loss_gain"]),
        })
    distances = np.asarray([row["pith_distance_norm"] for row in rows])
    edges = np.quantile(distances, [0.0, 0.25, 0.5, 0.75, 1.0])
    distance_groups = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        group = [row for row in rows if row["pith_distance_norm"] >= left and (row["pith_distance_norm"] < right if index < 3 else row["pith_distance_norm"] <= right)]
        distance_groups.append({
            "range": [float(left), float(right)],
            "n": len(group),
            "median_m0_bias_norm": float(np.median([row["m0_bias_norm"] for row in group])),
            "median_m1_bias_norm": float(np.median([row["m1_bias_norm"] for row in group])),
            "median_bias_reduction": float(np.median([row["reduction"] for row in group])),
            "fraction_reduction_ge_20pct": float(np.mean([row["reduction"] >= 0.20 for row in group])),
        })
    summary = {
        "samples": len(rows),
        "by_tree": grouped_summary(rows, "tree_id"),
        "by_size": grouped_summary(rows, "size"),
        "by_pith_in_patch": grouped_summary(rows, "pith_in_patch"),
        "by_true_pith_distance_quantile": distance_groups,
        "m0_bias_quantiles": dict(zip(["min", "q25", "median", "q75", "q90", "q99", "max"], map(float, np.quantile([row["m0_bias_norm"] for row in rows], [0, .25, .5, .75, .9, .99, 1])))),
        "m1_bias_quantiles": dict(zip(["min", "q25", "median", "q75", "q90", "q99", "max"], map(float, np.quantile([row["m1_bias_norm"] for row in rows], [0, .25, .5, .75, .9, .99, 1])))),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
