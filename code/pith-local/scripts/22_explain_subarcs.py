#!/usr/bin/env python3
"""Sub-arc explanation layer over the full 5182-crop dataset (geometric only).

Slices stage-1 fragments into 10% sub-arcs, computes per-sub-arc geometric
attributes (pointing agreement vs the stage-7 final center, span, length,
boundary adjacency, ring zone), inherits stage-6 ring-level delete-refit roles,
and emits a per-sub-arc explanation record plus a dataset-level attribution
summary (role x category contingency — "why it affects").
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from racpith.evaluation.position_metrics import GROUP_ORDER
from racpith.evaluation.subarc_explain import explain_fragment
from racpith.provenance import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    read_jsonl,
    sha256_file,
)


def canonical_sha256(payload: dict[str, Any]) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments", required=True, help="stage-1 fragments.jsonl")
    parser.add_argument("--crop-summary", required=True, help="stage-1 crop_summary.jsonl")
    parser.add_argument("--ring-roles", required=True, help="stage-6 parent_exact_contributions.jsonl")
    parser.add_argument("--predictions", required=True, help="stage-7 final_results.jsonl")
    parser.add_argument("--grouped-csv", required=True, help="per_crop_metrics.csv from script 21")
    parser.add_argument("--output", required=True, help="output root (explain/ written under it)")
    args = parser.parse_args()

    output_root = Path(args.output).expanduser().resolve()
    explain_dir = output_root / "explain"
    explain_dir.mkdir(parents=True, exist_ok=True)

    # aux tables
    crop_meta: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.crop_summary):
        crop_meta[str(row["crop_id"])] = row
    roles: dict[tuple[str, str], list[str]] = {}
    for row in read_jsonl(args.ring_roles):
        r = row.get("roles")
        if r:
            roles[(str(row["crop_id"]), str(row["ring_id"]))] = sorted(r)
    final_centers: dict[str, np.ndarray | None] = {}
    for row in read_jsonl(args.predictions):
        pt = row.get("pith_px")
        final_centers[str(row["crop_id"])] = (
            np.asarray(pt, dtype=np.float64) if pt is not None else None
        )
    group_by_crop: dict[str, str] = {}
    import pandas as pd

    df = pd.read_csv(args.grouped_csv, encoding="utf-8-sig")
    group_by_crop = dict(zip(df["crop_id"].astype(str), df["group"].astype(str)))

    # max ring label per crop (streaming pass over fragments)
    max_ring: dict[str, int] = defaultdict(int)
    for row in read_jsonl(args.fragments):
        max_ring[str(row["crop_id"])] = max(
            max_ring[str(row["crop_id"])], int(row["ring_id"])
        )

    config = {
        "schema_version": "racpith.subarc_explain_config.v1",
        "subarc_fraction": 0.1,
        "phase": 0.0,
        "boundary_margin_px": 5.0,
        "agreement_thresholds": {"supporting": 0.90, "deviating": 0.70},
        "final_center_source": "stage7 final_results pith_px (F-All)",
        "ring_role_source": "stage6 parent_exact_contributions roles",
        "boundaries": [
            "geometric diagnostics only; does not override gt_label/contrib_gt_norm",
            "HARMFUL_GT is post-hoc audit, not a deployment delete-arc rule",
            "pointing agreement uses the FINAL estimated center (consensus direction)",
        ],
    }
    config_hash = canonical_sha256(config)
    atomic_write_json(explain_dir / "explain_config.json", {**config, "config_sha256": config_hash})

    # main streaming pass
    records_path = explain_dir / "subarc_explanations.jsonl"
    n_subarcs = 0
    n_no_meta = 0
    category_counts: Counter[str] = Counter()
    role_category: Counter[tuple[str, str]] = Counter()
    category_by_group: Counter[tuple[str, str]] = Counter()
    role_by_group: Counter[tuple[str, str]] = Counter()

    def emit(records: list[dict[str, Any]]) -> None:
        nonlocal n_subarcs
        with records_path.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
        n_subarcs += len(records)

    if records_path.exists():
        records_path.unlink()
    pending: list[dict[str, Any]] = []
    current_crop: str | None = None

    def flush_crop(crop_id: str) -> None:
        nonlocal pending
        if not pending:
            return
        meta = crop_meta.get(crop_id)
        if meta is None:
            return
        center = final_centers.get(crop_id)
        group = group_by_crop.get(crop_id, "UNKNOWN")
        recs: list[dict[str, Any]] = []
        for frag in pending:
            recs.extend(
                explain_fragment(
                    frag,
                    scale_px=float(meta["scale_px"]),
                    center_px=meta["center_px"],
                    image_size=meta["image_size"],
                    final_center=center,
                    ring_role=roles.get((crop_id, str(frag["ring_id"]))),
                    max_ring=max_ring.get(crop_id, 0),
                )
            )
        emit(recs)
        for r in recs:
            category_counts[r["category"]] += 1
            role = r["ring_role"] or "NO_ROLE"
            role_category[(role, r["category"])] += 1
            category_by_group[(r["category"], group)] += 1
            role_by_group[(role, group)] += 1
        pending = []

    for row in read_jsonl(args.fragments):
        crop_id = str(row["crop_id"])
        if current_crop is None:
            current_crop = crop_id
        if crop_id != current_crop:
            flush_crop(current_crop)
            current_crop = crop_id
        if crop_id not in crop_meta:
            n_no_meta += 1
            continue
        pending.append(row)
    flush_crop(current_crop)

    # attribution summary
    summary: dict[str, Any] = {
        "schema_version": "racpith.subarc_explain_summary.v1",
        "config_sha256": config_hash,
        "n_subarc_records": n_subarcs,
        "n_fragments_no_crop_meta": n_no_meta,
        "category_counts": dict(category_counts.most_common()),
        "role_x_category": {
            f"{role} | {cat}": n for (role, cat), n in sorted(role_category.items())
        },
        "category_by_group": {
            f"{cat} | {g}": n for (cat, g), n in sorted(category_by_group.items())
        },
        "role_by_group": {
            f"{role} | {g}": n for (role, g), n in sorted(role_by_group.items())
        },
        "input_hashes": {
            "fragments": sha256_file(args.fragments),
            "crop_summary": sha256_file(args.crop_summary),
            "ring_roles": sha256_file(args.ring_roles),
            "predictions": sha256_file(args.predictions),
        },
    }
    atomic_write_json(explain_dir / "attribution_summary.json", summary)

    # markdown report section
    lines = ["# 子弧段解释层 v1", ""]
    lines.append(
        f"生成脚本:`scripts/22_explain_subarcs.py`;配置 sha256 `{config_hash}`;"
        f"共 {n_subarcs} 条子弧解释记录(10% 弧长子弧,与贡献管线 subarc:f=0.1:phase=0 一致)。"
    )
    lines.append("")
    lines.append("## 解释类别定义(规则链:边界 → 指向一致度 → 弧长)")
    lines.append(
        "| 类别 | 判定 | 影响推断 |\n|---|---|---|\n"
        "| BOUNDARY_TRUNCATED | 子弧贴 crop 边界(≤5px) | 曲率约束弱,删除后影响小 |\n"
        "| DEVIATING | mean|cos| < 0.70 | 指向冲突,删除后误差降 |\n"
        "| SUPPORTING | mean|cos| ≥ 0.90 且弧长 ≥ 30px | 支撑共识,删除后误差升 |\n"
        "| SUPPORTING_SHORT | mean|cos| ≥ 0.90 但弧长 < 30px | 支撑有限 |\n"
        "| MODERATE | 其余 | 影响取决于邻段 |\n"
        "| NO_CENTER | 最终估计无坐标(REJECT) | 无法计算 |"
    )
    lines.append("")
    lines.append("## 类别分布(全量)")
    lines.append("| 类别 | 数量 |\n|---|---|")
    for cat, n in category_counts.most_common():
        lines.append(f"| {cat} | {n} |")
    lines.append("")
    lines.append("## 环级角色 × 解释类别 列联(回答「为什么会影响」)")
    header = ["角色 \\ 类别"] + sorted(category_counts)
    rows = []
    all_roles = sorted({r for (r, _) in role_category})
    for role in all_roles:
        rows.append([role] + [str(role_category.get((role, cat), 0)) for cat in sorted(category_counts)])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines.append("")
    lines.append("## 边界声明")
    lines.append(
        "- 几何解释是诊断层,不覆盖 `gt_label`/`contrib_gt_norm`;不宣称几何=实测贡献;\n"
        "- HARMFUL_GT 是事后审计角色,不是部署删弧规则;\n"
        "- 指向一致度的参照是 stage-7 最终估计中心(共识方向),不是 GT 髓心;\n"
        "- 与子弧 delete-refit 实测的对照见 validation8 结果。"
    )
    report_path = output_root / "子弧段解释层_v1.md"
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    print(f"wrote {records_path} ({n_subarcs} records) and {report_path}")
    print("top categories:", category_counts.most_common(6))


if __name__ == "__main__":
    main()
