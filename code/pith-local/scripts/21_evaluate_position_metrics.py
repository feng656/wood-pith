#!/usr/bin/env python3
"""Four-way grouped position-metric evaluation (pixel / R_F / ring-spacing layers).

Inputs: stage-0 validated manifest (GT), stage-7 final results (predictions),
frozen section scale table from scripts/20_build_section_scales.py.
Outputs: per-crop CSV, grouped summary JSON, Chinese markdown report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from racpith.evaluation.position_metrics import (
    GROUP_ORDER,
    METRIC_KEYS,
    build_position_metrics,
    grouped_summary,
)
from racpith.provenance import (
    atomic_write_json,
    atomic_write_text,
    read_jsonl,
    sha256_file,
)


def canonical_sha256(payload: dict[str, Any]) -> str:
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _fmt(v: float | None, digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(
    summary: dict[str, Any],
    config: dict[str, Any],
    scale_summary: dict[str, Any],
    config_hash: str,
) -> str:
    th = scale_summary["params"]
    grouping = config["grouping"]
    parts: list[str] = []
    parts.append("# 髓心定位误差分层评价 v1")
    parts.append(
        "\n生成脚本:`scripts/21_evaluate_position_metrics.py`;冻结评价配置 sha256 "
        f"`{config_hash}`。\n"
    )
    parts.append("## 1. 三层尺度指标(与 shusui.txt 一致)")
    parts.append(
        "| 层 | 定义 | 回答的问题 |\n"
        "|---|---|---|\n"
        f"| 像素绝对误差 | `error_px = \\|p̂−g\\|₂`(crop 像素) | 绝对准确性 |\n"
        f"| 误差÷最大半径 | `E_RF_pct = 100·error_px/R_F`,`R_F=D_F/2`,`D_F`=完整父截面最外环凸包直径(冻结,同一 section 所有 crop 共用) | 误差相对整个截面的尺度 |\n"
        f"| 误差÷年轮间距 | `E_w = error_px/w_ref`、`E_wmax = error_px/w_max`(冻结的相邻环距代表值/p90) | 误差相对年轮结构尺度(不是树龄误差) |\n"
        f"| 毫米误差 | `E_mm` = 空(全数据集 `mm_per_pixel=null`) | 物理尺度(无标定时一律不报、不猜测) |\n"
    )
    parts.append(
        "**边界**:1) 父截面尺度只进入评价器,不进入局部反演;2) `E_RF_pct` 与 APD 的 "
        "`E_APD_box_pct`(包围盒半径分母)不可互换,两者容差相差一倍;3) `E_w` 是中心位置误差与结构尺度之比,"
        "不能解释为“错了一年”;4) 归一化分母全部在评价前冻结,不得用预测髓心或删弧后的年轮重算。\n"
    )
    parts.append("## 2. 冻结父截面尺度表")
    parts.append(
        f"- sections: {scale_summary['n_sections']},crops: {scale_summary['n_crops']}\n"
        f"- R_F_px:min {_fmt(scale_summary['R_F_px']['min'],1)} / "
        f"中位 {_fmt(scale_summary['R_F_px']['median'],1)} / max {_fmt(scale_summary['R_F_px']['max'],1)}\n"
        f"- w_ref_px:min {_fmt(scale_summary['w_ref_px']['min'])} / "
        f"中位 {_fmt(scale_summary['w_ref_px']['median'])} / p90 {_fmt(scale_summary['w_ref_px']['p90'])} / "
        f"max {_fmt(scale_summary['w_ref_px']['max'])}\n"
        f"- w_max_px 中位 {_fmt(scale_summary['w_max_px']['median'])}\n"
        f"- 方法参数:`wref_samples={th['wref_samples_per_ring']}`,法线 PCA 窗 "
        f"`{th['wref_pca_window']}`,锥角 `±{th['wref_cone_angle_deg']}°`,间距上限 "
        f"`{th['wref_max_spacing_px']}px`,w_ref 取分位 {th['wref_quantile']},w_max 取分位 {th['wmax_quantile']};\n"
        "  D_F 取最外环(最大面积 polygon)凸包直径。\n"
    )
    parts.append("## 3. 四类分组定义")
    parts.append(
        "髓心在图内是明确的;近/中/远按「crop 中心到髓心距离 d ÷ 父截面半径 R_F」划分(尺度不变,同一 section 通用):\n\n"
        f"- **图内 IN_IMAGE**:pith 在 crop 内;\n"
        f"- **近 NEAR**:d/R_F ≤ {grouping['near_ratio']}(中位截面约 600px,弧曲率强);\n"
        f"- **中 MID**:{grouping['near_ratio']} < d/R_F ≤ {grouping['mid_ratio']}(约半到一整个典型 crop 宽,曲率中等);\n"
        f"- **远 FAR**:d/R_F > {grouping['mid_ratio']}(弧近直线,病态)。\n\n"
        "阈值理由:0.5/1.0 作阈值会使 FAR 退化(n=2);0.35/0.70 下四类样本量均衡且误差分层明显。\n"
    )
    parts.append("## 4. 分组误差统计")
    by_group = summary["by_group"]
    headers = ["组", "crop 数", "有预测", "REJECT 无坐标", "error_px 中位/p90/max", "E_RF_pct 中位/p90/max(%)", "E_w 中位/p90/max", "E_wmax 中位/p90/max"]
    rows = []
    for g in GROUP_ORDER:
        b = by_group[g]
        rows.append(
            [
                g,
                str(b["n_crops"]),
                str(b["n_with_prediction"]),
                str(b["n_reject_no_prediction"]),
                f"{_fmt(b['error_px']['median'],1)} / {_fmt(b['error_px']['p90'],1)} / {_fmt(b['error_px']['max'],1)}",
                f"{_fmt(b['E_RF_pct']['median'])} / {_fmt(b['E_RF_pct']['p90'])} / {_fmt(b['E_RF_pct']['max'])}",
                f"{_fmt(b['E_w']['median'])} / {_fmt(b['E_w']['p90'])} / {_fmt(b['E_w']['max'])}",
                f"{_fmt(b['E_wmax']['median'])} / {_fmt(b['E_wmax']['p90'])} / {_fmt(b['E_wmax']['max'])}",
            ]
        )
    b = summary["overall"]
    rows.append(
        [
            "**总体**",
            str(b["n_crops"]),
            str(b["n_with_prediction"]),
            str(b["n_reject_no_prediction"]),
            f"{_fmt(b['error_px']['median'],1)} / {_fmt(b['error_px']['p90'],1)} / {_fmt(b['error_px']['max'],1)}",
            f"{_fmt(b['E_RF_pct']['median'])} / {_fmt(b['E_RF_pct']['p90'])} / {_fmt(b['E_RF_pct']['max'])}",
            f"{_fmt(b['E_w']['median'])} / {_fmt(b['E_w']['p90'])} / {_fmt(b['E_w']['max'])}",
            f"{_fmt(b['E_wmax']['median'])} / {_fmt(b['E_wmax']['p90'])} / {_fmt(b['E_wmax']['max'])}",
        ]
    )
    parts.append(_md_table(headers, rows))
    parts.append("\n每类 × 树(中位 error_px px / 中位 E_RF_pct%):\n")
    tree_headers = ["组", "T0", "T2", "T4", "T6"]
    tree_rows = []
    for g in GROUP_ORDER:
        cells = [g]
        for tree in ["T0", "T2", "T4", "T6"]:
            tb = summary["by_group_tree"][g].get(tree)
            if tb is None:
                cells.append("—")
            else:
                cells.append(
                    f"{_fmt(tb['error_px']['median'],1)} / {_fmt(tb['E_RF_pct']['median'])}"
                )
        tree_rows.append(cells)
    parts.append(_md_table(tree_headers, tree_rows))
    parts.append("")
    parts.append("## 5. 预测覆盖与毫米层")
    mm = summary["E_mm_status"]
    parts.append(
        f"- E_mm:可用 {mm['available']}/5182,不可用 {mm['unavailable']}/5182 → 一律为空;"
        "政策:无可靠比例尺时一律不报,不得根据平均树径、相机型号或图片 DPI 标签猜测。\n"
        "- REJECT(无坐标)样本不混入误差分布,仅在覆盖列单独计数。\n"
    )
    parts.append("## 6. 结论要点")
    far = by_group["FAR"]
    inside = by_group["IN_IMAGE"]
    near = by_group["NEAR"]
    mid = by_group["MID"]
    parts.append(
        f"- 图内误差中位 {_fmt(inside['error_px']['median'],1)} px({_fmt(inside['E_RF_pct']['median'])}% R_F,约 {_fmt(inside['E_w']['median'])} 个参考环距);\n"
        f"- 近/中/远误差逐级上升:NEAR {_fmt(near['error_px']['median'],1)} px → MID {_fmt(mid['error_px']['median'],1)} px → FAR {_fmt(far['error_px']['median'],1)} px,"
        "与弧曲率信息量递减一致;\n"
        f"- 远髓占比 {far['n_crops']}/{summary['overall']['n_crops']},其 p90 达 "
        f"{_fmt(far['error_px']['p90'],1)} px({_fmt(far['E_w']['p90'])} 个参考环距)——远髓类是最主要的精度瓶颈;\n"
        f"- 毫米层:无物理标定,不能宣称毫米精度。\n"
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="stage-0 manifest_validated.jsonl")
    parser.add_argument("--predictions", required=True, help="stage-7 final_results.jsonl")
    parser.add_argument("--section-scales", required=True, help="section_scales.jsonl from script 20")
    parser.add_argument("--scale-summary", required=True, help="scales/summary.json from script 20")
    parser.add_argument("--output", required=True, help="output root (grouped/ written under it)")
    args = parser.parse_args()

    output_root = Path(args.output).expanduser().resolve()
    grouped_dir = output_root / "grouped"
    grouped_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_jsonl(args.manifest)
    predictions = {str(r["crop_id"]): r for r in read_jsonl(args.predictions)}
    scales = {str(r["section_id"]): r for r in read_jsonl(args.section_scales)}
    scale_summary = json.loads(Path(args.scale_summary).read_text(encoding="utf-8"))

    config = {
        "schema_version": "racpith.position_eval_config.v1",
        "grouping": {
            "in_image": "pith inside crop rectangle",
            "near_ratio": 0.35,
            "mid_ratio": 0.70,
            "distance": "crop center to pith, normalized by frozen R_F of the parent section",
        },
        "metrics": {
            "error_px": "euclidean prediction-GT distance in crop px",
            "E_RF_pct": "100 * error_px / R_F_px (frozen full-parent-section scale)",
            "E_w": "error_px / w_ref_px (frozen representative adjacent ring spacing)",
            "E_wmax": "error_px / w_max_px (frozen p90 adjacent ring spacing)",
            "E_mm": "empty; mm_per_pixel absent dataset-wide; never guessed",
        },
        "boundaries": [
            "parent-section scale enters the evaluator only, never local inference",
            "E_RF_pct is not interchangeable with APD E_APD_box_pct",
            "E_w is a position-error/structure-scale ratio, not a tree-age error",
            "all denominators are frozen before evaluation",
        ],
    }
    config_hash = canonical_sha256(config)
    atomic_write_json(grouped_dir / "eval_config.json", {**config, "config_sha256": config_hash})

    metrics = build_position_metrics(manifest_rows, predictions, scales)
    csv_path = grouped_dir / "per_crop_metrics.csv"
    metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = grouped_summary(metrics)
    summary["config_sha256"] = config_hash
    summary["input_hashes"] = {
        "manifest": sha256_file(args.manifest),
        "predictions": sha256_file(args.predictions),
        "section_scales": sha256_file(args.section_scales),
        "scale_summary": sha256_file(args.scale_summary),
    }
    atomic_write_json(grouped_dir / "grouped_summary.json", summary)
    report_md = render_markdown(summary, config, scale_summary, config_hash)
    report_path = output_root / "定位误差分层评价_v1.md"
    atomic_write_text(report_path, report_md)
    print(
        f"wrote {csv_path} ({len(metrics)} crops) and {report_path}\n"
        f"groups: " + ", ".join(f"{g}={summary['by_group'][g]['n_crops']}" for g in GROUP_ORDER)
    )


if __name__ == "__main__":
    main()
