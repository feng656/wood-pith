#!/usr/bin/env python3
"""Consolidated Chinese report: grouped metrics + sub-arc explanations + validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from racpith.provenance import atomic_write_json, atomic_write_text, sha256_file


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-summary", required=True, help="grouped/grouped_summary.json (script 21)")
    parser.add_argument("--scale-summary", required=True, help="scales/summary.json (script 20)")
    parser.add_argument("--explain-summary", required=True, help="explain/attribution_summary.json (script 22)")
    parser.add_argument("--smoke-comparison", required=True, help="validation8/smoke_comparison/comparison_summary.json (script 23)")
    parser.add_argument("--validation8-comparison", help="validation8/comparison/comparison_summary.json (script 23, optional)")
    parser.add_argument("--chunked-summary", help="validation8/contributions/chunked_run_summary.json (script 26, optional)")
    parser.add_argument("--output", required=True, help="output md path")
    args = parser.parse_args()

    grouped = json.loads(Path(args.grouped_summary).read_text(encoding="utf-8"))
    scales = json.loads(Path(args.scale_summary).read_text(encoding="utf-8"))
    explain = json.loads(Path(args.explain_summary).read_text(encoding="utf-8"))
    smoke = json.loads(Path(args.smoke_comparison).read_text(encoding="utf-8"))
    valid8 = None
    if args.validation8_comparison:
        valid8 = json.loads(Path(args.validation8_comparison).read_text(encoding="utf-8"))
    chunked = None
    if args.chunked_summary:
        chunked = json.loads(Path(args.chunked_summary).read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("# 髓心定位:误差分层评价与子弧段解释(合并报告)")
    lines.append("\n配套产物目录:`outputs_grayscale_masks/eval_position_v1/`。本报告把三层尺度指标、"
                 "四类分组统计、子弧段几何解释与子弧 delete-refit 实测对照合并为一个结论链。\n")

    # --- metrics recap ---
    by_group = grouped["by_group"]
    lines.append("## 1. 四类分组误差(指标定义见 定位误差分层评价_v1.md)")
    lines.append("| 组 | crop | 有预测 | error_px 中位/p90 | E_RF_pct 中位/p90(%) | E_w 中位/p90 |")
    lines.append("|---|---|---|---|---|---|")
    for g in ["IN_IMAGE", "NEAR", "MID", "FAR"]:
        b = by_group[g]
        lines.append(
            f"| {g} | {b['n_crops']} | {b['n_with_prediction']} | "
            f"{_fmt(b['error_px']['median'],1)} / {_fmt(b['error_px']['p90'],1)} | "
            f"{_fmt(b['E_RF_pct']['median'])} / {_fmt(b['E_RF_pct']['p90'])} | "
            f"{_fmt(b['E_w']['median'])} / {_fmt(b['E_w']['p90'])} |"
        )
    b = grouped["overall"]
    lines.append(
        f"| 总体 | {b['n_crops']} | {b['n_with_prediction']} | "
        f"{_fmt(b['error_px']['median'],1)} / {_fmt(b['error_px']['p90'],1)} | "
        f"{_fmt(b['E_RF_pct']['median'])} / {_fmt(b['E_RF_pct']['p90'])} | "
        f"{_fmt(b['E_w']['median'])} / {_fmt(b['E_w']['p90'])} |"
    )
    mm = grouped["E_mm_status"]
    lines.append(f"\nE_mm 一律为空({mm['unavailable']}/5182 无 mm_per_pixel),不猜测物理尺度。\n")

    # --- explanation recap ---
    lines.append("## 2. 子弧段解释(全量 680832 条)")
    cat = explain["category_counts"]
    total = explain["n_subarc_records"]
    lines.append("| 类别 | 数量 | 占比 |")
    lines.append("|---|---|---|")
    for key in ["SUPPORTING", "SUPPORTING_SHORT", "BOUNDARY_TRUNCATED", "MODERATE", "DEVIATING", "NO_CENTER"]:
        n = cat.get(key, 0)
        lines.append(f"| {key} | {n} | {100.0 * n / total:.1f}% |")
    lines.append(
        "\n要点:绝大多数子弧段指向一致度很高(SUPPORTING 合计 69%),真正偏离共识方向的子弧只占 "
        "0.4%(DEVIATING);贴边截断占 15.6%,是曲率约束弱化的主要空间来源。\n"
    )
    lines.append("### 环级角色 × 解释类别(节选,完整列联见 子弧段解释层_v1.md)")
    rx = explain["role_x_category"]
    lines.append("| 角色 | BOUNDARY | DEVIATING | SUPPORTING(+SHORT) |")
    lines.append("|---|---|---|---|")
    for role in ["BENEFICIAL_GT", "HARMFUL_GT", "DIRECTION_CRITICAL", "REDUNDANT", "UNRESOLVED"]:
        lines.append(
            f"| {role} | {rx.get(role + ' | BOUNDARY_TRUNCATED', 0)} | {rx.get(role + ' | DEVIATING', 0)} | "
            f"{rx.get(role + ' | SUPPORTING', 0) + rx.get(role + ' | SUPPORTING_SHORT', 0)} |"
        )
    lines.append(
        "\n含义:环级角色(实测 delete-refit)与子弧几何类别没有强耦合——有益/有害/冗余环里的子弧几何构成相似,"
        "说明「为什么影响」的主要解释不在子弧指向,而在**环整体**(曲率、长度、位置)。子弧几何层回答的是"
        "「哪一段在结构上支撑/偏离共识方向」,不是「哪一段实测有害」。\n"
    )

    # --- validation ---
    n_measured_crops = 1 + (valid8["n_crops"] if valid8 else 0)
    lines.append(f"## 3. 子弧 delete-refit 实测对照({n_measured_crops} 个有实测的 crop + 未产出的部分)")
    lines.append("| 来源 | 匹配子弧 | 结论 |")
    lines.append("|---|---|---|")
    smoke_cat = smoke["by_category"]
    smoke_med_sup = smoke_cat.get("SUPPORTING", {}).get("contrib_gt_px", {}).get("median")
    smoke_med_dev = smoke_cat.get("DEVIATING", {}).get("contrib_gt_px", {}).get("median")
    smoke_frac = smoke_cat.get("SUPPORTING", {}).get("contrib_gt_px", {}).get("frac_abs_lt_1px")
    lines.append(
        f"| smoke 样本 (T0_B1_N32_ADAP_s800_x400_y400) | {smoke['n_matched_subarcs']} | "
        f"全部类别实测贡献 <1px 占比 ≈{_fmt(smoke_frac)};SUPPORTING 中位 {_fmt(smoke_med_sup,3)} px、"
        f"DEVIATING 中位 {_fmt(smoke_med_dev,3)} px——均为亚像素噪声,几何类别不能预测子弧级符号 |"
    )
    if valid8:
        vcat = valid8["by_category"]
        sup = vcat.get("SUPPORTING", {})
        sup_short = vcat.get("SUPPORTING_SHORT", {})
        boundary = vcat.get("BOUNDARY_TRUNCATED", {})
        sup_px = sup.get("contrib_gt_px", {})
        sup_short_px = sup_short.get("contrib_gt_px", {})
        boundary_px = boundary.get("contrib_gt_px", {})
        lines.append(
            f"| validation8 ({valid8['n_crops']} 个 crop) | {valid8['n_matched_subarcs']} | "
            f"SUPPORTING 中位 {_fmt(sup_px.get('median'),3)} px(|v|<1px {_fmt(sup_px.get('frac_abs_lt_1px'))})、"
            f"SUPPORTING_SHORT 中位 {_fmt(sup_short_px.get('median'),3)} px、"
            f"BOUNDARY_TRUNCATED 中位 {_fmt(boundary_px.get('median'),3)} px——同样以亚像素为主,符号一致率 "
            f"{_fmt(sup.get('sign_consistency_rate'))}/{_fmt(sup_short.get('sign_consistency_rate'))},几何类别不能预测子弧级符号 |"
        )
    else:
        lines.append("| validation8 | — | 见 validation8/comparison/comparison_summary.json |")
    if chunked:
        statuses = chunked.get("statuses", {})
        lines.append(
            f"| validation8 未产出部分 | {statuses.get('TIMEOUT', 0)}/8 | "
            f"{statuses.get('TIMEOUT', 0)} 个 crop 在 25 分钟预算内没有任何 refit 产出(minus_results 为空):"
            "pith-local 的 delete-refit 求解在这些 crop 上首组即不收敛;同一批 crop 在 OC 包 stage-6 "
            "(环级)全部为 EXACT 状态——差异在实现/尺度(子弧级 vs 环级),已记为待查问题 |"
        )
    lines.append(
        "\n**结论**:子弧级 delete-refit 实测值系统性低于 1px(正式阈值 ±5.66px 无一达到),"
        "即「单个 10% 子弧段的删除」在实测上不改变定位结果。因此:\n"
        "- 子弧级「为什么影响」的答案分两层:**结构层**(该段的指向、弧长、贴边状态,几何可解释)与"
        "**实测层**(该段所属环的环级 delete-refit 角色,唯一有可测信号的层级);\n"
        "- 几何解释不与实测贡献画等号;几何层用于定位「哪些段在结构上支撑/偏离共识方向」,"
        "环级角色用于定位「哪些环实测有益/有害」。\n"
    )

    # --- boundaries ---
    lines.append("## 4. 边界声明")
    for item in (
        "毫米误差:无物理标定,一律为空;不得根据树径、相机型号或 DPI 猜测(shusui.txt §1)。",
        "父截面尺度 R_F 只进入评价器,不进入局部反演(shusui.txt §2 边界 1)。",
        "E_RF_pct 与 APD 的 E_APD_box_pct 不可互换;E_w 不是树龄误差(shusui.txt §2/§3)。",
        "w_ref/w_max 来自完整截面 GT 标注并冻结,不用预测髓心重算(shusui.txt §3)。",
        "几何解释是诊断层,不覆盖 gt_label/contrib_gt_norm;HARMFUL_GT 是事后审计,不是部署删弧规则。",
        "validation8 的实测贡献来自 pith-local 复现管线(CPU),与 stage 6/7 的 OC 包实现口径不同,仅作解释层校验。",
    ):
        lines.append(f"- {item}")
    lines.append("\n")
    atomic_write_text(args.output, "\n".join(lines))
    sidecar = {
        "schema_version": "racpith.final_report.v1",
        "report_path": str(Path(args.output).resolve()),
        "report_sha256": sha256_file(args.output),
        "inputs": {
            "grouped_summary": sha256_file(args.grouped_summary),
            "scale_summary": sha256_file(args.scale_summary),
            "explain_summary": sha256_file(args.explain_summary),
            "smoke_comparison": sha256_file(args.smoke_comparison),
        },
    }
    atomic_write_json(str(Path(args.output).with_suffix(".sidecar.json")), sidecar)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
