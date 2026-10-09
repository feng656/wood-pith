#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_urudendro_no_background"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.1f}%"


def main() -> int:
    m0 = load(RESULTS / "step3_v2" / "m0_screen.json")
    m1 = load(RESULTS / "step3_v2" / "m1rpc_screen.json")
    repair = load(RESULTS / "bias_audit" / "repair_decision.json")
    bias = load(RESULTS / "bias_audit" / "bias_stratified_summary.json")
    skipped = load(RESULTS / "blind" / "SKIPPED.json")
    obs = load(RESULTS / "observability" / "stratified_summary.json")
    contribution = load(RESULTS / "contribution" / "summary.json")
    lines = [
        "# OC-ArcPith-RG Adjusted: UruDendro4 No-Background Experiment Report",
        "",
        "## Scope and evidence level",
        "",
        "This run follows EXPERIMENT_GUIDE.md in A-F order. UruDendro4 contains only four biological trees (T0, T2, T4, T6), so results are exploratory and do not provide an independent sealed-test estimate. Overlapping crops are correlated views, not independent biological samples.",
        "",
        "## Step A - Data grouping and manifest",
        "",
        "- PASS: 5,182 crops, 4 biological trees, 102 sections, 72,975 visible curve fragments.",
        "- All images and pith coordinate transforms were validated.",
        "- All 102 full closed-ring annotations have unique numeric labels whose polygon areas strictly increase with label; this is the evidence used to assign explicit inner-to-outer parent-ring order.",
        "- 2,670 crops contain multiple disconnected fragments from the same parent ring. They are retained separately; no artificial chord is inserted.",
        "",
        "## Step B - Candidate screening",
        "",
        f"M0 decision: **{m0['decision']}**. M1-RPC decision: **{m1['decision']}**.",
        "",
        "| Tree | M0 global truth-beats | M0 direction gap | M0 range gap | M0 translation gap | M1-RPC translation gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    m1_by_tree = {row["tree_id"]: row for row in m1["tree_metrics"]}
    for row in m0["tree_metrics"]:
        other = m1_by_tree[row["tree_id"]]
        lines.append(f"| {row['tree_id']} | {row['global_truth_beats_fraction']:.3f} | {row['direction_gap']:.3f} | {row['range_gap']:.3f} | {row['translation_gap']:.3f} | {other['translation_gap']:.3f} |")
    lines += [
        "",
        "Global direction/range geometry is separable on all four trees, but every tree has a negative local translation gap. M1-RPC changes the gap only marginally.",
        "",
        "## Step C - Local bias audit",
        "",
        f"Decision: **{repair['decision']}**. Paired samples: {repair['paired_samples']}; tree pass fraction: {repair['tree_pass_fraction']:.1f}; median tree bias reduction: {repair['median_tree_bias_reduction']:.3g}.",
        "",
        "| Crop size | N | M0 median bias norm | M1-RPC median bias norm | Median reduction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size, row in sorted(bias["by_size"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {size} | {row['n']} | {row['median_m0_bias_norm']:.4f} | {row['median_m1_bias_norm']:.4f} | {pct(row['median_bias_reduction'])} |")
    lines += [
        "",
        "M1-RPC does not reach the preregistered 20% tree-level reduction in any tree or size stratum. For M0, the median local bias improves strongly with larger crops: 0.1500 (800), 0.0638 (1200), 0.0263 (1600).",
        "",
        "## Step D - Blind inversion and tree-group CV",
        "",
        f"**{skipped['status']}**. {skipped['reason']}",
        "",
        "No G2 or KEEP_M1RPC claim is made. Running the long M1-RPC inversion after the failed Step C gate would violate the guide. M0 is retained only as the descriptive baseline for Steps E-F.",
        "",
        "## Step E - Observability / feasibility",
        "",
        f"Internal M0 profile states: POINT {obs['state_counts']['POINT']}, AXIS {obs['state_counts']['AXIS']}, MULTIMODAL {obs['state_counts']['MULTIMODAL']}, REJECT {obs['state_counts']['REJECT']}.",
        f"The GT is inside the preregistered profile support for only {pct(obs['gt_within_profile_support_fraction'])} of crops, despite {pct(obs['state_fractions']['POINT'])} receiving an internal POINT label. Therefore POINT means a sharp finite M0 solution, not necessarily a correct pith.",
        "",
        "| Stratum | N | POINT | GT in support |",
        "|---|---:|---:|---:|",
    ]
    for name, row in obs["by_true_pith_distance"].items():
        lines.append(f"| {name} | {row['n']} | {pct(row['state_fractions']['POINT'])} | {pct(row['gt_within_profile_support_fraction'])} |")
    lines += [
        "",
        "| Crop size | N | POINT | GT in support |",
        "|---:|---:|---:|---:|",
    ]
    for name, row in sorted(obs["by_crop_size_px"].items(), key=lambda item: int(item[0])):
        lines.append(f"| {name} | {row['n']} | {pct(row['state_fractions']['POINT'])} | {pct(row['gt_within_profile_support_fraction'])} |")
    lines += [
        "",
        "Far-field reliability deteriorates even when the M0 profile remains sharp: GT support falls from 92.8% inside the crop to 72.0% near-outside, 46.5% mid-outside, and 32.4% far-outside.",
        "",
        "## Step F - Parent-ring contribution",
        "",
        f"A frozen 24-sample sentinel covers 4 trees x 3 crop sizes x pith inside/outside. It produced {contribution['parent_ring_deletions']} parent-ring deletion results; all {contribution['information_rows_with_positive_definite_hessian']} Hessian comparisons were positive definite.",
        "",
        f"Raw negative GT effects occur in {pct(contribution['gt_conflict_fraction'])} of deletions, but the median effect is only {contribution['gt_effect_quantiles']['median']:.5f}. Effect-size-aware conflict rates are {pct(contribution['significant_gt_conflict_fraction_le_minus_0.005'])} at <= -0.005 norm and {pct(contribution['strong_gt_conflict_fraction_le_minus_0.01'])} at <= -0.01 norm.",
        f"Significant help is {pct(contribution['significant_gt_help_fraction_ge_0.005'])} at >= 0.005 norm. {contribution['samples_with_high_leverage_significant_conflict']}/24 samples contain at least one high-leverage, significant-conflict parent ring.",
        f"Median within-sample Spearman support-information correlation is {contribution['median_within_sample_support_info_spearman']:.3f}; support-leverage is {contribution['median_within_sample_support_leverage_spearman']:.3f}. Visible support alone is not a reliable contribution score.",
        "",
        "## 有实验依据的最终结论",
        "",
        "1. 无背景、允许重叠的裁剪提高了几何信息量，尤其是 1200 和 1600 像素裁剪，但仍未消除 M0 的局部平移偏差。",
        "2. 在父年轮顺序经过审计、断裂片段得到正确处理后，M1-RPC 仍未修复该偏差。因此不能通过增大 lambda 强行让 M1-RPC 通过。",
        "3. M0 经常给出尖锐且有限的最优点，但真实髓心仍可能落在支持域之外；该问题在髓心位于裁剪框外中等距离或远距离时尤其明显。因此，仅根据 profile 形状给出的可定位性判断会过度自信，部署前必须使用留出树进行校准。",
        "4. 父年轮的可见支持量、信息量、杠杆作用以及对真实髓心定位的帮助是不同概念。一部分年轮虽然信息量较高，却与真实髓心产生冲突。这支持下一步研究显式的共享形变干扰模型，而不是继续叠加更强的惩罚项。",
        "5. 当前数据只有 4 棵生物学意义上的树，不能据此得出最终泛化结论。要满足实验指南规定的正式 5 折门槛并进行独立封存测试，还需要增加树木数量。",
        "",
        "## 下一步有依据的实验",
        "",
        "保留 M0 作为基线，优先使用 1200/1600 像素裁剪，固定按树和截面划分数据的规则；加入显式的共享形变干扰模型，并在留出的生物学树木上检验它能否降低局部偏差。根据当前证据，不应继续运行耗时的 M1-RPC 盲反演。",
        "",
    ]
    output = RESULTS / "FINAL_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
