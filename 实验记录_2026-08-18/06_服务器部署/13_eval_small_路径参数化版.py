"""Evaluate the small-data run on all validation patches.

对 manifest_small 的 val 集逐张推理，统计：
- 总体与 near/far 分组的髓心误差（median/mean，只统计管线给出有限估计的样本）
- 拒识率（status=rejected / 无有限 center）
- 逐张明细 CSV + 4 张代表性 overlay
口径如实记录：本数据集无物理标定，误差单位为像素；10 轮 384 画布小数据运行，
仅用于验证链路与观察趋势，不构成正式精度结论。
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

from oapith.config import load_config
from oapith.workflows import run_inference

# ── Edit these（跨机器）───────────────────────────────────────
# 配置路径可用环境变量 OA_PITH_EVAL_CONFIG 覆盖（服务器运行用 urudendro_server.yaml）；
# data.root / output_directory / manifest 均从该配置读取，无需改本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMALL_CONFIG = Path(
    os.environ.get("OA_PITH_EVAL_CONFIG", str(PROJECT_ROOT / "configs/urudendro_small.yaml"))
)
_CONFIG = load_config(SMALL_CONFIG)
DATA_ROOT = Path(_CONFIG["data"]["root"])
MANIFEST_SMALL = PROJECT_ROOT / Path(_CONFIG["data"]["manifest"])
CHECKPOINT = Path(
    os.environ.get(
        "OA_PITH_EVAL_CHECKPOINT",
        str(PROJECT_ROOT / _CONFIG.get("output_directory", "runs/oa_pith") / "best.pt"),
    )
)
OUTPUT_DIR = PROJECT_ROOT / _CONFIG.get("output_directory", "runs/oa_pith") / "eval"
DEVICE = os.environ.get("OA_PITH_EVAL_DEVICE", "cpu")
OVERLAY_EVERY = 6          # 每 6 张出一张 overlay（24 张 val → 4 张）
# ───────────────────────────────────────────────────────────────

FIELD_NAMES = [
    "sample_id", "size_px", "pith_in_patch", "dist_to_pith", "angle_span_deg",
    "n_curves", "state_decision", "status", "pred_x", "pred_y",
    "data_rank", "error_px",
]


def main() -> None:
    os.chdir(PROJECT_ROOT)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT} (train first)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in MANIFEST_SMALL.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_records = [r for r in records if r.get("split") == "validation"]
    print(f"val 样本 {len(val_records)} 张，开始推理 …")

    rows: list[dict] = []
    t0 = time.time()
    for index, record in enumerate(val_records, 1):
        image_path = DATA_ROOT / record["image"]
        output_json = OUTPUT_DIR / f"{record['sample_id']}_prediction.json"
        overlay_png = (
            OUTPUT_DIR / f"{record['sample_id']}_overlay.png"
            if index % OVERLAY_EVERY == 1 or index == len(val_records)
            else None
        )
        if not output_json.is_file():
            run_inference(
                SMALL_CONFIG,
                CHECKPOINT,
                image_path,
                output_json,
                overlay_path=overlay_png,
                device=DEVICE,
                mm_per_pixel=None,
                exact_contributions=False,
                bootstrap_replicates=0,
            )
        prediction = json.loads(output_json.read_text(encoding="utf-8"))
        modes = prediction.get("modes") or []
        best = max(modes, key=lambda value: value.get("weight", 0.0)) if modes else None
        center = best.get("center_pixel_unclipped") if best else None
        rank = (best.get("data_information") or {}).get("rank") if best else None
        error = None
        if center is not None:
            error = math.hypot(center[0] - record["pith_px"][0], center[1] - record["pith_px"][1])
        rows.append({
            "sample_id": record["sample_id"],
            "size_px": record["metadata"]["size_px"],
            "pith_in_patch": record["metadata"]["pith_in_patch"],
            "dist_to_pith": record["metadata"]["dist_to_pith"],
            "angle_span_deg": record["metadata"]["angle_span_deg"],
            "n_curves": len(record["curves"]),
            "state_decision": prediction.get("state_decision"),
            "status": prediction.get("status"),
            "pred_x": round(center[0], 2) if center else "",
            "pred_y": round(center[1], 2) if center else "",
            "data_rank": rank if rank is not None else "",
            "error_px": round(error, 2) if error is not None else "",
        })
        elapsed = time.time() - t0
        print(f"  [{index}/{len(val_records)}] {record['sample_id']} "
              f"state={prediction.get('state_decision')} error={round(error) if error is not None else '—'}px "
              f"({elapsed / index:.0f}s/张, 剩余 {(len(val_records) - index) * elapsed / index / 60:.0f}min)")

    with (OUTPUT_DIR / "eval_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)

    def stats(subset: list[dict], label: str) -> dict:
        # 口径：status=rejected（joint 后验判 null）是无主张，不计入误差统计，单独计拒识率
        accepted = [r for r in subset if r["error_px"] != "" and r["status"] != "rejected"]
        errors = [r["error_px"] for r in accepted]
        rejected = sum(1 for r in subset if r["error_px"] == "" or r["status"] == "rejected")
        summary = {
            "n": len(subset),
            "n_accepted": len(errors),
            "n_rejected_or_no_finite": rejected,
            "median_error_px": round(statistics.median(errors), 1) if errors else None,
            "mean_error_px": round(statistics.mean(errors), 1) if errors else None,
        }
        print(f"\n== {label} ==")
        print(f"   样本 {summary['n']} · 接受（有限估计） {summary['n_accepted']} · "
              f"拒识/无有限点 {summary['n_rejected_or_no_finite']}")
        if errors:
            print(f"   误差: median={summary['median_error_px']}px mean={summary['mean_error_px']}px")
        return summary

    summary = {
        "config": str(SMALL_CONFIG),
        "checkpoint": str(CHECKPOINT),
        "overall": stats(rows, "总体"),
        "near": stats([r for r in rows if r["pith_in_patch"] is True], "near（髓心在图内）"),
        "far": stats([r for r in rows if r["pith_in_patch"] is False], "far（髓心在图外）"),
        "rows": rows,
    }
    (OUTPUT_DIR / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n总耗时 {time.time() - t0:.0f}s · 结果目录 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
