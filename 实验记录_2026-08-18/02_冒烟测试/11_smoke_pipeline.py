"""Smoke: 1-epoch train on the 36-sample manifest + inference on one val patch.

用 UruDendro 真实数据验证完整链路（训练 1 轮 → 检查点 → tiled 推理 →
弧提取 → 几何后验 → 输出 JSON），全部 CPU、快速 geometry、无 LOAO/bootstrap。
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from oapith.workflows import run_inference, run_training

# ── Edit these ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs/urudendro_smoke.yaml"
DEVICE = "cpu"
# ───────────────────────────────────────────────────────────────


def pick_val_image(manifest_path: Path) -> tuple[dict, Path]:
    """选一张 val 且髓心在图内的 patch 作推理冒烟（无则退回第一张 val）。"""
    data_root = Path("D:/教务处实习/wood_preproject/树髓定位")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_records = [r for r in records if r.get("split") == "validation"]
    assert val_records, "val split is empty"
    for record in val_records:
        if record.get("metadata", {}).get("pith_in_patch"):
            return record, data_root / record["image"]
    record = val_records[0]
    return record, data_root / record["image"]


def main() -> None:
    os.chdir(PROJECT_ROOT)
    print(f"Device: {DEVICE}")

    t0 = time.time()
    print("== 1) 训练 1 轮（36 样本 manifest）==")
    run_training(SMOKE_CONFIG, device=DEVICE)
    print(f"   训练耗时 {time.time() - t0:.0f}s")

    checkpoint = PROJECT_ROOT / "runs/oa_pith_smoke/best.pt"
    if not checkpoint.is_file():
        checkpoint = PROJECT_ROOT / "runs/oa_pith_smoke/last.pt"
    print(f"   checkpoint: {checkpoint.name} ({checkpoint.stat().st_size / 1e6:.1f} MB)")

    record, image_path = pick_val_image(PROJECT_ROOT / "data/manifest_smoke.jsonl")
    output_json = PROJECT_ROOT / "runs/oa_pith_smoke/smoke_prediction.json"
    overlay_png = PROJECT_ROOT / "runs/oa_pith_smoke/smoke_overlay.png"
    print(f"== 2) 推理: {record['sample_id']}  image={image_path}")

    t1 = time.time()
    run_inference(
        SMOKE_CONFIG,
        checkpoint,
        image_path,
        output_json,
        overlay_path=overlay_png,
        device=DEVICE,
        mm_per_pixel=None,
        exact_contributions=False,
        bootstrap_replicates=0,
    )
    print(f"   推理耗时 {time.time() - t1:.0f}s")

    prediction = json.loads(output_json.read_text(encoding="utf-8"))
    print(f"\n== 推理输出（{output_json.name}）==")
    print(f"   status={prediction.get('status')}  state_decision={prediction.get('state_decision')}")
    modes = prediction.get("modes") or []
    for i, mode in enumerate(modes):
        center = mode.get("center_pixel_unclipped")
        print(f"   mode[{i}] weight={mode.get('weight'):.3f} center_px={center} "
              f"output_type={mode.get('output_type')}")

    if record.get("pith_px") is not None and modes:
        center = modes[0].get("center_pixel_unclipped")
        if center is not None:
            error = math.hypot(center[0] - record["pith_px"][0], center[1] - record["pith_px"][1])
            print(f"\n   GT 髓心(px): {record['pith_px']}   预测: {center}")
            print(f"   误差: {error:.0f}px（1 轮训练冒烟，仅验证链路，不评估精度）")
    print(f"\n总耗时 {time.time() - t0:.0f}s ✅")


if __name__ == "__main__":
    main()
