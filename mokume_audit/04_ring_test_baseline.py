# -*- coding: utf-8 -*-
"""Step 4: 同口径年轮基线 — Mokume UNet vs 阶段三 best.pt 年轮头，同一 132 张 ring test。

两个模型都在原生 256×256 坐标系下评估（与 ring_targets 原生分辨率一致）：
- Mokume UNet：官方推理流程（BGR/255 · 64×64 patch + 高斯融合）→ ARL 灰度图
  → 阈值 200（官方 get_image_open_contours 约定）二值化
- 阶段三共享模型 best.pt：letterbox 640 → ring 头 → 映射回原生 256×256
指标（与 train_stage3_instance_multitask.ring_metrics 同公式）：
- boundary Dice：2|P∩T∩V| / (|P∩V| + |T∩V|)，另报 skeleton Dice 作参考
- distance MAE：|pred − target| 在 valid 像素上取均值
  （Mokume UNet 无距离场输出，其距离场由阈值化 ARL 的 distanceTransform 推算，
   结果仅作参考并在报告中标注）
"""
import os
import sys
import json
import math
import csv
import time
from pathlib import Path
from itertools import product

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import cv2
import torch
import torch.nn as nn

PROJECT_ROOT = Path(r"E:\dataset-clean\wood-multitask-final")
SCRIPTS = PROJECT_ROOT / "scripts"
SRC = PROJECT_ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 导入训练脚本会自动执行 torchvision 兼容 shim（本地 torchvision _C.pyd 不可用）
from train_stage3_instance_multitask import (  # noqa: E402
    letterbox_geometry, resize_rgb, image_tensor,
)
from wood_data.loaders import WoodCatalog  # noqa: E402
from wood_data.instance_multitask import SharedInstanceMultitaskModel  # noqa: E402

MOKUME_CKPT = Path(r"E:\dataset-clean\Mokume\unet_trained_model.pt")
STAGE3_BEST = Path(r"D:\教务处实习\wood_preproject\new history\resnet34_instance_nobridge_v1\best.pt")
OUT_DIR = Path(r"D:\教务处实习\wood_preproject\树髓定位\mokume_audit\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARL_THRESHOLD = 200          # 官方约定阈值（get_image_open_contours THRESHOLD=200）
DISTANCE_TAU = 16.0          # 与 ring_targets.finish_targets 一致


# ═══════════════════════════════════════════════════════════════
# 官方 Mokume U-Net 架构（COMMON/unet.py，与 03_official_architecture.py 一致）
# ═══════════════════════════════════════════════════════════════

class TwoConvBlock(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_mid, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(ch_mid)
        self.rl = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(ch_mid, ch_out, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.rl(self.bn1(self.conv1(x)))
        return self.rl(self.bn2(self.conv2(x)))


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        return self.bn2(self.conv(self.bn1(self.up(x))))


class UNet_2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.TCB1 = TwoConvBlock(in_dim, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.UC1 = UpConv(1024, 512)
        self.UC2 = UpConv(512, 256)
        self.UC3 = UpConv(256, 128)
        self.UC4 = UpConv(128, 64)
        self.conv1 = nn.Conv2d(64, out_dim, kernel_size=1)

    def forward(self, x):
        x1 = self.TCB1(x)
        x2 = self.TCB2(self.maxpool(x1))
        x3 = self.TCB3(self.maxpool(x2))
        x4 = self.TCB4(self.maxpool(x3))
        x = self.TCB5(self.maxpool(x4))
        x = self.TCB6(torch.cat([x4, self.UC1(x)], dim=1))
        x = self.TCB7(torch.cat([x3, self.UC2(x)], dim=1))
        x = self.TCB8(torch.cat([x2, self.UC3(x)], dim=1))
        x = self.TCB9(torch.cat([x1, self.UC4(x)], dim=1))
        return self.conv1(x)


def official_patch_inference(bgr_img, unet, patch_size=64):
    """官方 64×64 patch + 高斯融合推理（批量 patch，等价于逐 patch）。"""
    h, w, _ = bgr_img.shape
    cy = cx = (patch_size - 1) / 2
    sigma2 = 8.0 * 8.0
    yy, xx = np.mgrid[0:patch_size, 0:patch_size]
    gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / sigma2).astype(np.float32)

    stride = patch_size // 2
    positions = []
    for yi in range(0, h, stride):
        for xi in range(0, w, stride):
            positions.append((min(yi, h - patch_size), min(xi, w - patch_size)))
    positions = sorted(set(positions))

    patches = np.stack([
        bgr_img[y:y + patch_size, x:x + patch_size, :] for y, x in positions
    ]).astype(np.float32) / 255.0
    tensor = torch.from_numpy(patches.transpose(0, 3, 1, 2))
    out = np.zeros((h, w), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(positions), 64):
            chunk = tensor[start:start + 64]
            pred = unet(chunk).numpy()[:, 0] * 255.0
            for (y, x), p in zip(positions[start:start + 64], pred):
                out[y:y + patch_size, x:x + patch_size] += gauss * p
                cnt[y:y + patch_size, x:x + patch_size] += gauss
    return np.clip(out / cnt, 0, 255).astype(np.uint8)


def dice(pred, target, valid):
    denom = int((pred & valid).sum()) + int((target & valid).sum())
    if denom == 0:
        return 0.0
    return 2.0 * int((pred & target & valid).sum()) / denom


def mae(pred, target, valid):
    if not valid.any():
        return 0.0
    return float(np.abs(pred - target)[valid].mean())


def derived_distance(skeleton_pred):
    """由二值化 ARL 骨架推算归一化距离场（仅参考；Mokume 无距离输出）。"""
    d = cv2.distanceTransform((1 - skeleton_pred.astype(np.uint8)), cv2.DIST_L2, 3)
    return np.minimum(d / DISTANCE_TAU, 1.0).astype(np.float32)


def stage3_ring_predict(model, rgb_img, imgsz=640):
    """阶段三 ring 头推理：letterbox → forward → 概率与距离映射回原生分辨率。"""
    src_h, src_w = rgb_img.shape[:2]
    letterboxed = resize_rgb(rgb_img, imgsz)
    tensor = image_tensor(letterboxed)
    with torch.no_grad():
        out = model([tensor], ring=True)
    prob = torch.sigmoid(out["ring_boundary"])[0, 0].numpy()
    dist = out["ring_distance"][0, 0].numpy()
    # letterbox 逆映射回原生坐标系
    _, _, rh, rw, top, left = letterbox_geometry((src_h, src_w), imgsz)
    prob = prob[top:top + rh, left:left + rw]
    dist = dist[top:top + rh, left:left + rw]
    prob = cv2.resize(prob, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    dist = cv2.resize(dist, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    return prob, dist


def main():
    torch.set_num_threads(8)
    catalog = WoodCatalog(project_root=PROJECT_ROOT)
    ring_test = list(catalog.ring_samples(splits={"test"}).records)
    bridge_ids = {r.sample_id for r in catalog.bridge_samples(splits={"test"}).records}
    ring_test = [r for r in ring_test if r.sample_id not in bridge_ids]
    print(f"ring test 样本数：{len(ring_test)}")

    print("加载 Mokume UNet ...")
    unet = UNet_2D(in_dim=3, out_dim=1)
    unet.load_state_dict(torch.load(MOKUME_CKPT, map_location="cpu", weights_only=False), strict=True)
    unet.eval()

    print("加载阶段三 best.pt ...")
    stage3 = SharedInstanceMultitaskModel("resnet34", 640, anchor_profile="thin_crack")
    ckpt = torch.load(STAGE3_BEST, map_location="cpu", weights_only=False)
    stage3.load_state_dict(ckpt["model"])
    stage3.eval()
    print(f"  best.pt epoch = {ckpt['epoch']}")

    rows = []
    t0 = time.time()
    for idx, record in enumerate(ring_test, 1):
        sample = catalog.load("ring", record.sample_id)
        rgb = np.asarray(sample["image"])                       # (256,256,3) RGB
        tgt = sample["targets"]
        boundary = tgt["boundary"].astype(bool)                 # 原生 256×256
        skeleton = tgt["skeleton"].astype(bool)
        distance_t = tgt["distance"].astype(np.float32)
        valid = tgt["valid"].astype(bool)

        # ── Mokume UNet（BGR 输入） ──
        bgr = rgb[:, :, ::-1].copy()
        arl = official_patch_inference(bgr, unet)
        mok_pred = (arl >= ARL_THRESHOLD)
        mok_dist = derived_distance(mok_pred)
        row = {
            "sample_id": record.sample_id,
            "mokume_dice_boundary": dice(mok_pred, boundary, valid),
            "mokume_dice_skeleton": dice(mok_pred, skeleton, valid),
            "mokume_mae_distance_derived": mae(mok_dist, distance_t, valid),
        }
        # 阈值敏感性（仅报告参考，不用于选优）
        for t in (100, 150, 250):
            p = (arl >= t)
            row[f"mokume_dice_boundary_t{t}"] = dice(p, boundary, valid)

        # ── 阶段三 best.pt 年轮头（同一原生坐标系） ──
        prob, dist = stage3_ring_predict(stage3, rgb)
        s3_pred = prob >= 0.5
        row.update({
            "stage3_dice_boundary": dice(s3_pred, boundary, valid),
            "stage3_dice_skeleton": dice(s3_pred, skeleton, valid),
            "stage3_mae_distance": mae(dist, distance_t, valid),
        })
        rows.append(row)

        if idx <= 4:
            vis = np.zeros((256, 256 * 3, 3), dtype=np.uint8)
            vis[:, :256] = rgb
            vis[:, 256:512] = cv2.cvtColor((s3_pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis[:, 512:] = cv2.cvtColor((mok_pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(OUT_DIR / f"ringtest_{record.sample_id.replace(':', '_')}_vis.png"), vis)
        if idx % 20 == 0:
            print(f"  {idx}/{len(ring_test)}  ({time.time() - t0:.0f}s)")

    # ── 汇总 ──
    keys = ["mokume_dice_boundary", "mokume_dice_skeleton", "mokume_mae_distance_derived",
            "stage3_dice_boundary", "stage3_dice_skeleton", "stage3_mae_distance"]
    summary = {"n": len(rows), "arl_threshold": ARL_THRESHOLD,
               "stage3_checkpoint_epoch": ckpt["epoch"]}
    for k in keys:
        vals = [r[k] for r in rows]
        summary[k + "_mean"] = float(np.mean(vals))
        summary[k + "_median"] = float(np.median(vals))
    summary["per_sample"] = rows
    (OUT_DIR / "ring_test_baseline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT_DIR / "ring_test_baseline.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== 同口径年轮基线（原生 256×256，132 张 ring test）=====")
    print(f"Mokume UNet  boundary Dice = {summary['mokume_dice_boundary_mean']:.4f}"
          f"（skeleton Dice = {summary['mokume_dice_skeleton_mean']:.4f}）")
    print(f"Mokume UNet  距离 MAE（推算）= {summary['mokume_mae_distance_derived_mean']:.4f}")
    print(f"阶段三 best.pt  boundary Dice = {summary['stage3_dice_boundary_mean']:.4f}"
          f"（skeleton Dice = {summary['stage3_dice_skeleton_mean']:.4f}）")
    print(f"阶段三 best.pt  距离 MAE = {summary['stage3_mae_distance_mean']:.4f}")
    print(f"\n结果已保存：{OUT_DIR / 'ring_test_baseline.json'} / .csv / 4 张可视化 PNG")


if __name__ == "__main__":
    main()
