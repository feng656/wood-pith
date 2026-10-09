"""
用系统化网格 patch 重新生成 dataset，确保原图 100% 覆盖。

网格 size=1000, stride=500 (50% 重叠)，每个 B/D 面都被完整覆盖。
直接从原始 LabelMe 标注渲染后裁剪，与原始标注完全一致。
"""

import os, json, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

ANNOTATIONS = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\annotations\annual_rings")
ORIG_IMAGES = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\images_no_background")
OUT_DATASET  = Path(r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\dataset")
OUT_TRAIN    = Path(r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\train.txt")

PATCH_SIZE = 1000
STRIDE     = 500   # 50% 重叠

def render_full_annotation(ann_json_path, img_w, img_h):
    """渲染原始 LabelMe 标注到全图大小画布"""
    with open(ann_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    canvas = np.full((img_h, img_w), 255, dtype=np.uint8)
    for shape in data.get('shapes', []):
        pts = np.array([[p[0], p[1]] for p in shape['points']], dtype=np.int32)
        if len(pts) < 2:
            continue
        label = int(shape['label']) + 1
        cv2.polylines(canvas, [pts], isClosed=False, color=label, thickness=3)
    return canvas


def has_annotations(crop, min_pixels=100):
    """检查裁剪块是否包含足够多的标注像素"""
    return (crop < 255).sum() >= min_pixels


def main():
    # 找出所有 B 面和 D 面的原始样本
    ann_files = sorted(ANNOTATIONS.glob("*.json"))

    bd_samples = []
    for af in ann_files:
        sample = af.stem  # e.g. T0_B1_N27_B
        parts = sample.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('B', 'D'):
            bd_samples.append((sample, af))

    print(f"找到 {len(bd_samples)} 个 B/D 面样本")

    # 清理旧 dataset
    import shutil
    if OUT_DATASET.exists():
        shutil.rmtree(OUT_DATASET)
    OUT_DATASET.mkdir(parents=True, exist_ok=True)

    all_patch_ids = []
    total_patches = 0

    for sample, ann_path in tqdm(bd_samples, desc="网格化"):
        img_path = ORIG_IMAGES / f"{sample}.jpg"
        if not img_path.exists():
            tqdm.write(f"  [跳过] 找不到: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # 渲染全图标注
        full_canvas = render_full_annotation(ann_path, W, H)

        # 网格划分
        n = 0
        for y in range(0, max(1, H - PATCH_SIZE + 1), STRIDE):
            for x in range(0, max(1, W - PATCH_SIZE + 1), STRIDE):
                # 边界裁切
                x2 = min(x + PATCH_SIZE, W)
                y2 = min(y + PATCH_SIZE, H)
                x1 = max(0, x2 - PATCH_SIZE)
                y1 = max(0, y2 - PATCH_SIZE)

                crop = full_canvas[y1:y2, x1:x2]

                if not has_annotations(crop):
                    continue

                patch_name = f"{sample}_s{PATCH_SIZE}_n{n}"
                patch_dir = OUT_DATASET / patch_name
                patch_dir.mkdir(parents=True, exist_ok=True)

                face = sample.rsplit('_', 1)[1]  # B or D
                cv2.imwrite(str(patch_dir / f"{face}_ann.png"), crop)
                all_patch_ids.append(patch_name)
                n += 1

        total_patches += n
        if n > 0:
            tqdm.write(f"  {sample}: {W}×{H} → {n} patches (grid)")

    # 写 train.txt
    all_patch_ids.sort()
    with open(OUT_TRAIN, 'w') as f:
        for pid in all_patch_ids:
            f.write(pid + '\n')

    print(f"\n✅ 完成!")
    print(f"   共 {total_patches} 个 patch（网格平铺，100% 覆盖）")
    print(f"   train.txt: {len(all_patch_ids)} 个样本")
    print(f"\n   后续步骤:")
    print(f"   1. python stitch_ann_patches.py  # 重新拼接")
    print(f"   2. python overlay_ann_originals.py  # 重新叠加")
    print(f"   3. python compare_annotations.py  # 重新对比验证")


if __name__ == "__main__":
    main()
