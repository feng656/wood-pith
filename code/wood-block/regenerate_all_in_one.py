"""
一次性重建：dataset、位置JSON、拼接图、叠加图、对比图。
确保所有步骤使用完全一致的网格位置。
"""

import os, json, cv2, shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

ANNOTATIONS = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\annotations\annual_rings")
ORIG_IMAGES = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\images_no_background")
BASE_OUT    = Path(r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望")

DATASET_DIR = BASE_OUT / "dataset"
TRAIN_FILE  = BASE_OUT / "train.txt"
CROP_DIR    = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
STITCH_DIR  = BASE_OUT / "每个年轮完整标记图"
OVERLAY_DIR = STITCH_DIR / "与原图叠加"
COMPARE_DIR = STITCH_DIR / "标注对比"

PATCH_SIZE = 1000
STRIDE     = 500


def render_full_annotation(ann_json_path, img_w, img_h):
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
    return (crop < 255).sum() >= min_pixels


def make_overlay(base_gray, mask, color_bgr):
    base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    alpha = 0.5
    color = np.array(color_bgr, dtype=np.float32)
    overlay = base_rgb.astype(np.float32)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    # 清理并重建所有输出目录
    for d in [DATASET_DIR, STITCH_DIR, OVERLAY_DIR, COMPARE_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # 清理 B/D 面的旧 CROP_JSON
    for group_dir in sorted(CROP_DIR.iterdir()):
        if not group_dir.is_dir():
            continue
        parts = group_dir.name.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('B', 'D'):
            shutil.rmtree(group_dir)

    # 收集 B/D 面样本
    ann_files = sorted(ANNOTATIONS.glob("*.json"))
    bd_samples = []
    for af in ann_files:
        sample = af.stem
        parts = sample.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('B', 'D'):
            bd_samples.append((sample, af))

    print(f"找到 {len(bd_samples)} 个 B/D 面样本\n")

    all_train_ids = []

    for sample, ann_path in tqdm(bd_samples, desc="处理样本"):
        img_path = ORIG_IMAGES / f"{sample}.jpg"
        if not img_path.exists():
            tqdm.write(f"  [跳过] 找不到: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        face = sample.rsplit('_', 1)[1]
        orig_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # 渲染全图标注
        full_canvas = render_full_annotation(ann_path, W, H)

        # 计算网格并同时写入 dataset + 位置JSON
        sample_dir = CROP_DIR / sample
        sample_dir.mkdir(parents=True, exist_ok=True)

        positions = []  # (patch_name, bx, by, size)
        seen_positions = set()  # 去重
        n = 0

        # 用 range(0, H, STRIDE) 确保覆盖到边缘，
        # 配合 x1/x2/y1/y2 的 clamp 逻辑处理边界
        for y in range(0, H, STRIDE):
            for x in range(0, W, STRIDE):
                x2 = min(x + PATCH_SIZE, W)
                y2 = min(y + PATCH_SIZE, H)
                x1 = max(0, x2 - PATCH_SIZE)
                y1 = max(0, y2 - PATCH_SIZE)

                # 去重（边界处可能产生相同坐标）
                key = (x1, y1)
                if key in seen_positions:
                    continue

                crop = full_canvas[y1:y2, x1:x2]

                if not has_annotations(crop):
                    continue  # 跳过空白patch

                seen_positions.add(key)
                patch_name = f"{sample}_s{PATCH_SIZE}_n{n}"
                patch_dir = DATASET_DIR / patch_name
                patch_dir.mkdir(parents=True, exist_ok=True)

                # 保存 B_ann.png 或 D_ann.png
                cv2.imwrite(str(patch_dir / f"{face}_ann.png"), crop)

                # 保存位置 JSON
                with open(sample_dir / f"{patch_name}.json", 'w') as f:
                    json.dump({
                        'patch_name': patch_name,
                        'sample': sample,
                        'size': PATCH_SIZE,
                        'bx': x1,
                        'by': y1,
                    }, f)

                positions.append((patch_name, x1, y1, PATCH_SIZE))
                all_train_ids.append(patch_name)
                n += 1

        # --- 拼接重建 ---
        canvas = np.full((H, W), 255, dtype=np.uint8)
        for patch_name, bx, by, size in positions:
            ann_path_p = DATASET_DIR / patch_name / f"{face}_ann.png"
            ann_img = cv2.imread(str(ann_path_p), cv2.IMREAD_GRAYSCALE)
            if ann_img is None:
                continue
            h, w = ann_img.shape
            crop_h = min(h, size, H - by)
            crop_w = min(w, size, W - bx)
            if crop_h <= 0 or crop_w <= 0:
                continue
            patch_data = ann_img[:crop_h, :crop_w]
            mask = patch_data < 255
            canvas[by:by + crop_h, bx:bx + crop_w][mask] = patch_data[mask]

        cv2.imwrite(str(STITCH_DIR / f"{sample}_ann.png"), canvas)

        # --- 叠加原图 ---
        recon_mask = canvas < 255
        overlay = make_overlay(orig_gray, recon_mask, (0, 0, 255))
        cv2.putText(overlay, "Reconstructed", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(OVERLAY_DIR / f"{sample}_overlay.jpg"),
                    overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # --- 对比原始标注 ---
        orig_ann = render_full_annotation(ann_path, W, H)
        orig_mask = orig_ann < 255

        both = orig_mask & recon_mask
        only_orig = orig_mask & ~recon_mask
        only_recon = recon_mask & ~orig_mask

        agreement = both.sum() / max(orig_mask.sum(), recon_mask.sum(), 1) * 100

        p1 = make_overlay(orig_gray, orig_mask, (0, 255, 0))
        cv2.putText(p1, f"Original: {orig_mask.sum()}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        p2 = make_overlay(orig_gray, recon_mask, (0, 0, 255))
        cv2.putText(p2, f"Recon: {recon_mask.sum()}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        diff = np.zeros((H, W, 3), dtype=np.uint8)
        diff[both] = (200, 200, 200)
        diff[only_orig] = (0, 255, 0)
        diff[only_recon] = (0, 0, 255)
        cv2.putText(diff, f"Match={agreement:.1f}% Green=OrigOnly Red=ReconOnly",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        result = np.hstack([p1, p2, diff])
        max_w = 3000
        if result.shape[1] > max_w:
            scale = max_w / result.shape[1]
            result = cv2.resize(result, (max_w, int(result.shape[0] * scale)))

        cv2.imwrite(str(COMPARE_DIR / f"{sample}_compare.jpg"),
                    result, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 日志
        gap_orig = only_orig.sum()
        gap_recon = only_recon.sum()
        status = "✓ PERFECT" if gap_orig == 0 and gap_recon == 0 else \
                 f"orig_gap={gap_orig} recon_gap={gap_recon}"
        tqdm.write(f"  {sample}: {n} patches, agreement={agreement:.1f}%, {status}")

    # 写 train.txt
    all_train_ids.sort()
    with open(TRAIN_FILE, 'w') as f:
        for tid in all_train_ids:
            f.write(tid + '\n')

    print(f"\n{'='*60}")
    print(f"全部完成!")
    print(f"  dataset: {len(all_train_ids)} 个 patch → {DATASET_DIR}")
    print(f"  拼接图:  {STITCH_DIR}")
    print(f"  叠加图:  {OVERLAY_DIR}")
    print(f"  对比图:  {COMPARE_DIR}")
    print(f"  train.txt: {TRAIN_FILE}")


if __name__ == "__main__":
    main()
