"""
对比原始 LabelMe 标注 vs 拼接重建的标注，找出差异来源。
原始标注 → 裁剪JSON → convert_dataset.py → B_ann.png patches → 拼接重建
"""

import os, json, cv2
import numpy as np
from tqdm import tqdm

ANN_DIR  = r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\annotations\annual_rings"
ORIG_DIR = r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\images_no_background"
RECON_DIR = r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\每个年轮完整标记图"
OUT_DIR  = r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\每个年轮完整标记图\标注对比"
os.makedirs(OUT_DIR, exist_ok=True)

# 只看 B 面和 D 面
recon_files = [f for f in os.listdir(RECON_DIR) if f.endswith("_ann.png")]

for recon_file in tqdm(recon_files, desc="对比"):
    group = recon_file.replace("_ann.png", "")

    # 原始 LabelMe 标注
    ann_json = os.path.join(ANN_DIR, f"{group}.json")
    if not os.path.exists(ann_json):
        continue

    # 原始图片
    orig_img_path = os.path.join(ORIG_DIR, f"{group}.jpg")
    if not os.path.exists(orig_img_path):
        continue

    # 重建标注
    recon_path = os.path.join(RECON_DIR, recon_file)

    # 读取原图尺寸
    orig_img = cv2.imread(orig_img_path)
    H, W = orig_img.shape[:2]

    # ===== 渲染原始 LabelMe 标注 =====
    with open(ann_json, 'r') as f:
        data = json.load(f)

    orig_ann_canvas = np.full((H, W), 255, dtype=np.uint8)
    shapes = data['shapes']

    for shape in shapes:
        label = int(shape['label'])
        pts = np.array([[p[0], p[1]] for p in shape['points']], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(orig_ann_canvas, [pts], isClosed=False,
                          color=label + 1, thickness=3)

    recon_ann = cv2.imread(recon_path, cv2.IMREAD_GRAYSCALE)
    if recon_ann.shape[:2] != (H, W):
        recon_ann = cv2.resize(recon_ann, (W, H))

    # ===== 逐像素对比 =====
    orig_mask = orig_ann_canvas < 255
    recon_mask = recon_ann < 255

    both = orig_mask & recon_mask
    only_orig = orig_mask & ~recon_mask
    only_recon = recon_mask & ~orig_mask

    n_orig = orig_mask.sum()
    n_recon = recon_mask.sum()
    n_both = both.sum()
    n_only_orig = only_orig.sum()
    n_only_recon = only_recon.sum()

    # 一致性比率
    agreement = n_both / max(n_orig, n_recon, 1) * 100

    # ===== 生成对比图 =====
    # 左: 原始标注 (绿色)
    # 中: 重建标注 (红色)
    # 右: 重叠对比 (一致=白色, 仅原始=绿色, 仅重建=红色)

    def make_overlay(base_gray, mask, color_bgr):
        """在灰度底图上叠加彩色mask"""
        base_rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
        alpha = 0.5
        color = np.array(color_bgr, dtype=np.float32)
        overlay = base_rgb.astype(np.float32)
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
        return np.clip(overlay, 0, 255).astype(np.uint8)

    orig_gray = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)

    # Panel 1: 原始标注 (绿色)
    p1 = make_overlay(orig_gray, orig_mask, (0, 255, 0))
    cv2.putText(p1, f"Original: {n_orig} px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 2: 重建标注 (红色)
    p2 = make_overlay(orig_gray, recon_mask, (0, 0, 255))
    cv2.putText(p2, f"Reconstructed: {n_recon} px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 3: 差异图 — 一致=白, 仅原始=绿, 仅重建=红
    diff = np.zeros((H, W, 3), dtype=np.uint8)
    diff[both] = (200, 200, 200)          # 一致: 灰色
    diff[only_orig] = (0, 255, 0)          # 仅原始: 绿色
    diff[only_recon] = (0, 0, 255)         # 仅重建: 红色

    cv2.putText(diff, f"Match: {agreement:.1f}% | Green=OrigOnly Red=ReconOnly",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 拼接三栏
    result = np.hstack([p1, p2, diff])

    # 缩放（原图太大就缩小）
    max_w = 3000
    if result.shape[1] > max_w:
        scale = max_w / result.shape[1]
        result = cv2.resize(result, (max_w, int(result.shape[0] * scale)))

    out_path = os.path.join(OUT_DIR, f"{group}_compare.jpg")
    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])

    tqdm.write(f"{group}: 原始={n_orig}px 重建={n_recon}px "
               f"一致={agreement:.1f}% "
               f"仅原始={n_only_orig}px 仅重建={n_only_recon}px")

print(f"\n完成！对比图保存在: {OUT_DIR}")
