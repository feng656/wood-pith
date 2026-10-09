"""
将完整年轮标记图与原图叠加，用彩色标注不同的年轮编号。
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

ANN_DIR = r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\每个年轮完整标记图"
ORIG_DIR = r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\images_no_background"
OUT_DIR  = r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\每个年轮完整标记图\与原图叠加"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ann_files = [f for f in sorted(os.listdir(ANN_DIR)) if f.endswith("_ann.png")]

    for ann_file in tqdm(ann_files, desc="叠加"):
        group = ann_file.replace("_ann.png", "")
        ann_path = os.path.join(ANN_DIR, ann_file)
        orig_path = os.path.join(ORIG_DIR, f"{group}.jpg")

        if not os.path.exists(orig_path):
            tqdm.write(f"  跳过: 找不到原图 {group}.jpg")
            continue

        ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        orig = cv2.imread(orig_path)  # BGR

        if ann.shape[:2] != orig.shape[:2]:
            tqdm.write(f"  跳过: 尺寸不匹配 {group}")
            continue

        # 转成 RGB 方便后面操作
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        # 提取标注区域（非255=年轮像素），用不同颜色标记不同环号
        mask = ann < 255
        ring_values = ann[mask]

        # 用 HSV 色相区分不同年轮，确保相邻环颜色不同
        unique_rings = np.unique(ring_values)
        n_rings = len(unique_rings)

        # 创建半透明彩色叠加层
        overlay = orig_rgb.copy().astype(np.float32)
        alpha = 0.6  # 标注透明度

        # 给每个环分配颜色（HSV → RGB）
        for i, ring_id in enumerate(unique_rings):
            ring_mask = ann == ring_id
            # 色相均匀分布（0~180 for OpenCV）
            hue = int(180 * i / max(n_rings, 1))
            # HSV -> BGR
            color = cv2.cvtColor(
                np.array([[[hue, 255, 255]]], dtype=np.uint8),
                cv2.COLOR_HSV2RGB
            )[0, 0].astype(np.float32)

            overlay[ring_mask] = (1 - alpha) * overlay[ring_mask] + alpha * color

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # 转换回 BGR 保存
        result = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(OUT_DIR, f"{group}_overlay.jpg")
        cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n完成！保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
