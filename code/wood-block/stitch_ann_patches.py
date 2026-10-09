"""
将 teacher code 处理后的 B_ann.png / D_ann.png 拼回原始完整图像大小。
每个 patch 的位置信息从 裁剪矩形/ 目录下的 JSON 文件中读取。
"""

import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 路径
PATCH_ANN_DIR = r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\dataset"
CROP_JSON_DIR = r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形"
ORIGINAL_IMG_DIR = r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\images_no_background"
OUTPUT_DIR = r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\每个年轮完整标记图"


def extract_group_name(patch_dir_name):
    """从 patch 目录名提取原图组名。
    例: T0_B1_N27_B_s1000_n0 -> T0_B1_N27_B
    """
    # 目录名格式: {group}_s{size}_n{num}
    # 从后往前找 _s 和 _n
    parts = patch_dir_name.rsplit('_s', 1)
    if len(parts) == 2:
        group = parts[0]
        return group
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 收集所有 patch → 所属原图组
    print("正在扫描 patch 目录...")
    group_patches = defaultdict(list)  # group_name -> [patch_dir_name, ...]

    for d in os.listdir(PATCH_ANN_DIR):
        dpath = os.path.join(PATCH_ANN_DIR, d)
        if not os.path.isdir(dpath):
            continue
        group = extract_group_name(d)
        if group:
            group_patches[group].append(d)
        else:
            print(f"  [跳过] 无法解析组名: {d}")

    print(f"共找到 {sum(len(v) for v in group_patches.values())} 个 patch，"
          f"分布在 {len(group_patches)} 个原图面上")

    # 2. 对每个原图面，创建画布并拼接
    for group_name, patch_names in tqdm(sorted(group_patches.items()),
                                          desc="拼接原图"):
        # 查找原始图像以获取画布尺寸
        orig_img_path = os.path.join(ORIGINAL_IMG_DIR, f"{group_name}.jpg")
        if os.path.exists(orig_img_path):
            canvas_h, canvas_w = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE).shape
        else:
            # 回退：从 JSON 中推断最大范围
            print(f"  [警告] 找不到原图 {group_name}.jpg，从 JSON 推断画布大小")
            canvas_w = canvas_h = 0
            crop_dir = os.path.join(CROP_JSON_DIR, group_name)
            if os.path.isdir(crop_dir):
                for jf in os.listdir(crop_dir):
                    if not jf.endswith('.json') or jf.endswith('_新髓心.json'):
                        continue
                    with open(os.path.join(crop_dir, jf), 'r') as f:
                        info = json.load(f)
                    canvas_w = max(canvas_w, info['bx'] + info['size'])
                    canvas_h = max(canvas_h, info['by'] + info['size'])
            if canvas_w == 0:
                print(f"  [错误] 无法确定画布大小: {group_name}")
                continue

        # 创建画布（255 = 背景）
        canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)

        placed_count = 0
        for patch_name in patch_names:
            # 读取标注图
            ann_path = os.path.join(PATCH_ANN_DIR, patch_name, "B_ann.png")
            if not os.path.exists(ann_path):
                ann_path = os.path.join(PATCH_ANN_DIR, patch_name, "D_ann.png")
            if not os.path.exists(ann_path):
                continue

            ann_img = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            if ann_img is None:
                continue

            # 读取裁剪位置
            json_path = os.path.join(CROP_JSON_DIR, group_name, f"{patch_name}.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r') as f:
                crop_info = json.load(f)

            bx = crop_info['bx']      # 左上角 x
            by = crop_info['by']      # 左上角 y
            size = crop_info['size']  # patch 边长

            # 放置到画布上（只覆盖非背景像素）
            h, w = ann_img.shape
            crop_h = min(h, size, canvas_h - by)
            crop_w = min(w, size, canvas_w - bx)

            if crop_h <= 0 or crop_w <= 0:
                continue

            patch_data = ann_img[:crop_h, :crop_w]
            canvas_region = canvas[by:by + crop_h, bx:bx + crop_w]

            # 只覆盖 patch 中非背景（非255）的像素
            mask = patch_data < 255
            canvas_region[mask] = patch_data[mask]
            canvas[by:by + crop_h, bx:bx + crop_w] = canvas_region

            placed_count += 1

        # 保存
        out_path = os.path.join(OUTPUT_DIR, f"{group_name}_ann.png")
        cv2.imwrite(out_path, canvas)
        tqdm.write(f"  {group_name}: {placed_count}/{len(patch_names)} patches, "
                   f"画布 {canvas_w}×{canvas_h}")

    print(f"\n完成！输出目录: {OUTPUT_DIR}")
    print(f"共生成 {len(group_patches)} 张完整年轮标记图")


if __name__ == "__main__":
    main()
