"""
直接从原始 LabelMe 标注生成 B_ann.png / D_ann.png，确保与原始标注完全一致。

流程：
  原始 LabelMe JSON → 全图渲染 → 按 patch 位置裁剪 → B_ann.png / D_ann.png
  （跳过 patch JSON 的 ring 数据，消除重新索引和边界截断问题）
"""

import os, json, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 路径
ORIG_ANN_DIR = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\annotations\annual_rings")
ORIG_IMG_DIR = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\images_no_background")
PATCH_JSON_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
DATASET_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\dataset")
TRAIN_FILE  = Path(r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\train.txt")

# 环标签保留原始编号（不改动，直接从 LabelMe 的 label 来）
# LabelMe 中 label 是字符串数字如 "0", "1", "2"...

def render_original_annotation(ann_json_path, img_width, img_height):
    """渲染原始 LabelMe 标注到全图大小画布。
    返回: (canvas, ring_label_map)
      canvas: 灰度图，像素值=环标签值(1-based), 255=背景
    """
    with open(ann_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    canvas = np.full((img_height, img_width), 255, dtype=np.uint8)

    for shape in data.get('shapes', []):
        pts = np.array([[p[0], p[1]] for p in shape['points']], dtype=np.int32)
        if len(pts) < 2:
            continue
        # 保留原始环编号 (label 可能是字符串)
        label = int(shape['label']) + 1  # 1-based，与 convert_dataset.py 一致
        cv2.polylines(canvas, [pts], isClosed=False,
                      color=label, thickness=3)

    return canvas


def main():
    # 收集所有 dataset 中的 patch 目录
    patch_dirs = [d for d in os.listdir(DATASET_DIR)
                  if os.path.isdir(DATASET_DIR / d)]

    # 按原始样本分组
    sample_patches = defaultdict(list)
    for pd_name in patch_dirs:
        # 解析: T0_B1_N27_B_s1000_n0 → sample=T0_B1_N27_B
        parts = pd_name.rsplit('_s', 1)
        if len(parts) != 2:
            continue
        sample = parts[0]
        sample_patches[sample].append(pd_name)

    print(f"共 {len(patch_dirs)} 个 patch，分布在 {len(sample_patches)} 个原始样本中")

    # 缓存渲染好的全图标注画布（避免重复渲染）
    rendered_cache = {}

    train_ids = []
    updated_count = 0

    for sample, patches in tqdm(sorted(sample_patches.items()), desc="处理样本"):
        ann_json = ORIG_ANN_DIR / f"{sample}.json"
        orig_img = ORIG_IMG_DIR / f"{sample}.jpg"

        if not ann_json.exists():
            tqdm.write(f"  [跳过] 找不到标注: {ann_json}")
            continue
        if not orig_img.exists():
            tqdm.write(f"  [跳过] 找不到原图: {orig_img}")
            continue

        # 获取原图尺寸
        img = cv2.imread(str(orig_img))
        H, W = img.shape[:2]

        # 渲染全图标注（缓存）
        if sample not in rendered_cache:
            rendered_cache[sample] = render_original_annotation(ann_json, W, H)
        full_canvas = rendered_cache[sample]

        for patch_name in patches:
            # 读取 patch JSON 获取裁剪位置
            patch_json = PATCH_JSON_DIR / sample / f"{patch_name}.json"
            if not patch_json.exists():
                tqdm.write(f"  [跳过] 找不到 patch JSON: {patch_json}")
                continue

            with open(patch_json, 'r', encoding='utf-8') as f:
                pinfo = json.load(f)

            bx = pinfo['bx']
            by = pinfo['by']
            size = pinfo['size']

            # 从全图标注裁剪
            crop = full_canvas[by:by + size, bx:bx + size]

            # 确定是 B 面还是 D 面
            face = patch_name.split('_')[3]  # T0_B1_N27_B_s1000_n0 → B
            ann_filename = f"{face}_ann.png"

            out_path = DATASET_DIR / patch_name / ann_filename
            cv2.imwrite(str(out_path), crop)
            train_ids.append(patch_name)
            updated_count += 1

    # 更新 train.txt
    train_ids = sorted(set(train_ids))
    with open(TRAIN_FILE, 'w') as f:
        for tid in train_ids:
            f.write(tid + '\n')

    print(f"\n✅ 完成!")
    print(f"   更新了 {updated_count} 个 B_ann.png / D_ann.png")
    print(f"   train.txt: {len(train_ids)} 个样本")
    print(f"\n   现在需要重新运行拼接: python stitch_ann_patches.py")
    print(f"   然后重新运行叠加:   python overlay_ann_originals.py")


if __name__ == "__main__":
    main()
