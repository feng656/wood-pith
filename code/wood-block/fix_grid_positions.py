"""
为网格 patch 创建对应的位置 JSON（在 裁剪矩形/ 目录下），
只包含 stitch 需要的字段：bx, by, size
"""

import os, json, shutil
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\dataset")
CROP_DIR    = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")

PATCH_SIZE = 1000
STRIDE     = 500

# 清理旧的 B/D 面数据，只保留非 B/D 面（A, C, ADAP）
for group_dir in sorted(CROP_DIR.iterdir()):
    if not group_dir.is_dir():
        continue
    parts = group_dir.name.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in ('B', 'D'):
        shutil.rmtree(group_dir)

# 扫描 dataset 中的网格 patch，重建位置 JSON
patch_dirs = sorted([d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(DATASET_DIR / d)])

# 按 sample 分组
sample_patches = {}
for pd_name in patch_dirs:
    parts = pd_name.rsplit('_s', 1)
    if len(parts) != 2:
        continue
    sample = parts[0]
    if sample not in sample_patches:
        sample_patches[sample] = []
    sample_patches[sample].append(pd_name)

total = 0
for sample, patches in tqdm(sorted(sample_patches.items()), desc="生成位置JSON"):
    # 创建 sample 目录
    sample_dir = CROP_DIR / sample
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 网格位置与 regenerate_full_coverage.py 一致
    # 需要知道原图尺寸来确定网格范围
    # 从 patch 名称中的 n 序号反推位置
    # 网格: for y in range(0, H-PATCH_SIZE+1, STRIDE):
    #        for x in range(0, W-PATCH_SIZE+1, STRIDE):
    # 序号 n 是在这些位置中逐个递增的

    # 我们需要原图尺寸。从原图读取。
    import cv2
    orig_path = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\images_no_background") / f"{sample}.jpg"
    img = cv2.imread(str(orig_path))
    if img is None:
        tqdm.write(f"  [跳过] 找不到: {orig_path}")
        continue
    H, W = img.shape[:2]

    # 重新计算网格
    face = sample.rsplit('_', 1)[1]  # B or D
    positions = []
    n = 0
    for y in range(0, max(1, H - PATCH_SIZE + 1), STRIDE):
        for x in range(0, max(1, W - PATCH_SIZE + 1), STRIDE):
            x2 = min(x + PATCH_SIZE, W)
            y2 = min(y + PATCH_SIZE, H)
            x1 = max(0, x2 - PATCH_SIZE)
            y1 = max(0, y2 - PATCH_SIZE)

            patch_name = f"{sample}_s{PATCH_SIZE}_n{n}"
            if patch_name in patches:
                positions.append((patch_name, x1, y1, PATCH_SIZE))
            n += 1

    for patch_name, bx, by, size in positions:
        json_path = sample_dir / f"{patch_name}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'patch_name': patch_name,
                'sample': sample,
                'size': size,
                'bx': bx,
                'by': by,
            }, f)
        total += 1

print(f"\n完成！共写入 {total} 个位置 JSON")
print("现在可以重新运行: python stitch_ann_patches.py")
