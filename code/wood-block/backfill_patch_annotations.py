"""
给已有 patch 批量补上标注 JSON
==============================
读取 dataset.csv 和原始标注，为裁剪矩形目录里每个已有的 JPG patch
生成同名的 .json 标注文件，包含：
  - 局部年轮点坐标 (rings)
  - 局部髓心坐标 (cx_local, cy_local)
  - 全局髓心坐标 (cx_global, cy_global)
  - 裁剪偏移 (bx, by)
  - 角度跨度、距离髓心等指标
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ==================== 路径配置 ====================
PATCH_ROOT = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
CSV_PATH = Path(r"D:\教务处实习\木材数据1\patch_dataset\dataset（元数据）.csv")
ANN_DIR = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\annotations\annual_rings")
PITH_CSV = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\pith_location.txt")

# ==================== 加载数据 ====================
df = pd.read_csv(CSV_PATH)
pith_df = pd.read_csv(PITH_CSV)
pith_map = dict(zip(pith_df['Code'], zip(pith_df['cx'], pith_df['cy'])))

print(f"数据集: {len(df)} 块")
print(f"标注样本: {len(pith_map)} 个")

# ==================== 统计 ====================
new_count = 0
skip_count = 0
no_jpg_count = 0
no_ann_count = 0

for idx, row in df.iterrows():
    patch_name = row['patch_name']
    sample = row['sample']
    size = int(row['size'])
    bx = int(row['bx'])
    by = int(row['by'])

    # 图片路径
    jpg_path = PATCH_ROOT / sample / f"{patch_name}.jpg"
    json_path = PATCH_ROOT / sample / f"{patch_name}.json"

    # 图片不存在则跳过
    if not jpg_path.exists():
        no_jpg_count += 1
        continue

    # JSON 已存在则跳过
    if json_path.exists():
        skip_count += 1
        continue

    # 加载原始标注
    ann_path = ANN_DIR / f"{sample}.json"
    if not ann_path.exists():
        no_ann_count += 1
        continue

    with open(ann_path, 'r', encoding='utf-8') as f:
        ann_data = json.load(f)

    # 提取年轮点并转换到局部坐标
    ring_list = []
    total_points = 0
    for ri, shape in enumerate(ann_data.get('shapes', [])):
        pts = np.array(shape['points'])
        pts_local = pts - np.array([bx, by])
        valid = (pts_local[:, 0] >= 0) & (pts_local[:, 0] < size) & \
                (pts_local[:, 1] >= 0) & (pts_local[:, 1] < size)
        n_valid = int(valid.sum())
        if n_valid >= 5:
            valid_pts = pts_local[valid].astype(int)
            # 确保所有值转为原生 Python int
            ring_list.append({
                'ring_idx': int(ri),
                'n_points': n_valid,
                'points': [[int(x), int(y)] for x, y in valid_pts]
            })
            total_points += n_valid

    if len(ring_list) < 2:
        continue

    # 髓心坐标
    cx_global, cy_global = pith_map.get(sample, (None, None))
    if cx_global is None:
        continue

    cx_local = cx_global - bx
    cy_local = cy_global - by
    pith_in_patch = 0 <= cx_local < size and 0 <= cy_local < size
    dist_to_pith = np.sqrt((cx_local - size/2)**2 + (cy_local - size/2)**2)

    # 角度跨度
    angle_span = 0
    if len(ring_list) > 0:
        all_pts_list = [np.array(r['points']) for r in ring_list]
        all_pts = np.vstack(all_pts_list)
        angles = np.arctan2(all_pts[:,1] - cy_local, all_pts[:,0] - cx_local)
        angles = angles % (2*np.pi)
        if len(angles) > 1:
            angles_s = np.sort(angles)
            gaps = np.diff(np.append(angles_s, angles_s[0] + 2*np.pi))
            angle_span = 2*np.pi - gaps.max()

    # 构建标注
    annotation = {
        'patch_name': patch_name,
        'sample': sample,
        'size': size,
        'bx': bx, 'by': by,
        'cx_global': int(cx_global), 'cy_global': int(cy_global),
        'cx_local': int(cx_local), 'cy_local': int(cy_local),
        'pith_in_patch': bool(pith_in_patch),
        'dist_to_pith': round(float(dist_to_pith), 2),
        'angle_span_rad': round(float(angle_span), 4),
        'angle_span_deg': round(float(np.degrees(angle_span)), 1),
        'n_ring_arcs': int(len(ring_list)),
        'n_ring_points': int(total_points),
        'img_W': size, 'img_H': size,
        'rings': ring_list
    }

    # 写入
    with open(str(json_path), 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    new_count += 1

    if new_count % 500 == 0:
        print(f"  进度: {new_count} 已生成...")

# ==================== 报告 ====================
print(f"\n{'='*60}")
print(f"补标注完成！")
print(f"  新生成 JSON: {new_count}")
print(f"  已存在跳过:  {skip_count}")
print(f"  图片缺失:    {no_jpg_count}")
print(f"  标注缺失:    {no_ann_count}")
print(f"\n  标注位置:    {PATCH_ROOT}")
print(f"  每块现在有:  .jpg 图片 + .json 标注")
