"""
生成训练集 txt：将 dataset 灰度图路径 与 对应髓心坐标 配对。
优先匹配 矩形新的髓心坐标/ 中的坐标，匹配不到则用 pith_location.txt 计算。
"""

import os, json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATASET_DIR  = Path(r"D:\教务处实习\木材数据1\patch_dataset\teacher code output\处理为符合代码期望\dataset")
PITH_JSON_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\矩形新的髓心坐标")
PITH_TXT     = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\pith_location.txt")
CROP_DIR     = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
OUT_TRAIN    = Path(r"D:\教务处实习\木材数据1\patch_dataset\train_pith.txt")


def main():
    # 读取全局髓心坐标（备用）
    pith_df = pd.read_csv(PITH_TXT)
    global_pith = {}
    for _, row in pith_df.iterrows():
        global_pith[row['Code']] = (int(row['cx']), int(row['cy']))

    # 扫描 dataset 目录
    patch_dirs = sorted([d for d in os.listdir(DATASET_DIR)
                          if os.path.isdir(DATASET_DIR / d)])

    lines = []
    matched_json = 0
    computed = 0
    skipped = 0

    for patch_name in tqdm(patch_dirs, desc="匹配坐标"):
        # 格式: T0_B1_N27_B_s1000_n0
        parts = patch_name.rsplit('_s', 1)
        if len(parts) != 2:
            skipped += 1
            continue
        sample = parts[0]
        face = sample.rsplit('_', 1)[-1]

        # 找到对应的 ann 图路径（相对于 dataset 的路径）
        ann_file = DATASET_DIR / patch_name / f"{face}_ann.png"
        if not ann_file.exists():
            skipped += 1
            continue
        rel_ann_path = ann_file.relative_to(DATASET_DIR.parent)

        # 从 crop JSON 读取裁剪位置（网格 patch 必须有这个）
        crop_json = CROP_DIR / sample / f"{patch_name}.json"
        if not crop_json.exists():
            skipped += 1
            continue
        with open(crop_json, 'r', encoding='utf-8') as f:
            crop_info = json.load(f)
        bx = crop_info['bx']
        by = crop_info['by']

        # 方法1：用 矩形新的髓心坐标，但要验证位置一致
        pith_json = PITH_JSON_DIR / f"{patch_name}_新髓心.json"
        use_json = False
        if pith_json.exists():
            with open(pith_json, 'r', encoding='utf-8') as f:
                pinfo = json.load(f)
            # 验证裁剪起点一致（避免旧patch名匹配到新网格位置）
            if pinfo.get('裁剪起点', [None, None]) == [bx, by]:
                cx_local = pinfo['新髓心坐标'][0]
                cy_local = pinfo['新髓心坐标'][1]
                use_json = True
                matched_json += 1

        if not use_json:
            # 方法2：从 pith_location.txt + 裁剪起点 计算
            if sample not in global_pith:
                skipped += 1
                continue
            cx_global, cy_global = global_pith[sample]
            cx_local = cx_global - bx
            cy_local = cy_global - by
            computed += 1

        lines.append(f"{rel_ann_path} {cx_local:.1f} {cy_local:.1f}")

    # 写入
    with open(OUT_TRAIN, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    total = len(lines)
    pith_in = sum(1 for l in lines if float(l.split()[1]) >= 0 and float(l.split()[2]) >= 0)
    print(f"\n✅ 完成: {OUT_TRAIN}")
    print(f"   总样本: {total}")
    print(f"   从新髓心JSON匹配: {matched_json}")
    print(f"   从pith_location计算: {computed}")
    print(f"   跳过: {skipped}")
    print(f"   格式: 相对路径 cx_local cy_local")
    if lines:
        print(f"   示例: {lines[0]}")


if __name__ == "__main__":
    main()
