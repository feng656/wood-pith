"""
从 UruDendro4 生成局部年轮小块数据集
======================================
每块 = 从完整年轮盘随机裁出的矩形
含 ground truth：髓心在块内局部坐标
"""
import json, cv2, numpy as np, os, pandas as pd
from pathlib import Path
import random

DATA = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4")
OUT = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
OUT.mkdir(exist_ok=True, parents=True)

# 读取树髓坐标
pith_df = pd.read_csv(DATA / "pith_location.txt")

# 参数
PATCH_SIZES = [400, 600, 800, 1000, 1200]  # 块大小
PATCHES_PER_SIZE = 8                        # 每个大小生成几个块
MIN_RING_POINTS = 30                        # 块内最少年轮点数


def generate_patches(sample_code, cx, cy, img, rings, output_dir):
    """为单个样本生成多个随机小块"""
    H, W = img.shape[:2]
    results = []

    for size in PATCH_SIZES:
        for i in range(PATCHES_PER_SIZE):
            # 随机位置：偏向髓心周围和远离髓心都生成
            if random.random() < 0.4:
                # 髓心附近
                bx = int(cx + random.uniform(-size, size))
                by = int(cy + random.uniform(-size, size))
            else:
                # 全图随机
                bx = int(random.uniform(0, W - size))
                by = int(random.uniform(0, H - size))

            bx = max(0, min(W - size, bx))
            by = max(0, min(H - size, by))

            # 裁块
            patch = img[by:by+size, bx:bx+size]
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            # 跳过空白块
            if (patch_gray > 245).mean() > 0.9:
                continue

            # 块内髓心
            cx_local = cx - bx
            cy_local = cy - by

            # 块内年轮点
            local_rings = []
            for ring_pts in rings:
                local_pts = ring_pts - np.array([bx, by])
                valid = (local_pts[:, 0] >= 0) & (local_pts[:, 0] < size) & \
                        (local_pts[:, 1] >= 0) & (local_pts[:, 1] < size)
                if valid.sum() >= 5:
                    local_rings.append(local_pts[valid])

            total_ring_points = sum(len(r) for r in local_rings)
            if total_ring_points < MIN_RING_POINTS:
                continue

            # 计算指标
            pith_in_patch = 0 <= cx_local < size and 0 <= cy_local < size
            dist_to_pith = np.sqrt((cx_local - size/2)**2 + (cy_local - size/2)**2)
            angle_span = 0
            if len(local_rings) > 0:
                all_pts = np.vstack(local_rings)
                angles = np.arctan2(all_pts[:,1] - cy_local, all_pts[:,0] - cx_local)
                angles = angles % (2*np.pi)
                if len(angles) > 1:
                    angles_s = np.sort(angles)
                    gaps = np.diff(np.append(angles_s, angles_s[0] + 2*np.pi))
                    max_gap = gaps.max()
                    angle_span = 2*np.pi - max_gap

            patch_name = f"{sample_code}_s{size}_n{i}"
            cv2.imwrite(str(output_dir / f"{patch_name}.jpg"), patch)

            # 保存每块的标注文件
            ring_list = []
            for ri, rpts in enumerate(local_rings):
                ring_list.append({
                    'ring_idx': ri,
                    'n_points': len(rpts),
                    'points': rpts.astype(int).tolist()
                })

            annotation = {
                'patch_name': patch_name,
                'sample': sample_code,
                'size': size,
                'bx': bx, 'by': by,
                'cx_global': cx, 'cy_global': cy,
                'cx_local': cx_local, 'cy_local': cy_local,
                'pith_in_patch': pith_in_patch,
                'dist_to_pith': round(dist_to_pith, 2),
                'angle_span_rad': round(angle_span, 4),
                'angle_span_deg': round(np.degrees(angle_span), 1),
                'n_ring_arcs': len(local_rings),
                'n_ring_points': total_ring_points,
                'img_W': size, 'img_H': size,
                'rings': ring_list
            }
            with open(str(output_dir / f"{patch_name}.json"), 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)

            results.append({
                'patch_name': patch_name,
                'sample': sample_code,
                'size': size,
                'bx': bx, 'by': by,
                'cx_gt': cx_local,
                'cy_gt': cy_local,
                'pith_in_patch': pith_in_patch,
                'dist_to_pith': dist_to_pith,
                'angle_span_rad': angle_span,
                'n_ring_arcs': len(local_rings),
                'n_ring_points': total_ring_points,
                'img_W': size, 'img_H': size
            })

    return results


def main():
    all_results = []
    random.seed(42)

    ann_dir = DATA / "annotations" / "annual_rings"
    img_dir = DATA / "images_no_background"

    samples = []
    for _, row in pith_df.iterrows():
        code = row['Code']
        jf = ann_dir / f"{code}.json"
        imf = img_dir / f"{code}.jpg"
        if jf.exists() and imf.exists():
            samples.append((code, int(row['cx']), int(row['cy']), jf, imf))

    print(f"共 {len(samples)} 个样本")

    for idx, (code, cx, cy, jf, imf) in enumerate(samples):
        # 加载
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img = cv2.imread(str(imf))

        # 提取所有年轮点（所有shape都是年轮，label是序号）
        rings = []
        for shape in data.get('shapes', []):
            pts = np.array(shape['points'])
            if len(pts) >= 5:
                rings.append(pts)

        if len(rings) < 3:
            continue

        # 生成小块
        sample_out = OUT / code
        sample_out.mkdir(exist_ok=True)

        results = generate_patches(code, cx, cy, img, rings, sample_out)
        for r in results:
            r['sample_id'] = idx
        all_results.extend(results)

        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx+1}/{len(samples)}, 已生成 {len(all_results)} 块")

    # 保存
    df = pd.DataFrame(all_results)
    df.to_csv(OUT.parent / "dataset（元数据）.csv", index=False)

    print(f"\n{'='*60}")
    print(f"数据集生成完毕!")
    print(f"  样本数: {len(samples)}")
    print(f"  总块数: {len(all_results)}")
    print(f"  髓心在块内: {df['pith_in_patch'].sum()} 块")
    print(f"  髓心在块外: {(~df['pith_in_patch']).sum()} 块")
    print(f"  平均年轮弧段: {df['n_ring_arcs'].mean():.1f}")
    print(f"  平均年轮点数: {df['n_ring_points'].mean():.0f}")
    print(f"  距离髓心中位数: {df['dist_to_pith'].median():.0f}px")
    print(f"  角度跨度中位数: {np.degrees(df['angle_span_rad'].median()):.0f}deg")
    print(f"\n  裁剪图片: {OUT}")
    print(f"  CSV:      {OUT.parent / 'dataset.csv'}")


if __name__ == "__main__":
    main()
