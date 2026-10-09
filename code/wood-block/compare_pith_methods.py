"""
树髓定位方法对比评测
========================
对比多个方法在全量 patch_dataset 上的表现

方法:
  A: 独立圆拟合 + 加权中位数 (baseline, 已有)
  B: 线性同心圆回归 d = 2cx·x + 2cy·y + c_i  (新方法)
  C: 非线性同心圆优化 (以B为初值, 梯度下降)

原理: 同心圆共享圆心(cx,cy)
  d = x²+y² = 2cx·x + 2cy·y + (r²-cx²-cy²)
  对每条环分别拟合截距c_i, 但共享斜率(2cx, 2cy)
  相当于在 (x,y,d) 空间中对每条环拟合有各自偏差的平行平面
"""
import json, cv2, numpy as np, pandas as pd, time, warnings, os
from pathlib import Path
from numpy.linalg import lstsq
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
PATCH_ROOT = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
OUT_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\调试验证定位")
OUT_DIR.mkdir(exist_ok=True, parents=True)
DEBUG_DIR = OUT_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True, parents=True)
JSON_FILES = sorted(PATCH_ROOT.glob("*/*.json"))

# 排除 _新髓心.json
ALL_JSONS = [j for j in JSON_FILES if not j.stem.endswith("_新髓心")]
print(f"共 {len(ALL_JSONS)} 个 patch")

# ==================== 方法 A: 独立圆拟合 (baseline) ====================
def fit_circle_robust(points, rng, n_trials=80):
    """RANSAC 圆拟合"""
    if len(points) < 5:
        return None, None, None, 0
    x, y = points[:, 0], points[:, 1]
    best_n, best_cx, best_cy, best_r = 0, None, None, None

    for _ in range(min(n_trials, len(points) * 2)):
        idx = rng.choice(len(points), min(5, len(points)), replace=False)
        sx, sy = points[idx, 0], points[idx, 1]
        A = np.column_stack([2 * sx, 2 * sy, np.ones(len(sx))])
        try:
            sol = lstsq(A, sx ** 2 + sy ** 2, rcond=None)[0]
        except:
            continue
        r = np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1))
        if r <= 0 or r > 50000:
            continue
        n_in = (np.abs(np.sqrt((x - sol[0]) ** 2 + (y - sol[1]) ** 2) - r) < 0.10 * r).sum()
        if n_in > best_n:
            best_n, best_cx, best_cy, best_r = n_in, sol[0], sol[1], r

    if best_cx is None:
        return None, None, None, 0
    inl = np.abs(np.sqrt((x - best_cx) ** 2 + (y - best_cy) ** 2) - best_r) < 0.15 * best_r
    if inl.sum() < 5:
        return best_cx, best_cy, best_r, best_n / len(points)
    xi, yi = x[inl], y[inl]
    A = np.column_stack([2 * xi, 2 * yi, np.ones(len(xi))])
    try:
        sol = lstsq(A, xi ** 2 + yi ** 2, rcond=None)[0]
        r = np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1))
        return sol[0], sol[1], r, best_n / len(points)
    except:
        return best_cx, best_cy, best_r, best_n / len(points)


def method_A(rings, rng):
    """独立圆拟合 + 加权中位数"""
    centers, wts = [], []
    for pts in rings:
        cx, cy, r, inl = fit_circle_robust(pts, rng)
        if cx is not None and r > 0 and inl > 0.3:
            centers.append((cx, cy, r))
            wts.append(inl / r)

    if len(centers) < 2:
        return None, None, 0.0

    wts = np.array(wts)
    wts /= wts.sum()
    cxs = np.array([c[0] for c in centers])
    cys = np.array([c[1] for c in centers])

    s = np.argsort(cxs)
    cs = np.cumsum(wts[s])
    cx = cxs[s[np.searchsorted(cs, 0.5)]]
    s = np.argsort(cys)
    cs = np.cumsum(wts[s])
    cy = cys[s[np.searchsorted(cs, 0.5)]]

    conf = 1.0 / (1.0 + (np.sqrt(np.average((cxs - cx) ** 2, weights=wts)) +
                          np.sqrt(np.average((cys - cy) ** 2, weights=wts))) / 2 /
                  max(abs(cx) + abs(cy), 1))
    return cx, cy, conf


# ==================== 方法 B: 线性同心圆回归 ====================
def method_B(rings, rng):
    """
    线性同心圆回归
    对所有环点联合拟合: x²+y² = 2cx·x + 2cy·y + c_i
    每条环有各自的截距 c_i, 共享 (cx, cy)
    """
    if len(rings) < 2:
        return None, None, 0.0

    n_rings = len(rings)
    point_ring_map = []
    all_x, all_y, all_d = [], [], []

    for ri, pts in enumerate(rings):
        for p in pts:
            all_x.append(p[0])
            all_y.append(p[1])
            all_d.append(p[0] ** 2 + p[1] ** 2)
            point_ring_map.append(ri)

    if len(all_x) < 10:
        return None, None, 0.0

    n_pts = len(all_x)
    # 构建设计矩阵: [2x, 2y, dummy_ring1, dummy_ring2, ...]
    # 每列：cx的系数(2x), cy的系数(2y), 然后每个ring一个独热编码列
    n_cols = 2 + n_rings
    A = np.zeros((n_pts, n_cols))
    A[:, 0] = 2 * np.array(all_x)   # 2x 系数 -> cx
    A[:, 1] = 2 * np.array(all_y)   # 2y 系数 -> cy
    for i, ri in enumerate(point_ring_map):
        A[i, 2 + ri] = 1.0          # 该环的截距

    b = np.array(all_d)

    try:
        sol, residuals, rank, sv = lstsq(A, b, rcond=None)
    except:
        return None, None, 0.0

    cx = sol[0]
    cy = sol[1]

    # 检查合理性
    if abs(cx) > 50000 or abs(cy) > 50000:
        return None, None, 0.0

    # 置信度: 基于残差和环间一致性
    pred = A @ sol
    mae = np.abs(pred - b).mean()
    conf = 1.0 / (1.0 + mae / 1000)

    return cx, cy, conf


# ==================== 方法 C: 非线性同心圆优化 ====================
def method_C(rings, rng, cx_init, cy_init, max_iter=50):
    """
    以线性回归结果为初值, 用梯度下降微调
    最小化: Σ_i Σ_j (||p_ij - (cx,cy)|| - r_i)²
    其中 r_i = mean(||p_ij - (cx,cy)||)
    """
    if cx_init is None or len(rings) < 2:
        return None, None, 0.0

    all_pts = np.vstack(rings)
    ring_ids = np.concatenate([np.full(len(r), i) for i, r in enumerate(rings)])
    n_rings = len(rings)

    cx, cy = cx_init, cy_init
    lr = 0.05

    for it in range(max_iter):
        # 计算每个环的期望半径
        prev_cx, prev_cy = cx, cy
        radii = np.zeros(n_rings)
        for ri in range(n_rings):
            mask = ring_ids == ri
            if mask.sum() > 0:
                dists = np.sqrt((all_pts[mask, 0] - cx) ** 2 + (all_pts[mask, 1] - cy) ** 2)
                radii[ri] = np.median(dists)

        # 计算关于cx, cy的梯度
        grad_cx, grad_cy = 0, 0
        total = 0
        for ri in range(n_rings):
            mask = ring_ids == ri
            if mask.sum() == 0:
                continue
            pts_i = all_pts[mask]
            dists = np.sqrt((pts_i[:, 0] - cx) ** 2 + (pts_i[:, 1] - cy) ** 2)
            # 对每个点，误差 = dist - r_i
            # ∂/∂cx: -(x-cx)/dist * (dist - r_i)
            valid = dists > 1e-6
            if valid.sum() == 0:
                continue
            grad_cx += (-(pts_i[valid, 0] - cx) / dists[valid] * (dists[valid] - radii[ri])).sum()
            grad_cy += (-(pts_i[valid, 1] - cy) / dists[valid] * (dists[valid] - radii[ri])).sum()
            total += valid.sum()

        if total == 0:
            break
        grad_cx /= total
        grad_cy /= total

        # 自适应步长
        gnorm = np.sqrt(grad_cx**2 + grad_cy**2)
        if gnorm < 1e-6:
            break

        # 线搜索
        step = lr
        for _ in range(10):
            new_cx = cx - step * grad_cx / gnorm
            new_cy = cy - step * grad_cy / gnorm
            if abs(new_cx) < 50000 and abs(new_cy) < 50000:
                break
            step *= 0.5

        cx -= step * grad_cx / gnorm
        cy -= step * grad_cy / gnorm

        if np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2) < 0.01:
            break

    # 置信度
    final_pts = all_pts
    final_radii = []
    for ri in range(n_rings):
        mask = ring_ids == ri
        if mask.sum() > 0:
            final_radii.append(np.median(np.sqrt((final_pts[mask, 0] - cx) ** 2 +
                                                  (final_pts[mask, 1] - cy) ** 2)))

    if len(final_radii) < 2:
        return cx, cy, 0.5

    # 检查环间一致性
    cv_radii = np.std(final_radii) / max(np.mean(final_radii), 1)
    conf = 1.0 / (1.0 + cv_radii * 10)

    return cx, cy, conf


# ==================== 全量评测 ====================
rng = np.random.RandomState(42)
results = []

print(f"\n开始全量评测...")

for idx, jf in enumerate(ALL_JSONS):
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)

    rings = [np.array(r['points']) for r in d['rings']]
    if len(rings) < 2:
        continue

    gt_cx, gt_cy = d['cx_local'], d['cy_local']
    patch_name = d['patch_name']

    # 方法 A
    t0 = time.time()
    cx_a, cy_a, conf_a = method_A(rings, rng)
    t_a = time.time() - t0

    # 方法 B
    t0 = time.time()
    cx_b, cy_b, conf_b = method_B(rings, rng)
    t_b = time.time() - t0

    # 方法 C (以 B 为初值)
    t0 = time.time()
    if cx_b is not None:
        cx_c, cy_c, conf_c = method_C(rings, rng, cx_b, cy_b)
    else:
        cx_c, cy_c, conf_c = None, None, 0.0
    t_c = time.time() - t0

    def err(cx, cy):
        return np.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2) if cx is not None else np.nan

    results.append({
        'patch_name': patch_name,
        'sample': d['sample'],
        'size': d['size'],
        'gt_cx': gt_cx, 'gt_cy': gt_cy,
        'pith_in_patch': d['pith_in_patch'],
        'dist_to_pith': d['dist_to_pith'],
        'angle_span_deg': d['angle_span_deg'],
        'n_rings': len(rings),
        'n_ring_points': d['n_ring_points'],

        # 方法 A
        'A_cx': cx_a, 'A_cy': cy_a, 'A_err': err(cx_a, cy_a), 'A_conf': conf_a, 'A_time': t_a,
        # 方法 B
        'B_cx': cx_b, 'B_cy': cy_b, 'B_err': err(cx_b, cy_b), 'B_conf': conf_b, 'B_time': t_b,
        # 方法 C
        'C_cx': cx_c, 'C_cy': cy_c, 'C_err': err(cx_c, cy_c), 'C_conf': conf_c, 'C_time': t_c,
    })

    if (idx + 1) % 500 == 0:
        df_temp = pd.DataFrame(results)
        print(f"  进度: {idx+1}/{len(ALL_JSONS)} "
              f"| A中位数={df_temp['A_err'].median():.0f} "
              f"| B中位数={df_temp['B_err'].median():.0f} "
              f"| C中位数={df_temp['C_err'].median():.0f}")

# ==================== 汇总 ====================
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "comparison_all.csv", index=False)

def summarize(name, col_err, df):
    valid = df[col_err].notna()
    errs = df.loc[valid, col_err]
    return {
        'method': name,
        'valid': valid.sum(),
        'median_err': np.median(errs),
        'mean_err': np.mean(errs),
        'pct_lt_50': 100 * (errs < 50).sum() / len(errs),
        'pct_lt_100': 100 * (errs < 100).sum() / len(errs),
        'pct_lt_200': 100 * (errs < 200).sum() / len(errs),
        'pct_lt_500': 100 * (errs < 500).sum() / len(errs),
        'pct_lt_1000': 100 * (errs < 1000).sum() / len(errs),
    }

summary_rows = []
for method, col in [('A_独立圆拟合+加权中位数', 'A_err'),
                     ('B_线性同心圆回归', 'B_err'),
                     ('C_非线性同心圆优化', 'C_err')]:
    summary_rows.append(summarize(method, col, df))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "comparison_summary.csv", index=False)

print(f"\n{'='*80}")
print(f"{'方法':<30} {'有效':<6} {'中位数':<8} {'均值':<8} {'<50':<8} {'<100':<8} {'<200':<8} {'<500':<8} {'<1000':<8}")
print('-'*80)
for s in summary_rows:
    print(f"{s['method']:<30} {s['valid']:<6} {s['median_err']:<8.0f} {s['mean_err']:<8.0f} "
          f"{s['pct_lt_50']:<8.0f} {s['pct_lt_100']:<8.0f} {s['pct_lt_200']:<8.0f} {s['pct_lt_500']:<8.0f} {s['pct_lt_1000']:<8.0f}")

# ==================== 分组对比 ====================
print(f"\n{'='*80}")
print(f"分组对比 (中位数误差, px)")
group_defs = [
    ('pith_IN', df['pith_in_patch'] == True),
    ('pith_OUT', df['pith_in_patch'] == False),
    ('dist<500', df['dist_to_pith'] < 500),
    ('dist_500-1500', (df['dist_to_pith'] >= 500) & (df['dist_to_pith'] < 1500)),
    ('dist>1500', df['dist_to_pith'] >= 1500),
    ('ang<45', df['angle_span_deg'] < 45),
    ('ang_45-90', (df['angle_span_deg'] >= 45) & (df['angle_span_deg'] < 90)),
    ('ang>90', df['angle_span_deg'] >= 90),
]
print(f"{'分组':<16} {'数量':<6} {'A中位数':<10} {'B中位数':<10} {'C中位数':<10} {'改善%':<8}")
print('-'*60)
for label, cond in group_defs:
    sub = df[cond]
    if len(sub) > 0:
        a_m = sub['A_err'].median()
        c_m = sub['C_err'].median()
        imp = 100 * (1 - c_m / max(a_m, 1))
        print(f"{label:<16} {len(sub):<6} {a_m:<10.0f} {sub['B_err'].median():<10.0f} {c_m:<10.0f} {imp:<8.0f}")

# ==================== 生成 TOP 改善/恶化样本 ====================
print(f"\n{'='*80}")
print(f"生成调试可视化...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TOP 改善: A失败但C成功
df['improvement'] = df['A_err'] - df['C_err']
df_valid = df[df['A_err'].notna() & df['C_err'].notna()].copy()

top_improved = df_valid.nlargest(30, 'improvement')
top_worsened = df_valid.nsmallest(30, 'improvement')
worst_both = df_valid.nlargest(30, 'C_err')

def plot_comparison(row, out_path, title_prefix):
    """生成对比图：显示 ring arcs + 三种方法的估计"""
    sample = row['sample']
    patch_name = row['patch_name']
    img_path = PATCH_ROOT / sample / f"{patch_name}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        return

    # 读取 ring 数据
    jf = PATCH_ROOT / sample / f"{patch_name}.json"
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)
    rings = [np.array(r['points']) for r in d['rings']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax in axes:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # rings
        for ring in rings:
            ax.plot(ring[:, 0], ring[:, 1], 'cyan', lw=0.3, alpha=0.6)

        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

    # GT
    axes[0].plot(row['gt_cx'], row['gt_cy'], 'g*', markersize=18, label=f'GT ({row["gt_cx"]:.0f},{row["gt_cy"]:.0f})')
    axes[1].plot(row['gt_cx'], row['gt_cy'], 'g*', markersize=18, label=f'GT')

    # Method A
    if pd.notna(row['A_cx']):
        axes[0].plot(row['A_cx'], row['A_cy'], 'ro', markersize=10, label=f'A(err={row["A_err"]:.0f})')

    # Method B
    if pd.notna(row['B_cx']):
        mid_x = (row['A_cx'] + row['C_cx']) / 2 if pd.notna(row['A_cx']) else row['gt_cx']
        axes[1].plot(row['B_cx'], row['B_cy'], 'bs', markersize=8, label=f'B(err={row["B_err"]:.0f})')

    # Method C
    if pd.notna(row['C_cx']):
        axes[1].plot(row['C_cx'], row['C_cy'], 'm^', markersize=10, label=f'C(err={row["C_err"]:.0f})')

    for ax in axes:
        ax.legend(fontsize=8)

    fig.suptitle(f'{title_prefix} {patch_name} | angle={row["angle_span_deg"]:.0f}° dist={row["dist_to_pith"]:.0f}',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)

for i, (_, row) in enumerate(top_improved.head(10).iterrows()):
    out = DEBUG_DIR / f"improved_{i:02d}_{row['patch_name']}.png"
    plot_comparison(row, out, f'改善 #{i+1}')

for i, (_, row) in enumerate(top_worsened.head(10).iterrows()):
    out = DEBUG_DIR / f"worsened_{i:02d}_{row['patch_name']}.png"
    plot_comparison(row, out, f'恶化 #{i+1}')

for i, (_, row) in enumerate(worst_both.head(10).iterrows()):
    out = DEBUG_DIR / f"worst_{i:02d}_{row['patch_name']}.png"
    plot_comparison(row, out, f'最差 #{i+1}')

print(f"调试图像: {DEBUG_DIR} ({len(list(DEBUG_DIR.glob('*.png')))} 张)")

print(f"\n{'='*80}")
print(f"输出文件:")
print(f"  全量结果:   {OUT_DIR / 'comparison_all.csv'}")
print(f"  汇总对比:   {OUT_DIR / 'comparison_summary.csv'}")
print(f"  调试图像:   {DEBUG_DIR}/")
print(f"\n完成!")
