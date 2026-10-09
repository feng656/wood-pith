"""
混合树髓定位 + 全量评测
======================
A擅长: 髓心在块内/大角度 (中位数8px)
B擅长: 髓心远/小角度 (改善30%)

混合策略:
  - H1: 若A置信度>0.5 且 A估计在patch内 → 用A, 否则用B
  - H2: 置信度加权平均 A*conf_A + B*conf_B
  - H3: 若A的环间一致性高 → 用A, 否则用B
"""
import numpy as np, pandas as pd
from pathlib import Path

OUT_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\调试验证定位")
df = pd.read_csv(OUT_DIR / "comparison_all.csv")
print(f"加载 {len(df)} 条评测记录")

# ==================== 混合策略 ====================

# H1: A置信度高且A估计在图像范围内 → 用A; 否则用B
mask_h1 = (df['A_conf'] > 0.5) & \
          (df['A_cx'].notna()) & \
          (df['A_cx'] > -df['size'] * 2) & (df['A_cx'] < df['size'] * 3) & \
          (df['A_cy'] > -df['size'] * 2) & (df['A_cy'] < df['size'] * 3)
df['H1_cx'] = np.where(mask_h1, df['A_cx'], df['B_cx'])
df['H1_cy'] = np.where(mask_h1, df['A_cy'], df['B_cy'])
df['H1_err'] = np.sqrt((df['H1_cx'] - df['gt_cx'])**2 + (df['H1_cy'] - df['gt_cy'])**2)
print(f"  H1: {mask_h1.sum()} 块用A, {(~mask_h1).sum()} 块用B")

# H2: 置信度加权平均
w_a = df['A_conf'].fillna(0)
w_b = df['B_conf'].fillna(0)
w_sum = w_a + w_b + 1e-10
df['H2_cx'] = (df['A_cx'].fillna(0) * w_a + df['B_cx'].fillna(0) * w_b) / w_sum
df['H2_cy'] = (df['A_cy'].fillna(0) * w_a + df['B_cy'].fillna(0) * w_b) / w_sum
df['H2_err'] = np.sqrt((df['H2_cx'] - df['gt_cx'])**2 + (df['H2_cy'] - df['gt_cy'])**2)

# H3: 若A的估计在patch内(髓心可见) → A; 否则B
mask_h3 = (df['A_cx'] > 0) & (df['A_cx'] < df['size']) & \
          (df['A_cy'] > 0) & (df['A_cy'] < df['size'])
df['H3_cx'] = np.where(mask_h3, df['A_cx'], df['B_cx'])
df['H3_cy'] = np.where(mask_h3, df['A_cy'], df['B_cy'])
df['H3_err'] = np.sqrt((df['H3_cx'] - df['gt_cx'])**2 + (df['H3_cy'] - df['gt_cy'])**2)
print(f"  H3: {mask_h3.sum()} 块用A, {(~mask_h3).sum()} 块用B")

# ==================== 汇总 ====================
def summarize(name, col_err, df):
    valid = df[col_err].notna()
    errs = df.loc[valid, col_err]
    return {
        'method': name,
        'valid': int(valid.sum()),
        'median': np.median(errs),
        'mean': np.mean(errs),
        'pct<50': 100*(errs<50).sum()/len(errs),
        'pct<100': 100*(errs<100).sum()/len(errs),
        'pct<200': 100*(errs<200).sum()/len(errs),
        'pct<500': 100*(errs<500).sum()/len(errs),
        'pct<1000': 100*(errs<1000).sum()/len(errs),
    }

methods = [
    ('A_独立圆拟合', 'A_err'),
    ('B_线性同心回归', 'B_err'),
    ('C_非线性优化', 'C_err'),
    ('H1_置信度选择', 'H1_err'),
    ('H2_加权平均', 'H2_err'),
    ('H3_块内选A', 'H3_err'),
]

summary_rows = [summarize(n, c, df) for n, c in methods]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "hybrid_summary.csv", index=False)

print(f"\n{'='*80}")
print(f"{'方法':<20} {'中位':<8} {'均值':<8} {'<50':<7} {'<100':<7} {'<200':<7} {'<500':<7} {'<1000':<7}")
print('-'*75)
for s in summary_rows:
    print(f"{s['method']:<20} {s['median']:<8.0f} {s['mean']:<8.0f} "
          f"{s['pct<50']:<7.0f} {s['pct<100']:<7.0f} {s['pct<200']:<7.0f} {s['pct<500']:<7.0f} {s['pct<1000']:<7.0f}")

# ==================== 分组对比 ====================
group_defs = [
    ('pith_IN', df['pith_in_patch']==True),
    ('pith_OUT', df['pith_in_patch']==False),
    ('dist<500', df['dist_to_pith']<500),
    ('dist_500-1500', (df['dist_to_pith']>=500)&(df['dist_to_pith']<1500)),
    ('dist>1500', df['dist_to_pith']>=1500),
    ('ang<45', df['angle_span_deg']<45),
    ('ang_45-90', (df['angle_span_deg']>=45)&(df['angle_span_deg']<90)),
    ('ang>90', df['angle_span_deg']>=90),
]

print(f"\n{'='*100}")
print(f"分组中位数对比")
print(f"{'分组':<16} {'N':<6} {'A':<8} {'B':<8} {'C':<8} {'H1':<8} {'H2':<8} {'H3':<8} {'最优':<6}")
print('-'*70)
best_counts = {'A': 0, 'B': 0, 'C': 0, 'H1': 0, 'H2': 0, 'H3': 0}
for label, cond in group_defs:
    sub = df[cond]
    if len(sub) == 0: continue
    scores = {
        'A': sub['A_err'].median(),
        'B': sub['B_err'].median(),
        'C': sub['C_err'].median(),
        'H1': sub['H1_err'].median(),
        'H2': sub['H2_err'].median(),
        'H3': sub['H3_err'].median(),
    }
    best = min(scores, key=scores.get)
    best_counts[best] += 1
    print(f"{label:<16} {len(sub):<6} {scores['A']:<8.0f} {scores['B']:<8.0f} {scores['C']:<8.0f} "
          f"{scores['H1']:<8.0f} {scores['H2']:<8.0f} {scores['H3']:<8.0f} {best:<6}")

print(f"\n最优次数: {best_counts}")

# ==================== 保存混合结果 ====================
# 选最优策略 H1
out_cols = ['patch_name','sample','size','gt_cx','gt_cy',
            'pith_in_patch','dist_to_pith','angle_span_deg','n_rings','n_ring_points',
            'A_err','B_err','C_err','H1_err','H2_err','H3_err']
df[out_cols].to_csv(OUT_DIR / "all_methods_error.csv", index=False)
print(f"\n输出: {OUT_DIR / 'all_methods_error.csv'}")
print(f"输出: {OUT_DIR / 'hybrid_summary.csv'}")
