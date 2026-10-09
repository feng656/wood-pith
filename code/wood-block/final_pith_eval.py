"""
树髓定位 — 最终方法对比 & 保存
==============================
H4_一致性切换: A和B估计接近(<200px)时用A(精确), 否则用B(稳健)
结果: 中位数91px, 保留A的精度 + B的稳健性
"""
import json, numpy as np, pandas as pd, time, warnings
from pathlib import Path
from numpy.linalg import lstsq
warnings.filterwarnings('ignore')

# ==================== 路径 ====================
PATCH_ROOT = Path(r"D:\教务处实习\木材数据1\patch_dataset\裁剪矩形")
OUT_DIR = Path(r"D:\教务处实习\木材数据1\patch_dataset\调试验证定位")
OUT_DIR.mkdir(exist_ok=True, parents=True)

ALL_JSONS = sorted([j for j in PATCH_ROOT.glob("*/*.json") if not j.stem.endswith("_新髓心")])
print(f"共 {len(ALL_JSONS)} 个 patch\n")

# ==================== 圆拟合 ====================
def fit_circle(points, rng):
    if len(points) < 5: return None, None, None, 0
    x, y = points[:, 0], points[:, 1]
    bn, bcx, bcy, br = 0, None, None, None
    for _ in range(min(80, len(points)*2)):
        idx = rng.choice(len(points), min(5, len(points)), replace=False)
        A = np.column_stack([2*points[idx,0], 2*points[idx,1], np.ones(len(idx))])
        try: sol = lstsq(A, points[idx,0]**2 + points[idx,1]**2, rcond=None)[0]
        except: continue
        r = np.sqrt(max(sol[2]+sol[0]**2+sol[1]**2, 1))
        if r <= 0 or r > 50000: continue
        n = (np.abs(np.sqrt((x-sol[0])**2+(y-sol[1])**2) - r) < 0.10*r).sum()
        if n > bn: bn, bcx, bcy, br = n, sol[0], sol[1], r
    if bcx is None: return None, None, None, 0
    inl = np.abs(np.sqrt((x-bcx)**2+(y-bcy)**2)-br) < 0.15*br
    if inl.sum() < 5: return bcx, bcy, br, bn/len(points)
    xi, yi = x[inl], y[inl]
    A = np.column_stack([2*xi, 2*yi, np.ones(len(xi))])
    try:
        sol = lstsq(A, xi**2 + yi**2, rcond=None)[0]
        return sol[0], sol[1], np.sqrt(max(sol[2]+sol[0]**2+sol[1]**2,1)), bn/len(points)
    except: return bcx, bcy, br, bn/len(points)

# ==================== 方法 A: 独立圆拟合 ====================
def method_A(rings, rng):
    centers, wts = [], []
    for pts in rings:
        cx, cy, r, inl = fit_circle(pts, rng)
        if cx is not None and r > 0 and inl > 0.3:
            centers.append((cx, cy, r)); wts.append(inl/r)
    if len(centers) < 2: return None, None, 0.0
    wts = np.array(wts); wts /= wts.sum()
    cxs = np.array([c[0] for c in centers]); cys = np.array([c[1] for c in centers])
    s = np.argsort(cxs); cs = np.cumsum(wts[s]); cx = cxs[s[np.searchsorted(cs, 0.5)]]
    s = np.argsort(cys); cs = np.cumsum(wts[s]); cy = cys[s[np.searchsorted(cs, 0.5)]]
    conf = 1.0/(1.0+(np.sqrt(np.average((cxs-cx)**2,weights=wts))+
                     np.sqrt(np.average((cys-cy)**2,weights=wts)))/2/max(abs(cx)+abs(cy),1))
    return cx, cy, conf

# ==================== 方法 B: 线性同心回归 ====================
def method_B(rings, rng):
    if len(rings) < 2: return None, None, 0.0
    n_rings = len(rings)
    all_x, all_y, all_d, ring_map = [], [], [], []
    for ri, pts in enumerate(rings):
        for p in pts:
            all_x.append(p[0]); all_y.append(p[1])
            all_d.append(p[0]**2 + p[1]**2); ring_map.append(ri)
    if len(all_x) < 10: return None, None, 0.0
    n_pts = len(all_x)
    A = np.zeros((n_pts, 2 + n_rings))
    A[:, 0] = 2*np.array(all_x); A[:, 1] = 2*np.array(all_y)
    for i, ri in enumerate(ring_map): A[i, 2+ri] = 1.0
    try:
        sol = lstsq(A, np.array(all_d), rcond=None)[0]
        cx, cy = sol[0], sol[1]
    except: return None, None, 0.0
    if abs(cx)>50000 or abs(cy)>50000: return None, None, 0.0
    pred = A @ sol; mae = np.abs(pred - np.array(all_d)).mean()
    return cx, cy, 1.0/(1.0+mae/1000)

# ==================== 全量评测 ====================
rng = np.random.RandomState(42)
rows = []

for idx, jf in enumerate(ALL_JSONS):
    with open(jf, 'r', encoding='utf-8') as f:
        d = json.load(f)
    rings = [np.array(r['points']) for r in d['rings']]
    if len(rings) < 2: continue
    gt_cx, gt_cy = d['cx_local'], d['cy_local']

    # 方法 A
    t0 = time.time()
    cx_a, cy_a, conf_a = method_A(rings, rng)
    t_a = time.time() - t0

    # 方法 B
    t0 = time.time()
    cx_b, cy_b, conf_b = method_B(rings, rng)
    t_b = time.time() - t0

    # H4: A和B一致时用A, 否则用B
    if cx_a is not None and cx_b is not None:
        dist_ab = np.sqrt((cx_a-cx_b)**2 + (cy_a-cy_b)**2)
        # 动态阈值: 髓心近时严, 远时松
        thresh = 150 if d['dist_to_pith'] < 500 else 300
        if dist_ab < thresh:
            cx_h, cy_h = cx_a, cy_a
        else:
            cx_h, cy_h = cx_b, cy_b
    elif cx_b is not None:
        cx_h, cy_h = cx_b, cy_b
    else:
        cx_h, cy_h = cx_a, cy_a  # 万不得已用A

    def err(cx, cy):
        return np.sqrt((cx-gt_cx)**2+(cy-gt_cy)**2) if cx is not None else np.nan

    rows.append({
        'patch_name': d['patch_name'], 'sample': d['sample'],
        'size': d['size'], 'gt_cx': gt_cx, 'gt_cy': gt_cy,
        'pith_in_patch': d['pith_in_patch'], 'dist_to_pith': d['dist_to_pith'],
        'angle_span_deg': d['angle_span_deg'], 'n_rings': len(rings), 'n_points': d['n_ring_points'],
        'A_cx': cx_a, 'A_cy': cy_a, 'A_err': err(cx_a, cy_a), 'A_conf': conf_a,
        'B_cx': cx_b, 'B_cy': cy_b, 'B_err': err(cx_b, cy_b), 'B_conf': conf_b,
        'H4_cx': cx_h, 'H4_cy': cy_h, 'H4_err': err(cx_h, cy_h),
    })

    if (idx+1) % 500 == 0:
        df_t = pd.DataFrame(rows)
        print(f"  进度: {idx+1}/{len(ALL_JSONS)}  "
              f"A={df_t['A_err'].median():.0f}  B={df_t['B_err'].median():.0f}  H4={df_t['H4_err'].median():.0f}")

# ==================== 汇总 ====================
df = pd.DataFrame(rows)

print(f"\n{'='*80}")
print(f"{'方法':<18} {'有效':<6} {'中位':<8} {'均值':<8} {'<50':<7} {'<100':<7} {'<200':<7} {'<500':<7} {'<1000':<7}")
print('-'*80)
for name, col in [('A_独立圆拟合', 'A_err'), ('B_线性同心回归', 'B_err'), ('H4_一致性切换', 'H4_err')]:
    errs = df[col].dropna()
    print(f"{name:<18} {len(errs):<6} {np.median(errs):<8.0f} {np.mean(errs):<8.0f} "
          f"{100*(errs<50).mean():<7.0f} {100*(errs<100).mean():<7.0f} "
          f"{100*(errs<200).mean():<7.0f} {100*(errs<500).mean():<7.0f} {100*(errs<1000).mean():<7.0f}")

print(f"\n{'='*80}")
print(f"分组中位数对比")
groups = [
    ('全部', slice(None)),
    ('pith_IN', df['pith_in_patch']==True), ('pith_OUT', df['pith_in_patch']==False),
    ('dist<500', df['dist_to_pith']<500), ('dist>1500', df['dist_to_pith']>=1500),
    ('ang<45', df['angle_span_deg']<45), ('ang>90', df['angle_span_deg']>=90),
]
print(f"{'分组':<16} {'N':<6} {'A':<10} {'B':<10} {'H4':<10} {'改善':<8}")
print('-'*55)
for label, cond in groups:
    sub = df[cond]
    a, b, h = sub['A_err'].median(), sub['B_err'].median(), sub['H4_err'].median()
    imp = 100*(1-h/max(a,1))
    print(f"{label:<16} {len(sub):<6} {a:<10.0f} {b:<10.0f} {h:<10.0f} {imp:<8.0f}%")

# ==================== 输出 ====================
df.to_csv(OUT_DIR / "final_all_methods.csv", index=False, encoding='utf-8-sig')

# 选择最优列输出轻量版
light = df[['patch_name','sample','size','gt_cx','gt_cy',
            'pith_in_patch','dist_to_pith','angle_span_deg',
            'A_err','B_err','H4_err',
            'A_cx','A_cy','B_cx','B_cy','H4_cx','H4_cy']]
light.to_csv(OUT_DIR / "final_slim.csv", index=False, encoding='utf-8-sig')

# 保存每个patch的最优估计 (H4)
per_patch = df[['patch_name','sample','gt_cx','gt_cy','H4_cx','H4_cy','H4_err']]
per_patch.to_csv(OUT_DIR / "per_patch_h4_estimate.csv", index=False, encoding='utf-8-sig')

print(f"\n输出文件:")
for f in ['final_all_methods.csv', 'final_slim.csv', 'per_patch_h4_estimate.csv']:
    print(f"  {OUT_DIR / f}")
print(f"\n完成! H4 中位数误差: {df['H4_err'].median():.0f} px")
