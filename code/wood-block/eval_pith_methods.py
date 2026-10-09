"""
patch_dataset 全量评测
======================
输出结构:
  patch_dataset/
  ├── results/          ← 评测结果CSV
  │   ├── eval_all.csv
  │   └── summary.csv
  └── debug/            ← 调试验证文件（每块可视化）
"""
import sys, os, types, json, cv2, numpy as np, pandas as pd, time, warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
random_state = np.random.RandomState(42)

DATA = Path(r"D:\教务处实习\木材数据1\patch_dataset")
PATCHES = DATA / "裁剪矩形"
RESULTS = DATA / "results"
DEBUG = DATA / "debug"
RESULTS.mkdir(exist_ok=True, parents=True)
DEBUG.mkdir(exist_ok=True, parents=True)

# ====== 圆拟合 ======
def fit_circle_robust(points):
    if len(points) < 5: return None,None,None,0
    x,y = points[:,0], points[:,1]
    bn,bcx,bcy,br = 0,None,None,None
    for _ in range(min(100,len(points)*2)):
        idx = random_state.choice(len(points), min(5,len(points)), replace=False)
        sx,sy = points[idx,0],points[idx,1]
        A = np.column_stack([2*sx,2*sy,np.ones(len(sx))])
        try: sol = np.linalg.lstsq(A, sx**2+sy**2, rcond=None)[0]
        except: continue
        ct = np.sqrt(max(sol[2]+sol[0]**2+sol[1]**2, 1))
        if ct<=0 or ct>50000: continue
        n = (np.abs(np.sqrt((x-sol[0])**2+(y-sol[1])**2)-ct) < 0.10*ct).sum()
        if n>bn: bn,bcx,bcy,br = n,sol[0],sol[1],ct
    if bcx is None: return None,None,None,0
    inl = np.abs(np.sqrt((x-bcx)**2+(y-bcy)**2)-br) < 0.15*br
    if inl.sum()<5: return bcx,bcy,br,bn/len(points)
    xi,yi = x[inl],y[inl]
    A = np.column_stack([2*xi,2*yi,np.ones(len(xi))])
    try:
        sol = np.linalg.lstsq(A, xi**2+yi**2, rcond=None)[0]
        return sol[0],sol[1],np.sqrt(max(sol[2]+sol[0]**2+sol[1]**2,1)),bn/len(points)
    except: return bcx,bcy,br,bn/len(points)

def estimate_pith_from_rings(rings):
    """独立圆拟合 + 加权中位数"""
    centers, wts = [], []
    for pts in rings:
        cx,cy,r,inl = fit_circle_robust(pts)
        if cx is not None and r>0 and inl>0.3:
            centers.append((cx,cy,r)); wts.append(inl/r)
    if len(centers)<2: return None,None,None

    wts = np.array(wts); wts /= wts.sum()
    cxs = np.array([c[0] for c in centers]); cys = np.array([c[1] for c in centers])
    s = np.argsort(cxs); cs = np.cumsum(wts[s]); cx = cxs[s[np.searchsorted(cs,0.5)]]
    s = np.argsort(cys); cs = np.cumsum(wts[s]); cy = cys[s[np.searchsorted(cs,0.5)]]

    # 置信度
    consist = 1.0/(1.0 + (np.sqrt(np.average((cxs-cx)**2,weights=wts)) +
                          np.sqrt(np.average((cys-cy)**2,weights=wts)))/2/
                  max(abs(cx)+abs(cy),1))
    return cx, cy, consist


def evaluate_one(row, df_orig):
    """评测单个小块"""
    patch_name = row['patch_name']
    sample = row['sample']
    img_path = PATCHES / sample / f"{patch_name}.jpg"

    if not img_path.exists(): return None

    gt_cx, gt_cy = row['cx_gt'], row['cy_gt']
    bx, by = row['bx'], row['by']
    size = row['size']

    # 从原始标注提取块内年轮点
    ann_dir = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\annotations\annual_rings")
    jf = ann_dir / f"{sample}.json"
    if not jf.exists(): return None
    with open(jf,'r',encoding='utf-8') as f: full_data = json.load(f)

    local_rings = []
    for shape in full_data.get('shapes',[]):
        pts = np.array(shape['points'])
        pts_local = pts - np.array([bx,by])
        valid = (pts_local[:,0]>=0)&(pts_local[:,0]<size)&(pts_local[:,1]>=0)&(pts_local[:,1]<size)
        if valid.sum()>=5: local_rings.append(pts_local[valid])

    if len(local_rings)<2: return None

    t0 = time.time()
    cx, cy, conf = estimate_pith_from_rings(local_rings)
    dt = time.time()-t0

    if cx is None: return None

    err = np.sqrt((cx-gt_cx)**2+(cy-gt_cy)**2)
    return {
        'patch_name': patch_name, 'sample': sample,
        'gt_cx': gt_cx, 'gt_cy': gt_cy,
        'est_cx': cx, 'est_cy': cy,
        'error': err, 'confidence': conf,
        'size': size, 'dist_to_pith': row['dist_to_pith'],
        'angle_span_deg': np.degrees(row['angle_span_rad']),
        'n_rings': row['n_ring_arcs'],
        'n_ring_points': row['n_ring_points'],
        'time_s': dt
    }


def plot_sample_debug(row, local_rings, out_path):
    """生成单块调试可视化"""
    img_path = DATA / row['sample'] / f"{row['patch_name']}.jpg"
    img = cv2.imread(str(img_path))
    if img is None: return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gt_cx, gt_cy = row['gt_cx'], row['gt_cy']
    est_cx, est_cy = row['est_cx'], row['est_cy']

    for ax, title in zip(axes, ['Ground Truth rings', 'Estimated pith vs GT']):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for ring in local_rings:
            ax.plot(ring[:,0], ring[:,1], 'c-', lw=0.5, alpha=0.6)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)

    axes[0].plot(gt_cx, gt_cy, 'r*', markersize=15, label=f'GT({gt_cx:.0f},{gt_cy:.0f})')
    axes[1].plot(gt_cx, gt_cy, 'g*', markersize=15, label=f'GT({gt_cx:.0f},{gt_cy:.0f})')
    axes[1].plot(est_cx, est_cy, 'ro', markersize=10, label=f'Est({est_cx:.0f},{est_cy:.0f})')
    axes[1].plot([gt_cx,est_cx],[gt_cy,est_cy],'r--',lw=1)

    for ax in axes: ax.legend(fontsize=8)

    fig.suptitle(f'{row["patch_name"]} | error={row["error"]:.0f}px | '
                 f'dist={row["dist_to_pith"]:.0f} | ang={row["angle_span_deg"]:.0f}deg',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA / "dataset（元数据）.csv")
    n = len(df)
    print(f"全量评测 {n} 块...")

    results = []
    debug_samples = set()

    for i, (_, row) in enumerate(df.iterrows()):
        r = evaluate_one(row, df)
        if r: results.append(r)

        # 每100块生成几个调试样本
        if r and (r['error'] < 5 or r['error'] > 2000 or i%200==0):
            if len(debug_samples) < 50:
                debug_samples.add(i)
                # 重新获取rings用于可视化
                sample = row['sample']
                bx,by,size = row['bx'],row['by'],row['size']
                ann_dir = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\annotations\annual_rings")
                jf = ann_dir / f"{sample}.json"
                if jf.exists():
                    with open(jf,'r',encoding='utf-8') as f: fd = json.load(f)
                    lr = []
                    for s in fd.get('shapes',[]):
                        pts = np.array(s['points'])-np.array([bx,by])
                        v = (pts[:,0]>=0)&(pts[:,0]<size)&(pts[:,1]>=0)&(pts[:,1]<size)
                        if v.sum()>=5: lr.append(pts[v])
                    debug_out = DEBUG / f"{r['patch_name']}.png"
                    plot_sample_debug(r, lr, str(debug_out))

        if (i+1) % 500 == 0:
            print(f"  进度: {i+1}/{n} ({len(results)} valid)")

    # 保存结果
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS / "eval_all.csv", index=False)

    # 汇总统计
    errs = res_df['error']
    summary = {
        'total_patches': n,
        'valid_results': len(res_df),
        'median_error': np.median(errs),
        'mean_error': np.mean(errs),
        'pct_lt_100': 100*(errs<100).sum()/len(errs),
        'pct_lt_200': 100*(errs<200).sum()/len(errs),
        'pct_lt_500': 100*(errs<500).sum()/len(errs),
        'pct_lt_1000': 100*(errs<1000).sum()/len(errs),
    }

    # 分组统计
    groups = [
        ('pith_IN', res_df['dist_to_pith'] < res_df['size']),
        ('pith_OUT', res_df['dist_to_pith'] >= res_df['size']),
        ('dist<500', res_df['dist_to_pith'] < 500),
        ('dist_500-1500', (res_df['dist_to_pith']>=500)&(res_df['dist_to_pith']<1500)),
        ('dist>1500', res_df['dist_to_pith'] >= 1500),
        ('ang<45', res_df['angle_span_deg'] < 45),
        ('ang_45-90', (res_df['angle_span_deg']>=45)&(res_df['angle_span_deg']<90)),
        ('ang>90', res_df['angle_span_deg'] >= 90),
        ('rings<5', res_df['n_rings'] < 5),
        ('rings_5-10', (res_df['n_rings']>=5)&(res_df['n_rings']<10)),
        ('rings>10', res_df['n_rings'] >= 10),
    ]

    group_rows = []
    for label, cond in groups:
        sub = res_df[cond]
        if len(sub)>0:
            e = sub['error']
            group_rows.append({
                'group': label, 'n': len(sub),
                'median_err': np.median(e), 'mean_err': np.mean(e),
                'pct_lt_200': 100*(e<200).sum()/len(e),
                'pct_lt_500': 100*(e<500).sum()/len(e),
            })

    group_df = pd.DataFrame(group_rows)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(RESULTS / "summary.csv", index=False)
    group_df.to_csv(RESULTS / "by_group.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Done! {len(res_df)}/{n} valid patches")
    print(f"  中位数误差: {summary['median_error']:.0f}px")
    print(f"  <200px:     {summary['pct_lt_200']:.0f}%")
    print(f"  <500px:     {summary['pct_lt_500']:.0f}%")
    print(f"  <1000px:    {summary['pct_lt_1000']:.0f}%")
    print(f"\n  结果: {RESULTS}")
    print(f"  调试: {DEBUG} ({len(debug_samples)} 张可视化)")

    # 打印分组
    print(f"\n{'Group':<16} {'n':<6} {'median':<10} {'<200':<8} {'<500':<8}")
    print('-'*50)
    for _, g in group_df.iterrows():
        print(f"{g['group']:<16} {g['n']:<6} {g['median_err']:<10.0f} {g['pct_lt_200']:<8.0f} {g['pct_lt_500']:<8.0f}")


if __name__ == "__main__":
    main()
