"""
B面年轮盘拼接 最终版
=====================
髓心几何定位：每块放 (pith_x - cx, pith_y - cy)，所有块的髓心对齐
这自然产生无重叠、同心圆排列。
边缘匹配只用于验证 + 标记相邻关系。
"""
import sys, json, cv2, numpy as np, os, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

sys.path.insert(0, r'C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block')


def fit_circle_robust(points):
    if len(points) < 5: return None,None,None,0
    x,y = points[:,0], points[:,1]
    bn,bcx,bcy,br = 0,None,None,None
    for _ in range(min(150,len(points)*3)):
        idx = np.random.choice(len(points),min(5,len(points)),replace=False)
        sx,sy = points[idx,0],points[idx,1]
        A = np.column_stack([2*sx,2*sy,np.ones(len(sx))])
        try: sol = np.linalg.lstsq(A,sx**2+sy**2,rcond=None)[0]
        except: continue
        ct = np.sqrt(max(sol[2]+sol[0]**2+sol[1]**2,1))
        if ct<=0 or ct>100000: continue
        n=(np.abs(np.sqrt((x-sol[0])**2+(y-sol[1])**2)-ct)<0.10*ct).sum()
        if n>bn: bn,bcx,bcy,br=n,sol[0],sol[1],ct
    if bcx is None: return None,None,None,0
    inl = np.abs(np.sqrt((x-bcx)**2+(y-bcy)**2)-br)<0.15*br
    if inl.sum()<5: return bcx,bcy,br,bn/len(points)
    xi,yi = x[inl],y[inl]
    A = np.column_stack([2*xi,2*yi,np.ones(len(xi))])
    try:
        sol = np.linalg.lstsq(A,xi**2+yi**2,rcond=None)[0]
        return sol[0],sol[1],np.sqrt(max(sol[2]+sol[0]**2+sol[1]**2,1)),bn/len(points)
    except: return bcx,bcy,br,bn/len(points)


def extract_edge_crossings(ring_pts, img_W, img_H, margin=35):
    crossings = {'top':[],'bottom':[],'left':[],'right':[]}
    n = len(ring_pts)
    for i in range(n-1):
        p1,p2 = ring_pts[i], ring_pts[i+1]
        x1,y1,x2,y2 = p1[0],p1[1],p2[0],p2[1]
        # top
        if (y1<margin and y2>=margin) or (y2<margin and y1>=margin):
            t = (margin-y1)/(y2-y1+1e-10)
            if 0<=t<=1: crossings['top'].append((x1+t*(x2-x1))/img_W)
        elif y1<margin and y2<margin:
            cx = (x1+x2)/2/img_W
            if cx not in [c for c in crossings['top']]:
                crossings['top'].append(cx)
        # bottom
        thr = img_H-margin
        if (y1>thr and y2<=thr) or (y2>thr and y1<=thr):
            t = (thr-y1)/(y2-y1+1e-10)
            if 0<=t<=1: crossings['bottom'].append((x1+t*(x2-x1))/img_W)
        elif y1>thr and y2>thr:
            cx = (x1+x2)/2/img_W
            if cx not in [c for c in crossings['bottom']]:
                crossings['bottom'].append(cx)
        # left
        if (x1<margin and x2>=margin) or (x2<margin and x1>=margin):
            t = (margin-x1)/(x2-x1+1e-10)
            if 0<=t<=1: crossings['left'].append((y1+t*(y2-y1))/img_H)
        elif x1<margin and x2<margin:
            cy = (y1+y2)/2/img_H
            if cy not in [c for c in crossings['left']]:
                crossings['left'].append(cy)
        # right
        thr = img_W-margin
        if (x1>thr and x2<=thr) or (x2>thr and x1<=thr):
            t = (thr-x1)/(x2-x1+1e-10)
            if 0<=t<=1: crossings['right'].append((y1+t*(y2-y1))/img_H)
        elif x1>thr and x2>thr:
            cy = (y1+y2)/2/img_H
            if cy not in [c for c in crossings['right']]:
                crossings['right'].append(cy)

    for e in crossings:
        crossings[e] = sorted(set(round(p,3) for p in crossings[e]))
    return crossings


def load_blocks():
    BASE = Path(r"D:\教务处实习\木材数据1\立方体")
    blocks = []
    for board in [1,2,3]:
        d = BASE / f"B_{board}_blocks"
        csv_f = d / "_blocks_positions.csv"
        if not csv_f.exists(): continue
        pos_df = pd.read_csv(csv_f)
        for _,row in pos_df.iterrows():
            name = row['filename'].replace('.jpg','')
            img_p = d / f"{name}.jpg"; jf = d / f"{name}.json"
            if not img_p.exists() or not jf.exists(): continue
            img = cv2.imread(str(img_p))
            if img is None: continue
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if (gray<250).sum()/gray.size<0.03: continue

            with open(jf,'r',encoding='utf-8') as f: data = json.load(f)
            rings = [np.array(s['points']) for s in data.get('shapes',[])
                     if s.get('label')=='年轮' and len(s['points'])>=5]
            if len(rings)<2: continue

            centers,wts=[],[]
            for pts in rings:
                cx,cy,r,inl = fit_circle_robust(pts)
                if cx is not None and r>0 and inl>0.3:
                    centers.append((cx,cy,r)); wts.append(inl/r)
            if len(centers)>=2:
                wts=np.array(wts); wts/=wts.sum()
                cxs=np.array([c[0] for c in centers])
                cys=np.array([c[1] for c in centers])
                s=np.argsort(cxs); cs=np.cumsum(wts[s]); cx_e=cxs[s[np.searchsorted(cs,0.5)]]
                s=np.argsort(cys); cs=np.cumsum(wts[s]); cy_e=cys[s[np.searchsorted(cs,0.5)]]
            else:
                cx_e,cy_e=img.shape[1]/2,img.shape[0]/2

            ec = {}
            for pts in rings:
                ec_ring = extract_edge_crossings(pts, img.shape[1], img.shape[0])
                for e in ['top','bottom','left','right']:
                    ec.setdefault(e,[]).extend(ec_ring[e])
            for e in ec: ec[e]=sorted(set(round(p,3) for p in ec[e]))

            blocks.append({
                'name': name, 'board': board,
                'img': img, 'rings': rings,
                'W': img.shape[1], 'H': img.shape[0],
                'cx': cx_e, 'cy': cy_e,
                'edge_crossings': ec,
                'circles': [(c[0],c[1],c[2]) for c in centers]
            })
    return blocks


def compute_consensus_pith(blocks):
    """加权中位数共识髓心"""
    cxs = np.array([b['cx'] for b in blocks])
    cys = np.array([b['cy'] for b in blocks])
    ws = np.array([len(b['rings']) for b in blocks], dtype=float)
    ws /= ws.sum()
    s=np.argsort(cxs); cs=np.cumsum(ws[s]); cx=cxs[s[np.searchsorted(cs,0.5)]]
    s=np.argsort(cys); cs=np.cumsum(ws[s]); cy=cys[s[np.searchsorted(cs,0.5)]]
    return cx, cy


def find_edge_neighbors(blocks):
    """基于边缘交点匹配找邻居（用于验证和连线）"""
    neighbors = {}
    opposite = {'top':'bottom','bottom':'top','left':'right','right':'left'}
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks):
            if i >= j: continue
            for ea, eb in opposite.items():
                ca = np.array(a['edge_crossings'].get(ea,[]))
                cb = np.array(b['edge_crossings'].get(eb,[]))
                if len(ca)==0 or len(cb)==0: continue
                score=0
                for p1 in ca:
                    best=min((abs(p1-p2) for p2 in cb),default=1.0)
                    if best<0.08: score+=1.0-best/0.08
                score/=max(len(ca),len(cb))
                if score>0.4:
                    k = (a['name'], b['name'])
                    if k not in neighbors or score>neighbors[k][0]:
                        neighbors[k] = (score, ea, eb, a, b)
    return neighbors


def deduplicate_blocks(blocks, overlap_thresh=0.5):
    """
    去除重复块：如果两块髓心估计非常接近且年轮半径范围重叠超过阈值，
    说明它们是同一物理区域的不同照片 → 保留质量更好的
    """
    n = len(blocks)
    to_remove = set()

    for i in range(n):
        if i in to_remove: continue
        for j in range(i+1, n):
            if j in to_remove: continue

            a, b = blocks[i], blocks[j]

            # 髓心距离 vs 块尺寸
            pith_dist = np.sqrt((a['cx']-b['cx'])**2 + (a['cy']-b['cy'])**2)
            avg_diag = np.sqrt(a['W']**2+a['H']**2 + b['W']**2+b['H']**2)/2

            if pith_dist < 0.3 * avg_diag:
                # 髓心估计非常接近 → 可能重复
                # 比较年轮半径范围
                a_rs = [c[2] for c in a.get('circles', []) if len(c)>2]
                b_rs = [c[2] for c in b.get('circles', []) if len(c)>2]

                if a_rs and b_rs:
                    a_range = (min(a_rs), max(a_rs))
                    b_range = (min(b_rs), max(b_rs))
                    # 半径范围重叠比
                    r_overlap = (min(a_range[1], b_range[1]) - max(a_range[0], b_range[0]))
                    r_span = max(a_range[1], b_range[1]) - min(a_range[0], b_range[0])
                    if r_span > 0 and r_overlap / r_span > overlap_thresh:
                        # 重复 → 保留年轮更多的
                        if len(a['rings']) >= len(b['rings']):
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break

    unique = [b for i, b in enumerate(blocks) if i not in to_remove]
    if len(to_remove) > 0:
        print(f"  去重: 移除 {len(to_remove)} 个重复块, 保留 {len(unique)} 个唯一块")
    return unique


def visualize_geometric(blocks, out_path):
    """髓心几何对齐 + 边缘邻居连线 + 重叠消除"""
    cx_c, cy_c = compute_consensus_pith(blocks)

    # === 基于边缘匹配的拼图布局 ===
    # 1. 从髓心最近的块开始
    # 2. 每次放置与已放置块边缘匹配最好的未放置块
    # 3. 精确放置：匹配边的环交点对齐

    for b in blocks:
        b['placed'] = False
        b['gx'] = 0.0
        b['gy'] = 0.0

    # 找所有边缘匹配
    all_matches = []
    opposite = {'top':'bottom','bottom':'top','left':'right','right':'left'}
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks):
            if i >= j: continue
            for ea, eb in opposite.items():
                ca = np.array(a['edge_crossings'].get(ea, []))
                cb = np.array(b['edge_crossings'].get(eb, []))
                if len(ca)==0 or len(cb)==0: continue
                score=0
                for p1 in ca:
                    best=min((abs(p1-p2) for p2 in cb),default=1.0)
                    if best<0.08: score+=1.0-best/0.08
                score/=max(len(ca),len(cb))
                if score>0.3:
                    all_matches.append((score, i, j, ea, eb))
    all_matches.sort(key=lambda x: -x[0])
    print(f"  共 {len(all_matches)} 个边缘匹配候选")

    # 起点：髓心最接近块中心的
    start_idx = min(range(len(blocks)),
                    key=lambda i: np.sqrt((blocks[i]['cx']-blocks[i]['W']/2)**2
                                         +(blocks[i]['cy']-blocks[i]['H']/2)**2))
    blocks[start_idx]['gx'] = cx_c - blocks[start_idx]['cx']
    blocks[start_idx]['gy'] = cy_c - blocks[start_idx]['cy']
    blocks[start_idx]['placed'] = True
    placed_count = 1
    print(f"  起点: {blocks[start_idx]['name']} at ({blocks[start_idx]['gx']:.0f},{blocks[start_idx]['gy']:.0f})")

    # 贪心放置其余块
    round_num = 0
    while placed_count < len(blocks):
        round_num += 1
        best_placement = None
        best_score = -1

        for score, i, j, ea, eb in all_matches:
            a, b = blocks[i], blocks[j]
            if a['placed'] and not b['placed']:
                parent, child = a, b
                parent_edge, child_edge = ea, eb
            elif b['placed'] and not a['placed']:
                parent, child = b, a
                parent_edge, child_edge = eb, ea
            else:
                continue

            # 计算child的精确位置
            if parent_edge == 'right' and child_edge == 'left':
                gx = parent['gx'] + parent['W']
                gy = parent['gy']
            elif parent_edge == 'left' and child_edge == 'right':
                gx = parent['gx'] - child['W']
                gy = parent['gy']
            elif parent_edge == 'bottom' and child_edge == 'top':
                gx = parent['gx']
                gy = parent['gy'] + parent['H']
            elif parent_edge == 'top' and child_edge == 'bottom':
                gx = parent['gx']
                gy = parent['gy'] - child['H']
            else:
                continue

            # 确保不与其他已放置块重叠太多
            overlap_area = 0
            for other in blocks:
                if not other['placed']: continue
                ox, oy = other['gx'], other['gy']
                ow, oh = other['W'], other['H']
                ol = max(gx, ox)
                ot = max(gy, oy)
                or_ = min(gx+child['W'], ox+ow)
                ob = min(gy+child['H'], oy+oh)
                if ol < or_ and ot < ob:
                    overlap_area += (or_-ol)*(ob-ot)

            if overlap_area < 0.3 * child['W'] * child['H']:
                if score > best_score:
                    best_score = score
                    best_placement = (child, gx, gy, parent['name'], score)

        if best_placement is not None:
            child, gx, gy, pname, sc = best_placement
            child['gx'] = gx
            child['gy'] = gy
            child['placed'] = True
            placed_count += 1
            if placed_count % 5 == 0:
                print(f"  已放置 {placed_count}/{len(blocks)}...")
        else:
            # Fallback: 放在最远端
            for b in blocks:
                if not b['placed']:
                    max_x = max((blk['gx']+blk['W'] for blk in blocks if blk['placed']), default=0)
                    min_y = min((blk['gy'] for blk in blocks if blk['placed']), default=0)
                    b['gx'] = max_x + 100
                    b['gy'] = min_y
                    b['placed'] = True
                    placed_count += 1
                    print(f"  Fallback: {b['name']}")
                    break

    # === 偏移到正坐标 ===
    min_x = min(b['gx'] for b in blocks)
    min_y = min(b['gy'] for b in blocks)
    margin = 2000
    for b in blocks:
        b['gx'] = b['gx'] - min_x + margin
        b['gy'] = b['gy'] - min_y + margin

    max_x = max(b['gx']+b['W'] for b in blocks) + margin
    max_y = max(b['gy']+b['H'] for b in blocks) + margin
    cw, ch = int(max_x), int(max_y)

    # 全局髓心画布位置
    pith_x = cx_c - min_x + margin
    pith_y = cy_c - min_y + margin

    # 边缘邻居
    neighbors = find_edge_neighbors(blocks)

    # ==== 可视化 ====
    fig, ax = plt.subplots(1, 1, figsize=(28, 24))
    ax.set_facecolor('#0d1117')

    for r in [2000,4000,6000,8000,10000,12000]:
        circle = plt.Circle((pith_x, pith_y), r, fill=False,
                            edgecolor='#f39c12', lw=0.5, alpha=0.12, ls='--')
        ax.add_patch(circle)

    cmap = plt.cm.tab20b
    n_b = len(blocks)
    colors = [cmap(i%20) for i in range(n_b)]

    overlap_count = 0
    placed_rects = []

    for i, b in enumerate(blocks):
        gx, gy = int(b['gx']), int(b['gy'])

        r1 = [gx, gy, gx+b['W'], gy+b['H']]
        for r2 in placed_rects:
            if r1[0]<r2[2] and r1[2]>r2[0] and r1[1]<r2[3] and r1[3]>r2[1]:
                overlap_count += 1
        placed_rects.append(r1)

        color = colors[i]
        alpha = 0.4

        rect = FancyBboxPatch((gx, gy), b['W'], b['H'],
                              boxstyle="round,pad=3", facecolor=color,
                              edgecolor='white', lw=0.8, alpha=alpha)
        ax.add_patch(rect)

        for ring_pts in b['rings']:
            ax.plot(gx+ring_pts[:,0], gy+ring_pts[:,1],
                    '-', color='white', lw=0.5, alpha=0.7)

        px, py = gx+b['cx'], gy+b['cy']
        err = np.sqrt((b['cx']-cx_c)**2+(b['cy']-cy_c)**2)
        mc = '#2ecc71' if err<1500 else ('#f39c12' if err<3000 else '#e74c3c')
        ax.plot(px, py, 'o', color=mc, markersize=9, alpha=0.9,
                markeredgecolor='white', lw=0.5)

        label = b['name'].replace('B_1_','B1-').replace('B_2_','B2-').replace('B_3_','B3-')
        ax.text(gx+b['W']/2, gy+b['H']/2, label,
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.15',
                                         facecolor='black', alpha=0.55))

    for (a_name, b_name), (score, ea, eb, a, b) in neighbors.items():
        ax_gx = int(a['gx'] + a['W']/2)
        ax_gy = int(a['gy'] + a['H']/2)
        bx_gx = int(b['gx'] + b['W']/2)
        bx_gy = int(b['gy'] + b['H']/2)
        ax.plot([ax_gx, bx_gx], [ax_gy, bx_gy], '-', color='#58a6ff',
                lw=score*2.5, alpha=min(score,1)*0.5)

    ax.plot(pith_x, pith_y, 'r*', markersize=45, markeredgewidth=4,
            markeredgecolor='yellow', zorder=200,
            label=f'Consensus Pith ({cx_c:.0f}, {cy_c:.0f})')

    errs = [np.sqrt((b['cx']-cx_c)**2+(b['cy']-cy_c)**2) for b in blocks]
    median_err = np.median(errs)

    ax.set_title(f'B Face — Tree Ring Cross-Section ({len(blocks)} unique blocks)\n'
                 f'Pith=({cx_c:.0f},{cy_c:.0f})  |  '
                 f'Median error={median_err:.0f}px  |  '
                 f'Overlaps={overlap_count}  |  Neighbors={len(neighbors)}  |  '
                 f'Approx. circular',
                 fontsize=16, fontweight='bold', color='white')
    ax.legend(fontsize=12, loc='upper right')

    ax.set_xlim(0, cw)
    ax.set_ylim(ch, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    for ext in ['png','pdf']:
        fig.savefig(out_path + f'.{ext}', dpi=150, bbox_inches='tight',
                    facecolor='#0d1117')
    plt.close(fig)

    print(f"\nB面最终拼图:")
    print(f"  画布: {cw}x{ch}px")
    print(f"  共识髓心: ({cx_c:.0f}, {cy_c:.0f})")
    print(f"  中位数误差: {median_err:.0f}px")
    print(f"  平均误差:   {np.mean(errs):.0f}px")
    print(f"  重叠块对数: {overlap_count}")
    print(f"  边缘邻居对数: {len(neighbors)}")
    print(f"  => {out_path}.png")

    return out_path


if __name__ == "__main__":
    blocks = load_blocks()
    print(f"加载 {len(blocks)} 个B面有效块")

    out_path = r"C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block\pith_consensus_output\B_face_final"

    # 去重
    blocks = deduplicate_blocks(blocks)
    visualize_geometric(blocks, out_path)

    # 启动
    import subprocess
    subprocess.run(['start', '', str(Path(out_path).with_suffix('.png'))], shell=True)
