"""
木块立方体展开图生成工具 v2
人脸命名：
  A = 前面 (+z, 居中)
  B = 顶面 (+y, A上方)
  C = 左面 (-x, A左侧)
  D = 底面 (-y, A下方)
  E = 右面 (+x, A右侧)
  F = 后面 (-z, E右侧)

十字展开图标签布局（图片位置不变：D的图在上方，B的图在下方）：
                [B] 顶面   ← 显示D的图
    [C] [A] [E] [F]
                [D] 底面   ← 显示B的图

12条棱的衔接检查：
  直接相邻(5): A-D, A-B, A-C, A-E, E-F
  折叠相连(7): D-C, D-E, D-F, C-B, B-E, B-F, C-F
"""

import cv2
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
BASE_DIR = r'D:\教务处实习\木材数据1\立方体'
OUTPUT_DIR = r'C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block\cube_unfold_output'
ROWS, COLS = 5, 4
CUBE_FACES = ['A', 'B', 'C', 'D', 'E', 'F']
EMPTY_POSITIONS = {(3, 3)}

EDGE_CN = {'top': '上', 'bottom': '下', 'left': '左', 'right': '右'}
FACE_CN = {'A': '前面', 'B': '顶面', 'C': '左面', 'D': '底面', 'E': '右面', 'F': '后面'}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 数据加载 ====================
def load_face_image(letter, r, c):
    block_dir = os.path.join(BASE_DIR, f'{letter}_1_blocks')
    fname = f'{letter}_1_{r}{c}.jpg'
    fpath = os.path.join(block_dir, fname)
    if not os.path.exists(fpath):
        return None
    img = cv2.imread(fpath)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (gray > 240).mean() > 0.95:
        return None
    return img


def load_annotations(letter, r, c):
    block_dir = os.path.join(BASE_DIR, f'{letter}_1_blocks')
    fname = f'{letter}_1_{r}{c}.json'
    fpath = os.path.join(block_dir, fname)
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cube(r, c):
    faces = {}
    for letter in CUBE_FACES:
        img = load_face_image(letter, r, c)
        annotations = load_annotations(letter, r, c)
        faces[letter] = {'image': img, 'annotations': annotations, 'label': f'{letter}_1_{r}{c}'}
    return faces


# ==================== 标注绘制 ====================
def draw_rings(img, annotations, color=(50, 50, 220), thickness=2):
    if img is None:
        return None
    vis = img.copy()
    if annotations is None:
        return vis
    for shape in annotations.get('shapes', []):
        if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.polylines(vis, [points], False, color, thickness)
    return vis


def face_to_square(face_info, size=400):
    """将面图像缩放到正方形"""
    img = face_info['image']
    ann = face_info['annotations']
    if img is None:
        empty = np.zeros((size, size, 3), dtype=np.uint8)
        empty[:] = [50, 50, 50]
        cv2.putText(empty, '(空)', (size//3, size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        return empty
    vis = draw_rings(img, ann)
    h, w = vis.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    vis = cv2.resize(vis, (nw, nh))
    result = np.zeros((size, size, 3), dtype=np.uint8)
    result[(size - nh)//2:(size - nh)//2 + nh, (size - nw)//2:(size - nw)//2 + nw] = vis
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def get_edge_strip(face_info, edge, length=300, width=40):
    """提取边缘条带"""
    img = face_info['image']
    ann = face_info['annotations']
    if img is None:
        blank = np.ones((length if edge in ['left','right'] else width,
                         length if edge in ['top','bottom'] else width, 3), dtype=np.uint8) * 50
        return blank
    vis = draw_rings(img, ann)
    h, w = vis.shape[:2]
    if edge == 'top':
        s = vis[:width, :, :]
    elif edge == 'bottom':
        s = vis[-width:, :, :]
    elif edge == 'left':
        s = vis[:, :width, :]
    elif edge == 'right':
        s = vis[:, -width:, :]
    if edge in ['top', 'bottom']:
        s = cv2.resize(s, (length, width))
    else:
        s = cv2.resize(s, (width, length))
    return cv2.cvtColor(s, cv2.COLOR_BGR2RGB)


# ==================== 展开图 ====================
def create_unfold(faces, face_size=380):
    """
    十字展开图标签:
                    [B] 顶面   row 0, col 1  (显示D的图)
        [C] [A] [E] [F]       row 1
                    [D] 底面   row 2, col 1  (显示B的图)
    """
    # 图片位置不变（D的图在A上方，B的图在A下方），仅标签换字母：上=顶面标B，下=底面标D
    label_at_position = {
        (1, 0): 'B',    # 上面位置标B（顶面）
        (1, 1): 'A',
        (1, 2): 'D',    # 下面位置标D（底面）
        (0, 1): 'C',
        (2, 1): 'E',
        (3, 1): 'F',
    }
    # layout 决定每个位置显示哪个面的图片（图片不动：D的图在上，B的图在下）
    layout = {
        (1, 0): 'D',    # 上面显示D的图
        (1, 1): 'A',
        (1, 2): 'B',    # 下面显示B的图
        (0, 1): 'C',
        (2, 1): 'E',
        (3, 1): 'F',
    }
    gap = 5
    grid_w, grid_h = 4, 3
    cw = grid_w * face_size + (grid_w - 1) * gap
    ch = grid_h * face_size + (grid_h - 1) * gap
    canvas = np.ones((ch, cw, 3), dtype=np.uint8) * 240

    face_imgs = {}
    for letter in CUBE_FACES:
        face_imgs[letter] = face_to_square(faces[letter], size=face_size)

    positions = {}
    for (gcol, grow), img_letter in layout.items():
        px = gcol * (face_size + gap)
        py = grow * (face_size + gap)
        label_letter = label_at_position.get((gcol, grow), img_letter)
        positions[label_letter] = (px, py)
        canvas[py:py + face_size, px:px + face_size] = face_imgs[img_letter]

        # 面标签（用label_letter）
        label = f"{label_letter} ({FACE_CN[label_letter]})"
        cv2.putText(canvas, label, (px + 8, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(canvas, label, (px + 8, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # 标注状态（检查实际图片面的标注）
        status = '✓' if faces[img_letter]['annotations'] is not None else '○'
        sc = (0, 200, 0) if faces[img_letter]['annotations'] is not None else (150, 150, 150)
        cv2.putText(canvas, status, (px + face_size - 30, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(canvas, status, (px + face_size - 30, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 1)

    # 边缘颜色标记
    edge_colors = {'top': (220, 50, 50), 'bottom': (50, 150, 220),
                   'left': (50, 200, 50), 'right': (220, 180, 50)}
    ml = 30
    for letter, (px, py) in positions.items():
        fs = face_size
        for edge, color in edge_colors.items():
            if edge == 'top':
                cv2.line(canvas, (px + fs//2 - ml, py + 2), (px + fs//2 + ml, py + 2), color, 4)
            elif edge == 'bottom':
                cv2.line(canvas, (px + fs//2 - ml, py + fs - 3), (px + fs//2 + ml, py + fs - 3), color, 4)
            elif edge == 'left':
                cv2.line(canvas, (px + 2, py + fs//2 - ml), (px + 2, py + fs//2 + ml), color, 4)
            elif edge == 'right':
                cv2.line(canvas, (px + fs - 3, py + fs//2 - ml), (px + fs - 3, py + fs//2 + ml), color, 4)

    return canvas, positions


# ==================== 12条棱的边对 ====================
# 直接相邻（展开图中已相邻）
DIRECT_EDGES = [
    ('A', 'top', 'D', 'bottom'),
    ('A', 'bottom', 'B', 'top'),
    ('A', 'left', 'C', 'right'),
    ('A', 'right', 'E', 'left'),
    ('E', 'right', 'F', 'left'),
]

# 折叠相连（展开图中不相邻，需折叠后相连）
FOLD_EDGES = [
    ('D', 'left', 'C', 'top'),
    ('D', 'right', 'E', 'top'),
    ('D', 'top', 'F', 'top'),
    ('C', 'bottom', 'B', 'left'),
    ('B', 'right', 'E', 'bottom'),
    ('B', 'bottom', 'F', 'bottom'),
    ('C', 'left', 'F', 'right'),
]

ALL_EDGES = DIRECT_EDGES + FOLD_EDGES


def extract_cross_points(face_info, edge, margin=35):
    """提取边缘处的年轮线交点在沿边方向的位置（0~1）"""
    img = face_info['image']
    ann = face_info['annotations']
    if img is None or ann is None:
        return []
    h, w = img.shape[:2]
    pts = []
    for shape in ann.get('shapes', []):
        if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
            points = np.array(shape['points'], dtype=np.float32)
            if edge == 'top':
                near = points[points[:, 1] < margin]
                for p in near:
                    pts.append(p[0] / w)
            elif edge == 'bottom':
                near = points[points[:, 1] > h - margin]
                for p in near:
                    pts.append(p[0] / w)
            elif edge == 'left':
                near = points[points[:, 0] < margin]
                for p in near:
                    pts.append(p[1] / h)
            elif edge == 'right':
                near = points[points[:, 0] > w - margin]
                for p in near:
                    pts.append(p[1] / h)
    return sorted(pts)


def compute_alignment(pts1, pts2):
    """计算两组交点位置的对齐分数"""
    if not pts1 or not pts2:
        return 0.0
    total = 0.0
    matched = set()
    for p1 in pts1:
        best_d = min((abs(p1 - p2) for p2 in pts2), default=1.0)
        if best_d < 0.15:
            total += 1.0 - best_d / 0.15
    for j, p2 in enumerate(pts2):
        if j not in matched:
            best_d = min((abs(p2 - p1) for p1 in pts1), default=1.0)
            if best_d < 0.15:
                total += 0.5 * (1.0 - best_d / 0.15)
    return min(1.0, total / max(len(pts1), len(pts2)))


# ==================== 详细边缘对比图 ====================
def create_edge_detail(r, c, faces, positions, face_size=380):
    """创建12条棱的边缘衔接对比图"""
    fig = plt.figure(figsize=(28, 22))
    fig.suptitle(f'立方体 ({r},{c}) — 12条棱边缘衔接检查', fontsize=16, fontweight='bold')

    # 左侧：展开图
    ax_unfold = fig.add_axes([0.01, 0.05, 0.32, 0.90])
    unfold_img, _ = create_unfold(faces, face_size)
    ax_unfold.imshow(unfold_img)
    ax_unfold.set_title('十字展开图', fontsize=11)
    ax_unfold.axis('off')

    # 右侧：12条棱分两组
    gs_right = GridSpec(12, 3, left=0.36, right=0.99, top=0.94, bottom=0.04,
                         hspace=0.5, wspace=0.15)

    for idx, (f1, e1, f2, e2) in enumerate(ALL_EDGES):
        row = idx
        is_direct = (f1, e1, f2, e2) in DIRECT_EDGES

        # 边缘1条带
        ax1 = fig.add_subplot(gs_right[row, 0])
        s1 = get_edge_strip(faces[f1], e1)
        ax1.imshow(s1)
        ax1.set_title(f'{f1}({FACE_CN[f1]}) {EDGE_CN[e1]}边', fontsize=9)
        ax1.axis('off')

        # 边缘2条带
        ax2 = fig.add_subplot(gs_right[row, 1])
        s2 = get_edge_strip(faces[f2], e2)
        ax2.imshow(s2)
        tag = '直接' if is_direct else '折叠'
        ax2.set_title(f'{f2}({FACE_CN[f2]}) {EDGE_CN[e2]}边 [{tag}]', fontsize=9)
        ax2.axis('off')

        # 对齐分析
        ax3 = fig.add_subplot(gs_right[row, 2])
        pts1 = extract_cross_points(faces[f1], e1)
        pts2 = extract_cross_points(faces[f2], e2)
        score = compute_alignment(pts1, pts2)

        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)

        # 画两个面的交点位置
        for p in pts1:
            ax3.plot(p, 0.75, 'o', color='#3366FF', markersize=5)
        for p in pts2:
            ax3.plot(p, 0.25, 'o', color='#FF6633', markersize=5)

        # 连线
        for p1 in pts1:
            best_p2 = min(pts2, key=lambda p2: abs(p1 - p2)) if pts2 else None
            if best_p2 and abs(p1 - best_p2) < 0.15:
                ax3.plot([p1, best_p2], [0.75, 0.25], '-', color='gray', alpha=0.5, lw=0.8)

        ax3.text(0.5, 0.75, f'{f1}({len(pts1)}条)', ha='center', va='bottom', fontsize=7, color='#3366FF')
        ax3.text(0.5, 0.25, f'{f2}({len(pts2)}条)', ha='center', va='top', fontsize=7, color='#FF6633')

        sc = 'green' if score > 0.5 else ('orange' if score > 0.25 else 'red')
        ax3.set_title(f'{score:.3f}', fontsize=9, color=sc, fontweight='bold')
        ax3.axis('off')

    output_path = os.path.join(OUTPUT_DIR, f'cube_r{r}c{c}_edge_detail.png')
    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path


# ==================== 主程序 ====================
def process_cube(r, c):
    print(f"\n  立方体 ({r},{c})...", end=' ')
    faces = load_cube(r, c)

    present = sum(1 for f in faces.values() if f['image'] is not None)
    annotated = sum(1 for f in faces.values() if f['annotations'] is not None)
    print(f'{present}/6面有图, {annotated}/6面有标注')

    # 展开图
    unfold_img, positions = create_unfold(faces)
    unfold_path = os.path.join(OUTPUT_DIR, f'cube_r{r}c{c}_unfold.png')
    cv2.imwrite(unfold_path, cv2.cvtColor(unfold_img, cv2.COLOR_RGB2BGR))

    # 边缘详图
    edge_path = create_edge_detail(r, c, faces, positions)

    # 提取各边交点统计
    edge_stats = {}
    for letter in CUBE_FACES:
        for edge in ['top', 'bottom', 'left', 'right']:
            pts = extract_cross_points(faces[letter], edge)
            if pts:
                edge_stats[(letter, edge)] = len(pts)

    # 棱对齐评分
    pair_scores = {}
    for f1, e1, f2, e2 in ALL_EDGES:
        pts1 = extract_cross_points(faces[f1], e1)
        pts2 = extract_cross_points(faces[f2], e2)
        score = compute_alignment(pts1, pts2)
        pair_scores[(f1, e1, f2, e2)] = (score, len(pts1), len(pts2))

    return faces, pair_scores, edge_stats


def print_summary(all_scores):
    """打印汇总"""
    print(f"\n{'='*70}")
    print("  所有立方体边缘衔接评分汇总")
    print(f"{'='*70}")

    for (r, c), pair_scores in sorted(all_scores.items()):
        if not pair_scores:
            continue
        avg = np.mean([s for s, _, _ in pair_scores.values()])
        n_good = sum(1 for s, _, _ in pair_scores.values() if s > 0.5)
        n_ok = sum(1 for s, _, _ in pair_scores.values() if 0.25 < s <= 0.5)
        n_bad = sum(1 for s, _, _ in pair_scores.values() if s <= 0.25)
        print(f"  ({r},{c}): 平均={avg:.3f}  好={n_good}  中={n_ok}  差={n_bad}")

    # 总体统计
    all_avgs = []
    for pair_scores in all_scores.values():
        if pair_scores:
            all_avgs.append(np.mean([s for s, _, _ in pair_scores.values()]))
    if all_avgs:
        print(f"\n  总体平均: {np.mean(all_avgs):.3f}")


def main():
    print("=" * 70)
    print("  木块立方体展开图生成工具 v2")
    print("  A=前面 B=顶面 C=左面 D=底面 E=右面 F=后面")
    print("=" * 70)

    all_scores = {}

    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            if (r, c) in EMPTY_POSITIONS:
                print(f"\n  位置 ({r},{c}) 为空，跳过")
                continue
            faces, pair_scores, edge_stats = process_cube(r, c)
            all_scores[(r, c)] = pair_scores

    print_summary(all_scores)

    print(f"\n  所有结果: {OUTPUT_DIR}")
    print(f"  - cube_r*c*_unfold.png: 展开图 (19张)")
    print(f"  - cube_r*c*_edge_detail.png: 12条棱详图 (19张)")


if __name__ == '__main__':
    main()
