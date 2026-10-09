"""
边缘衔接检查工具 v2
通过分析年轮标注线在边缘处的交点位置和对齐情况，
检查木块立方体各面之间的衔接质量。

改进：
- 使用 Microsoft YaHei 中文字体
- 基于标注线（而非原始像素）进行边缘比对
- 提取年轮线在边缘处的交点位置和角度
- 生成清晰的可视化对比图
"""

import cv2
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ==================== 中文字体设置 ====================
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
BASE_DIR = r'D:\教务处实习\木材数据1\立方体'
OUTPUT_DIR = r'C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block\edge_check_output_v2'
CUBES = ['A', 'B', 'C', 'D', 'E', 'F']
ROWS, COLS = 5, 4
EDGE_MARGIN = 40  # 判断标注点是否在边缘附近的阈值（像素）

EDGE_CN = {'top': '上边', 'bottom': '下边', 'left': '左边', 'right': '右边'}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 数据加载 ====================
def load_block(cube, face, r, c):
    """加载单个木块图片和标注"""
    block_dir = os.path.join(BASE_DIR, f'{cube}_{face}_blocks')
    prefix = f'{cube}_{face}_{r}{c}'

    img_path = os.path.join(block_dir, f'{prefix}.jpg')
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    annotations = None
    json_path = os.path.join(block_dir, f'{prefix}.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

    return img, annotations


def load_face(cube, face):
    """加载一个面的所有木块"""
    blocks = {}
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            img, annotations = load_block(cube, face, r, c)
            blocks[(r, c)] = {
                'image': img,
                'annotations': annotations,
            }
    return blocks


# ==================== 标注线边缘交点分析 ====================
def extract_ring_lines(annotations):
    """
    从标注中提取年轮线条
    返回: list of (points_array, label)
    """
    if annotations is None:
        return []
    lines = []
    for shape in annotations.get('shapes', []):
        if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
            points = np.array(shape['points'], dtype=np.float32)
            if len(points) >= 2:
                lines.append(points)
    return lines


def find_edge_intersections(ring_lines, img_shape, edge, margin=EDGE_MARGIN):
    """
    找到年轮线与指定边缘的交点

    参数:
        ring_lines: 年轮线列表
        img_shape: (h, w) 图片尺寸
        edge: 'top' | 'bottom' | 'left' | 'right'
        margin: 判定为"在边缘附近"的像素阈值

    返回:
        intersections: list of dicts [{position, angle, points}, ...]
        其中 position 是沿边缘方向的位置（归一化到0-1）
        angle 是线与边缘法线的夹角
    """
    h, w = img_shape[:2]
    intersections = []

    for line_points in ring_lines:
        # 查找线段与边缘的交点
        near_edge_points = []

        for i in range(len(line_points)):
            px, py = line_points[i]

            if edge == 'top' and py < margin:
                near_edge_points.append((px, py, 'top', py))
            elif edge == 'bottom' and py > h - margin:
                near_edge_points.append((px, py, 'bottom', h - py))
            elif edge == 'left' and px < margin:
                near_edge_points.append((px, py, 'left', px))
            elif edge == 'right' and px > w - margin:
                near_edge_points.append((px, py, 'right', w - px))

        if len(near_edge_points) >= 1:
            # 取最靠近边缘的那个点
            best = min(near_edge_points, key=lambda p: p[3])
            px, py, _, _ = best

            # 沿边缘方向的位置
            if edge in ['top', 'bottom']:
                edge_pos = px / w  # 归一化到0-1
            else:
                edge_pos = py / h  # 归一化到0-1

            # 计算线与边缘法线的夹角
            # 找附近的线段来计算角度
            angle = compute_line_angle_at_edge(line_points, edge, margin)

            intersections.append({
                'edge_pos': edge_pos,
                'angle': angle,
                'point': (px, py),
            })

    return intersections


def compute_line_angle_at_edge(line_points, edge, margin):
    """
    计算年轮线在边缘处的方向角
    返回相对于边缘法线的角度（度）
    对于水平边缘(top/bottom): 0度=垂直方向
    对于垂直边缘(left/right): 0度=水平方向
    """
    if len(line_points) < 2:
        return 0.0

    # 找靠近边缘的线段
    best_segment = None
    best_dist = float('inf')

    for i in range(len(line_points) - 1):
        p1 = line_points[i]
        p2 = line_points[i + 1]

        if edge == 'top':
            dist = min(p1[1], p2[1])
        elif edge == 'bottom':
            dist = min(abs(p1[1]), abs(p2[1]))
            # Actually we need image height for this
            continue  # skip for now, handle separately
        elif edge == 'left':
            dist = min(p1[0], p2[0])
        elif edge == 'right':
            dist = float('inf')  # 需要图片宽度

        if edge in ['top', 'left'] and dist < margin:
            if dist < best_dist:
                best_dist = dist
                best_segment = (p1, p2)
        elif edge == 'bottom':
            # 需要图片高度的上下文，暂时跳过
            pass
        elif edge == 'right':
            # 需要图片宽度的上下文，暂时跳过
            pass

    if best_segment is None:
        # 使用最后两个点
        p1 = line_points[-2]
        p2 = line_points[-1]
    else:
        p1, p2 = best_segment

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if edge in ['top', 'bottom']:
        # 水平边缘：0度=垂直，计算与垂直方向的偏差
        angle = np.degrees(np.arctan2(dx, -dy if edge == 'top' else dy))
    else:
        # 垂直边缘：0度=水平，计算与水平方向的偏差
        angle = np.degrees(np.arctan2(dy if edge == 'left' else -dy, dx))

    return angle


# ==================== 全脸组装 ====================
def assemble_face_image(blocks):
    """将5×4的木块组装成完整的面图像"""
    # 获取基准尺寸（使用最大高度和宽度确保一致性）
    max_h = max(b['image'].shape[0] for b in blocks.values() if b['image'] is not None)
    max_w = max(b['image'].shape[1] for b in blocks.values() if b['image'] is not None)

    canvas_h = ROWS * max_h
    canvas_w = COLS * max_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    block_h = max_h
    block_w = max_w

    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            block = blocks.get((r, c))
            if block is None or block['image'] is None:
                continue
            img = block['image']
            y = (r - 1) * block_h
            x = (c - 1) * block_w
            # 居中放置
            y_off = (block_h - img.shape[0]) // 2
            x_off = (block_w - img.shape[1]) // 2
            canvas[y + y_off:y + y_off + img.shape[0],
                   x + x_off:x + x_off + img.shape[1]] = img

    return canvas, block_h, block_w


def collect_all_ring_intersections(blocks, edge, margin=EDGE_MARGIN):
    """
    收集一个面指定边缘上所有木块的年轮线交点
    返回沿边缘方向的交点位置列表
    """
    all_intersections = []

    if edge == 'top':
        for c in range(1, COLS + 1):
            block = blocks.get((1, c))
            if block and block['annotations'] and block['image'] is not None:
                ring_lines = extract_ring_lines(block['annotations'])
                img_shape = block['image'].shape
                intersections = find_edge_intersections(ring_lines, img_shape, 'top', margin)
                # 调整沿边缘位置：加上块在列中的偏移
                for inter in intersections:
                    inter['edge_pos'] = (inter['edge_pos'] + (c - 1)) / COLS
                    inter['block_col'] = c
                    inter['block_row'] = 1
                all_intersections.extend(intersections)

    elif edge == 'bottom':
        for c in range(1, COLS + 1):
            block = blocks.get((ROWS, c))
            if block and block['annotations'] and block['image'] is not None:
                ring_lines = extract_ring_lines(block['annotations'])
                img_shape = block['image'].shape
                intersections = find_edge_intersections(ring_lines, img_shape, 'bottom', margin)
                for inter in intersections:
                    inter['edge_pos'] = (inter['edge_pos'] + (c - 1)) / COLS
                    inter['block_col'] = c
                    inter['block_row'] = ROWS
                all_intersections.extend(intersections)

    elif edge == 'left':
        for r in range(1, ROWS + 1):
            block = blocks.get((r, 1))
            if block and block['annotations'] and block['image'] is not None:
                ring_lines = extract_ring_lines(block['annotations'])
                img_shape = block['image'].shape
                intersections = find_edge_intersections(ring_lines, img_shape, 'left', margin)
                for inter in intersections:
                    inter['edge_pos'] = (inter['edge_pos'] + (r - 1)) / ROWS
                    inter['block_col'] = 1
                    inter['block_row'] = r
                all_intersections.extend(intersections)

    elif edge == 'right':
        for r in range(1, ROWS + 1):
            block = blocks.get((r, COLS))
            if block and block['annotations'] and block['image'] is not None:
                ring_lines = extract_ring_lines(block['annotations'])
                img_shape = block['image'].shape
                intersections = find_edge_intersections(ring_lines, img_shape, 'right', margin)
                for inter in intersections:
                    inter['edge_pos'] = (inter['edge_pos'] + (r - 1)) / ROWS
                    inter['block_col'] = COLS
                    inter['block_row'] = r
                all_intersections.extend(intersections)

    return sorted(all_intersections, key=lambda x: x['edge_pos'])


# ==================== 可视化 ====================
def create_face_edge_overview(cube, faces):
    """
    创建每个面的边缘年轮线交点概览图
    展示三个面各自的四个边缘上有哪些年轮线交点
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    fig.suptitle(f'木块 {cube} — 各面边缘年轮线交点分布', fontsize=16, fontweight='bold')

    for face_idx, face_name in enumerate(['1', '2', '3']):
        ax = axes[face_idx]
        blocks = faces[face_name]

        # 组装面图像
        face_img, bh, bw = assemble_face_image(blocks)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # 缩小以适应
        scale = min(1000 / max(face_rgb.shape[:2]), 1.0)
        new_h, new_w = int(face_rgb.shape[0] * scale), int(face_rgb.shape[1] * scale)
        face_rgb_small = cv2.resize(face_rgb, (new_w, new_h))

        ax.imshow(face_rgb_small)
        ax.set_title(f'面 {face_name}', fontsize=14)
        ax.axis('off')

        # 在每个边缘标注年轮线交点
        edge_colors = {'top': '#FF4444', 'bottom': '#FF8800',
                       'left': '#00CCFF', 'right': '#FFDD00'}

        for edge in ['top', 'bottom', 'left', 'right']:
            intersections = collect_all_ring_intersections(blocks, edge)

            for inter in intersections:
                ep = inter['edge_pos']
                if edge == 'top':
                    x = ep * new_w
                    y = 0
                elif edge == 'bottom':
                    x = ep * new_w
                    y = new_h
                elif edge == 'left':
                    x = 0
                    y = ep * new_h
                elif edge == 'right':
                    x = new_w
                    y = ep * new_h

                ax.plot(x, y, 'o', color=edge_colors[edge], markersize=8,
                        markeredgecolor='white', markeredgewidth=1.5, zorder=10)

        # 图例
        legend_elements = [
            mpatches.Patch(color=c, label=f'{EDGE_CN[e]} ({len(collect_all_ring_intersections(blocks, e))}个交点)')
            for e, c in edge_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
                  framealpha=0.8)

    output_path = os.path.join(OUTPUT_DIR, f'{cube}_face_edge_overview.png')
    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  面边缘交点图已保存: {output_path}")
    return output_path


def create_alignment_check_figure(cube, faces):
    """
    创建详细的边缘衔接检查图
    对于三对面，展示：
    1. 两个面的边缘放大图（并排）
    2. 年轮线交点位置的对比
    3. 衔接评分
    """
    fig = plt.figure(figsize=(28, 18))
    fig.suptitle(f'木块 {cube} — 边缘衔接详细检查', fontsize=16, fontweight='bold')

    gs = GridSpec(3, 5, figure=fig, hspace=0.4, wspace=0.3,
                  top=0.93, bottom=0.05, left=0.03, right=0.97)

    face_pairs = [('1', '2'), ('1', '3'), ('2', '3')]

    for row_idx, (f1, f2) in enumerate(face_pairs):
        blocks1 = faces[f1]
        blocks2 = faces[f2]

        # ===== 列1: 面1的四个边缘的年轮线交点 =====
        ax_edges1 = fig.add_subplot(gs[row_idx, 0])
        visualize_face_edges(ax_edges1, blocks1, f'面{f1}各边缘交点')

        # ===== 列2: 面2的四个边缘的年轮线交点 =====
        ax_edges2 = fig.add_subplot(gs[row_idx, 1])
        visualize_face_edges(ax_edges2, blocks2, f'面{f2}各边缘交点')

        # ===== 列3-5: 最佳边缘配对对比 =====
        # 找出面f1和面f2之间所有边缘配对的评分
        pair_scores = []
        for e1 in ['top', 'bottom', 'left', 'right']:
            for e2 in ['top', 'bottom', 'left', 'right']:
                inter1 = collect_all_ring_intersections(blocks1, e1)
                inter2 = collect_all_ring_intersections(blocks2, e2)
                if inter1 and inter2:
                    score = compute_intersection_alignment(inter1, inter2)
                    pair_scores.append((e1, e2, score, inter1, inter2))

        pair_scores.sort(key=lambda x: -x[2])

        # 展示前3个最佳配对
        for sub_idx in range(min(3, len(pair_scores))):
            e1, e2, score, inter1, inter2 = pair_scores[sub_idx]
            ax = fig.add_subplot(gs[row_idx, 2 + sub_idx])

            # 绘制交点位置对比
            plot_intersection_comparison(ax, inter1, inter2, e1, e2, score, f1, f2)

            if sub_idx == 0 and score > 0.3:
                ax.patch.set_facecolor('#e8ffe8')  # 浅绿底色表示可能匹配

        if len(pair_scores) == 0:
            ax_empty = fig.add_subplot(gs[row_idx, 2])
            ax_empty.text(0.5, 0.5, '无标注数据\n无法比较', ha='center', va='center',
                          transform=ax_empty.transAxes, fontsize=14, color='gray')
            ax_empty.axis('off')

    output_path = os.path.join(OUTPUT_DIR, f'{cube}_alignment_check.png')
    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  边缘衔接检查图已保存: {output_path}")
    return output_path


def visualize_face_edges(ax, blocks, title):
    """在子图中展示一个面的四个边缘的交点分布"""
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    edge_configs = {
        'top':    {'pos': 1.05, 'orient': 'h', 'color': '#FF4444'},
        'bottom': {'pos': -0.05, 'orient': 'h', 'color': '#FF8800'},
        'left':   {'pos': -0.05, 'orient': 'v', 'color': '#00CCFF'},
        'right':  {'pos': 1.05, 'orient': 'v', 'color': '#FFDD00'},
    }

    for edge, config in edge_configs.items():
        intersections = collect_all_ring_intersections(blocks, edge)
        for inter in intersections:
            ep = inter['edge_pos']
            if config['orient'] == 'h':
                ax.plot(ep, config['pos'], 'o', color=config['color'],
                        markersize=10, markeredgecolor='white', markeredgewidth=1)
            else:
                ax.plot(config['pos'], ep, 'o', color=config['color'],
                        markersize=10, markeredgecolor='white', markeredgewidth=1)

        # 画边缘线
        if config['orient'] == 'h':
            ax.axhline(y=config['pos'], color=config['color'], linewidth=2, alpha=0.3)
        else:
            ax.axvline(x=config['pos'], color=config['color'], linewidth=2, alpha=0.3)

    # 面区域
    rect = plt.Rectangle((0, 0), 1, 1, fill=True, facecolor='lightgray', alpha=0.15, zorder=0)
    ax.add_patch(rect)

    ax.set_title(title, fontsize=10)
    ax.axis('off')


def plot_intersection_comparison(ax, inter1, inter2, e1, e2, score, f1, f2):
    """
    绘制两个边缘的年轮线交点位置对比
    左半边显示面1的edge分布，右半边显示面2的edge分布
    用连线表示匹配
    """
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.05)

    # 左半边：面1
    for inter in inter1:
        ax.plot(0.15, inter['edge_pos'], 'o', color='#3366FF', markersize=8,
                markeredgecolor='white', markeredgewidth=1, zorder=5)
        ax.annotate(f"{inter['edge_pos']:.2f}", (0.15, inter['edge_pos']),
                    textcoords="offset points", xytext=(-35, 0),
                    fontsize=7, color='#3366FF', ha='right', va='center')

    # 右半边：面2
    for inter in inter2:
        ax.plot(0.85, inter['edge_pos'], 'o', color='#FF6633', markersize=8,
                markeredgecolor='white', markeredgewidth=1, zorder=5)
        ax.annotate(f"{inter['edge_pos']:.2f}", (0.85, inter['edge_pos']),
                    textcoords="offset points", xytext=(35, 0),
                    fontsize=7, color='#FF6633', ha='left', va='center')

    # 连线表示最近邻匹配
    for i1 in inter1:
        best_i2 = min(inter2, key=lambda i2: abs(i1['edge_pos'] - i2['edge_pos']))
        dist = abs(i1['edge_pos'] - best_i2['edge_pos'])
        if dist < 0.15:
            alpha = max(0.2, 1.0 - dist * 5)
            ax.plot([0.15, 0.85], [i1['edge_pos'], best_i2['edge_pos']],
                    '-', color='gray', alpha=alpha, linewidth=1)

    # 边缘标记
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.text(0.15, -0.03, f'面{f1}\n{EDGE_CN[e1]}', ha='center', fontsize=8,
            color='#3366FF', fontweight='bold')
    ax.text(0.85, -0.03, f'面{f2}\n{EDGE_CN[e2]}', ha='center', fontsize=8,
            color='#FF6633', fontweight='bold')

    score_color = 'green' if score > 0.5 else ('orange' if score > 0.3 else 'red')
    ax.set_title(f'匹配评分: {score:.3f}', fontsize=10, color=score_color, fontweight='bold')
    ax.axis('off')


def compute_intersection_alignment(inter1, inter2, max_dist=0.15):
    """
    计算两个边缘的年轮线交点对齐分数
    使用最近邻匹配，距离越近分数越高

    返回: 0-1之间的对齐分数
    """
    if not inter1 or not inter2:
        return 0.0

    # 对inter1中的每个交点，找inter2中最近的交点
    total_score = 0.0
    matched = set()

    for i1 in inter1:
        best_dist = float('inf')
        best_idx = -1
        for idx, i2 in enumerate(inter2):
            dist = abs(i1['edge_pos'] - i2['edge_pos'])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_dist < max_dist:
            total_score += 1.0 - (best_dist / max_dist)
            matched.add(best_idx)

    # 也考虑反向匹配
    for idx, i2 in enumerate(inter2):
        if idx not in matched:
            best_dist = float('inf')
            for i1 in inter1:
                dist = abs(i1['edge_pos'] - i2['edge_pos'])
                if dist < best_dist:
                    best_dist = dist
            if best_dist < max_dist:
                total_score += 0.5 * (1.0 - (best_dist / max_dist))

    max_possible = max(len(inter1), len(inter2))
    if max_possible == 0:
        return 0.0

    return min(1.0, total_score / max_possible)


def create_side_by_side_edge_comparison(cube, faces):
    """
    创建并排边缘对比图
    将推测对应的边缘放大显示，方便肉眼检查年轮衔接
    """
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f'木块 {cube} — 边缘并排对比（用于肉眼检查年轮衔接）', fontsize=16, fontweight='bold')

    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3,
                  top=0.92, bottom=0.05, left=0.04, right=0.96)

    face_pairs = [('1', '2'), ('1', '3'), ('2', '3')]

    for row_idx, (f1, f2) in enumerate(face_pairs):
        blocks1 = faces[f1]
        blocks2 = faces[f2]

        # 找最佳边缘配对
        best_pair = None
        best_score = -1
        for e1 in ['top', 'bottom', 'left', 'right']:
            for e2 in ['top', 'bottom', 'left', 'right']:
                inter1 = collect_all_ring_intersections(blocks1, e1)
                inter2 = collect_all_ring_intersections(blocks2, e2)
                if inter1 and inter2:
                    score = compute_intersection_alignment(inter1, inter2)
                    if score > best_score:
                        best_score = score
                        best_pair = (e1, e2, inter1, inter2)

        if best_pair:
            e1, e2, inter1, inter2 = best_pair

            # 列1: 面1的edge
            ax1 = fig.add_subplot(gs[row_idx, 0])
            edge_img1 = extract_edge_image(blocks1, e1)
            if edge_img1 is not None:
                edge_rgb = cv2.cvtColor(edge_img1, cv2.COLOR_BGR2RGB)
                ax1.imshow(edge_rgb)
                # 标注交点
                for inter in inter1:
                    if e1 in ['top', 'bottom']:
                        x = inter['edge_pos'] * edge_rgb.shape[1]
                        ax1.axvline(x=x, color='lime', linewidth=2, alpha=0.8)
                    else:
                        y = inter['edge_pos'] * edge_rgb.shape[0]
                        ax1.axhline(y=y, color='lime', linewidth=2, alpha=0.8)
            ax1.set_title(f'面{f1} {EDGE_CN[e1]} ({len(inter1)}条年轮线)', fontsize=10)
            ax1.axis('off')

            # 列2: 面2的edge
            ax2 = fig.add_subplot(gs[row_idx, 1])
            edge_img2 = extract_edge_image(blocks2, e2)
            if edge_img2 is not None:
                edge_rgb2 = cv2.cvtColor(edge_img2, cv2.COLOR_BGR2RGB)
                ax2.imshow(edge_rgb2)
                for inter in inter2:
                    if e2 in ['top', 'bottom']:
                        x = inter['edge_pos'] * edge_rgb2.shape[1]
                        ax2.axvline(x=x, color='lime', linewidth=2, alpha=0.8)
                    else:
                        y = inter['edge_pos'] * edge_rgb2.shape[0]
                        ax2.axhline(y=y, color='lime', linewidth=2, alpha=0.8)
            ax2.set_title(f'面{f2} {EDGE_CN[e2]} ({len(inter2)}条年轮线)', fontsize=10)
            ax2.axis('off')

            # 列3: 交点位置对齐图
            ax3 = fig.add_subplot(gs[row_idx, 2])
            plot_edge_alignment_diagram(ax3, inter1, inter2, e1, e2, f1, f2, best_score)

            # 列4: 注释文本
            ax4 = fig.add_subplot(gs[row_idx, 3])
            ax4.axis('off')
            text = f"面{f1}的{EDGE_CN[e1]} ↔ 面{f2}的{EDGE_CN[e2]}\n\n"
            text += f"交点数量:\n  面{f1}: {len(inter1)}\n  面{f2}: {len(inter2)}\n\n"
            text += f"对齐评分: {best_score:.3f}\n\n"

            if best_score > 0.5:
                text += "✓ 对齐较好"
                color = 'green'
            elif best_score > 0.3:
                text += "△ 对齐一般"
                color = 'orange'
            else:
                text += "✗ 对齐较差"
                color = 'red'

            text += "\n\n年轮线交点位置:\n"
            ep1 = [f"{i['edge_pos']:.3f}" for i in inter1[:5]]
            ep2 = [f"{i['edge_pos']:.3f}" for i in inter2[:5]]
            text += f"面{f1}: {', '.join(ep1)}{'...' if len(inter1) > 5 else ''}\n"
            text += f"面{f2}: {', '.join(ep2)}{'...' if len(inter2) > 5 else ''}"

            ax4.text(0.1, 0.9, text, transform=ax4.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax_empty = fig.add_subplot(gs[row_idx, 0:4])
            ax_empty.text(0.5, 0.5, f'面{f1}和面{f2}的标注数据不足以进行边缘比较',
                          ha='center', va='center', fontsize=12, color='gray')
            ax_empty.axis('off')

    output_path = os.path.join(OUTPUT_DIR, f'{cube}_side_by_side.png')
    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  并排对比图已保存: {output_path}")
    return output_path


def extract_edge_image(blocks, edge, strip_width=100):
    """提取指定边缘的图像条带"""
    strips = []

    if edge == 'top':
        for c in range(1, COLS + 1):
            block = blocks.get((1, c))
            if block and block['image'] is not None:
                img = block['image']
                strips.append(img[:strip_width, :, :])

    elif edge == 'bottom':
        for c in range(1, COLS + 1):
            block = blocks.get((ROWS, c))
            if block and block['image'] is not None:
                img = block['image']
                strips.append(img[-strip_width:, :, :])

    elif edge == 'left':
        for r in range(1, ROWS + 1):
            block = blocks.get((r, 1))
            if block and block['image'] is not None:
                img = block['image']
                strips.append(img[:, :strip_width, :])

    elif edge == 'right':
        for r in range(1, ROWS + 1):
            block = blocks.get((r, COLS))
            if block and block['image'] is not None:
                img = block['image']
                strips.append(img[:, -strip_width:, :])

    if not strips:
        return None

    if edge in ['top', 'bottom']:
        max_h = max(s.shape[0] for s in strips)
        padded = []
        for s in strips:
            if s.shape[0] < max_h:
                s = np.pad(s, ((0, max_h - s.shape[0]), (0, 0), (0, 0)))
            padded.append(s)
        return np.hstack(padded)
    else:
        max_w = max(s.shape[1] for s in strips)
        padded = []
        for s in strips:
            if s.shape[1] < max_w:
                s = np.pad(s, ((0, 0), (0, max_w - s.shape[1]), (0, 0)))
            padded.append(s)
        return np.vstack(padded)


def plot_edge_alignment_diagram(ax, inter1, inter2, e1, e2, f1, f2, score):
    """绘制对齐示意图"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 两个面的交点位置
    for inter in inter1:
        ax.plot(inter['edge_pos'], 0.7, 'o', color='#3366FF', markersize=8)
    for inter in inter2:
        ax.plot(inter['edge_pos'], 0.3, 'o', color='#FF6633', markersize=8)

    # 最近邻连线
    for i1 in inter1:
        best_i2 = min(inter2, key=lambda i2: abs(i1['edge_pos'] - i2['edge_pos']))
        dist = abs(i1['edge_pos'] - best_i2['edge_pos'])
        alpha = max(0.1, 1.0 - dist * 5)
        ax.plot([i1['edge_pos'], best_i2['edge_pos']], [0.7, 0.3],
                '-', color='gray', alpha=alpha, linewidth=1.5)

    ax.text(0.5, 0.7, f'面{f1} {EDGE_CN[e1]}', ha='center', va='bottom',
            fontsize=8, color='#3366FF', fontweight='bold')
    ax.text(0.5, 0.3, f'面{f2} {EDGE_CN[e2]}', ha='center', va='top',
            fontsize=8, color='#FF6633', fontweight='bold')

    score_color = 'green' if score > 0.5 else ('orange' if score > 0.3 else 'red')
    ax.set_title(f'对齐评分: {score:.3f}', fontsize=9, color=score_color, fontweight='bold')
    ax.axis('off')


def create_summary_report(all_results):
    """
    创建所有木块的汇总比对报告
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('所有木块边缘衔接评分汇总', fontsize=18, fontweight='bold')

    for idx, (cube, results) in enumerate(all_results.items()):
        ax = axes[idx // 3, idx % 3]
        face_pair_scores = results  # dict: (f1,f2) -> best_score

        if not face_pair_scores:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.axis('off')
            continue

        labels = []
        scores = []
        colors = []
        for (f1, f2), score in face_pair_scores.items():
            labels.append(f'面{f1} ↔ 面{f2}')
            scores.append(score)
            colors.append('green' if score > 0.5 else ('orange' if score > 0.3 else 'red'))

        bars = ax.bar(labels, scores, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('对齐评分', fontsize=10)
        ax.set_title(f'木块 {cube}', fontsize=14)

        # 标注数值
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')

        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.4, label='较好')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.4, label='一般')
        ax.legend(fontsize=8)

    output_path = os.path.join(OUTPUT_DIR, 'summary_report.png')
    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n汇总报告已保存: {output_path}")


def print_terminal_report(cube, face_pair_scores, faces):
    """在终端输出详细的文字报告"""
    print(f"\n{'='*60}")
    print(f"  木块 {cube} 边缘衔接分析报告")
    print(f"{'='*60}")

    for face_name in ['1', '2', '3']:
        blocks = faces[face_name]
        annotated = sum(1 for b in blocks.values() if b['annotations'] is not None)
        print(f"\n  面 {face_name}:")
        for edge in ['top', 'bottom', 'left', 'right']:
            inter = collect_all_ring_intersections(blocks, edge)
            if inter:
                positions = [f"{i['edge_pos']:.3f}" for i in inter]
                print(f"    {EDGE_CN[edge]}: {len(inter)}条年轮线, 交点位置={positions}")
            else:
                print(f"    {EDGE_CN[edge]}: 无年轮线交点")

    print(f"\n  面对之间的最佳边缘匹配:")
    for (f1, f2), score in sorted(face_pair_scores.items()):
        score_color = '✓' if score > 0.5 else ('△' if score > 0.3 else '✗')
        print(f"    面{f1} ↔ 面{f2}: {score_color} 对齐评分={score:.3f}")


# ==================== 主分析函数 ====================
def analyze_cube(cube):
    """分析一个立方体的边缘衔接情况"""
    print(f"\n{'='*60}")
    print(f"  分析木块 {cube}")
    print(f"{'='*60}")

    # 加载三个面
    faces = {}
    for face_name in ['1', '2', '3']:
        blocks = load_face(cube, face_name)
        faces[face_name] = blocks
        annotated = sum(1 for b in blocks.values() if b['annotations'] is not None)
        print(f"  面 {face_name}: {annotated}/{len(blocks)} 个木块有标注")

    # 对每对面，找最佳边缘匹配
    face_pairs = [('1', '2'), ('1', '3'), ('2', '3')]
    face_pair_scores = {}

    for f1, f2 in face_pairs:
        blocks1 = faces[f1]
        blocks2 = faces[f2]
        best_score = 0
        best_pair = None

        for e1 in ['top', 'bottom', 'left', 'right']:
            for e2 in ['top', 'bottom', 'left', 'right']:
                inter1 = collect_all_ring_intersections(blocks1, e1)
                inter2 = collect_all_ring_intersections(blocks2, e2)
                if inter1 and inter2:
                    score = compute_intersection_alignment(inter1, inter2)
                    if score > best_score:
                        best_score = score
                        best_pair = (e1, e2)

        face_pair_scores[(f1, f2)] = best_score
        if best_pair:
            print(f"  面{f1} ↔ 面{f2}: {EDGE_CN[best_pair[0]]} ↔ {EDGE_CN[best_pair[1]]} (评分: {best_score:.3f})")

    # 生成可视化
    create_face_edge_overview(cube, faces)
    create_alignment_check_figure(cube, faces)
    create_side_by_side_edge_comparison(cube, faces)

    print_terminal_report(cube, face_pair_scores, faces)

    return face_pair_scores


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("  木块立方体边缘衔接检查工具 v2")
    print("  基于年轮标注线的交点分析")
    print("=" * 60)
    print(f"  数据目录: {BASE_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")

    all_results = {}

    for cube in CUBES:
        try:
            face_pair_scores = analyze_cube(cube)
            all_results[cube] = face_pair_scores
        except Exception as e:
            print(f"  [错误] 处理木块 {cube} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 汇总报告
    if all_results:
        create_summary_report(all_results)

    print(f"\n{'='*60}")
    print(f"  检查完成！所有结果已保存到: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
