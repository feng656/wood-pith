"""
边缘衔接检查工具
检查木块立方体各面之间年轮线条在边缘处的衔接情况

对每个立方体（A-F），将三个面的边缘提取出来，
通过图像相关性自动匹配对应的边缘，生成对比可视化。
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.ndimage import rotate as scipy_rotate

# ==================== 配置 ====================
BASE_DIR = r'D:\教务处实习\木材数据1\立方体'
OUTPUT_DIR = r'C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block\edge_check_output'
CUBES = ['A', 'B', 'C', 'D', 'E', 'F']
ROWS, COLS = 5, 4
EDGE_STRIP_WIDTH = 80  # 提取边缘的像素宽度

# 边缘中文名
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
        return None, None, None

    annotations = None
    json_path = os.path.join(block_dir, f'{prefix}.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

    return img, annotations, img_path


def load_face(cube, face):
    """加载一个面的所有木块"""
    blocks = {}
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            img, annotations, img_path = load_block(cube, face, r, c)
            blocks[(r, c)] = {
                'image': img,
                'annotations': annotations,
                'path': img_path,
            }
    return blocks


# ==================== 边缘提取 ====================
def get_edge_strip(blocks, edge, strip_width=EDGE_STRIP_WIDTH):
    """
    从木块网格中提取指定边缘的像素条带
    edge: 'top' | 'bottom' | 'left' | 'right'

    返回:
        strip_img: 拼接后的边缘图像
        block_boundaries: 各个木块边界的像素位置列表
    """
    strips = []
    boundaries = [0]

    if edge == 'top':
        for c in range(1, COLS + 1):
            block = blocks.get((1, c))
            if block and block['image'] is not None:
                img = block['image']
                h = min(strip_width, img.shape[0])
                strip = img[:h, :, :]
                strips.append(strip)
                boundaries.append(boundaries[-1] + img.shape[1])

    elif edge == 'bottom':
        for c in range(1, COLS + 1):
            block = blocks.get((ROWS, c))
            if block and block['image'] is not None:
                img = block['image']
                h = min(strip_width, img.shape[0])
                strip = img[-h:, :, :]
                strips.append(strip)
                boundaries.append(boundaries[-1] + img.shape[1])

    elif edge == 'left':
        for r in range(1, ROWS + 1):
            block = blocks.get((r, 1))
            if block and block['image'] is not None:
                img = block['image']
                w = min(strip_width, img.shape[1])
                strip = img[:, :w, :]
                strips.append(strip)
                boundaries.append(boundaries[-1] + img.shape[0])

    elif edge == 'right':
        for r in range(1, ROWS + 1):
            block = blocks.get((r, COLS))
            if block and block['image'] is not None:
                img = block['image']
                w = min(strip_width, img.shape[1])
                strip = img[:, -w:, :]
                strips.append(strip)
                boundaries.append(boundaries[-1] + img.shape[0])

    if not strips:
        return None, None

    if edge in ['top', 'bottom']:
        # 水平边缘：横向拼接
        max_h = max(s.shape[0] for s in strips)
        padded = []
        for s in strips:
            if s.shape[0] < max_h:
                pad = np.zeros((max_h - s.shape[0], s.shape[1], 3), dtype=np.uint8)
                s = np.vstack([s, pad])
            padded.append(s)
        strip_img = np.hstack(padded)
    else:
        # 垂直边缘：纵向拼接
        max_w = max(s.shape[1] for s in strips)
        padded = []
        for s in strips:
            if s.shape[1] < max_w:
                pad = np.zeros((s.shape[0], max_w - s.shape[1], 3), dtype=np.uint8)
                s = np.hstack([s, pad])
            padded.append(s)
        strip_img = np.vstack(padded)

    return strip_img, boundaries


def compute_edge_profile(edge_img):
    """
    计算边缘图像的1D特征曲线（用于比对）
    对水平边缘：沿横向的平均像素值
    对垂直边缘：沿纵向的平均像素值
    """
    if edge_img is None:
        return None

    gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)

    if edge_img.shape[0] < edge_img.shape[1]:
        # 水平条带 → 纵向平均
        profile = gray.mean(axis=0)
    else:
        # 垂直条带 → 横向平均
        profile = gray.mean(axis=1)

    return profile


# ==================== 边缘比对 ====================
def compare_profiles(p1, p2, p2_reversed=False):
    """
    比较两个1D特征曲线的相似度
    使用归一化互相关
    """
    if p1 is None or p2 is None:
        return 0.0

    if p2_reversed:
        p2 = p2[::-1]

    # 归一化
    p1_norm = (p1 - np.mean(p1)) / (np.std(p1) + 1e-8)
    p2_norm = (p2 - np.mean(p2)) / (np.std(p2) + 1e-8)

    # 重采样到相同长度
    target_len = min(len(p1_norm), len(p2_norm))
    x_old = np.linspace(0, 1, len(p1_norm))
    x_new = np.linspace(0, 1, target_len)
    p1_resampled = np.interp(x_new, x_old, p1_norm)

    x_old = np.linspace(0, 1, len(p2_norm))
    p2_resampled = np.interp(x_new, x_old, p2_norm)

    # 皮尔逊相关系数
    corr = np.corrcoef(p1_resampled, p2_resampled)[0, 1]
    return max(0, corr)  # 取绝对值


def compare_edge_images(img1, img2, flip=False):
    """
    使用SSIM类方法比较两个边缘图像
    """
    if img1 is None or img2 is None:
        return 0.0

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if flip:
        # 尝试翻转（对垂直边缘可能需要上下翻转）
        gray2 = np.flip(gray2, axis=0)

    # 缩放到相同尺寸
    target_h = min(gray1.shape[0], gray2.shape[0])
    target_w = min(gray1.shape[1], gray2.shape[1])
    gray1 = cv2.resize(gray1, (target_w, target_h))
    gray2 = cv2.resize(gray2, (target_w, target_h))

    # 使用归一化互相关
    g1_norm = (gray1 - gray1.mean()) / (gray1.std() + 1e-8)
    g2_norm = (gray2 - gray2.mean()) / (gray2.std() + 1e-8)
    ncc = (g1_norm * g2_norm).mean()

    # 也计算SSIM的简化版
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = gray1.mean(), gray2.mean()
    sigma1_sq, sigma2_sq = gray1.var(), gray2.var()
    sigma12 = ((gray1 - mu1) * (gray2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))

    return (ncc + ssim) / 2  # 综合分数


# ==================== 主分析函数 ====================
def analyze_cube(cube):
    """
    分析一个立方体的三个面之间的边缘衔接情况
    """
    print(f"\n{'=' * 70}")
    print(f"  分析木块 {cube}")
    print(f"{'=' * 70}")

    # 1. 加载三个面
    faces = {}
    for face_name in ['1', '2', '3']:
        blocks = load_face(cube, face_name)
        faces[face_name] = blocks
        annotated = sum(1 for b in blocks.values() if b['annotations'] is not None)
        print(f"  面 {face_name}: {annotated}/{len(blocks)} 个木块有标注")

    # 2. 提取所有边缘
    edge_strips = {}
    edge_profiles = {}
    for face_name, blocks in faces.items():
        for edge in ['top', 'bottom', 'left', 'right']:
            strip, boundaries = get_edge_strip(blocks, edge)
            edge_strips[(face_name, edge)] = (strip, boundaries)
            edge_profiles[(face_name, edge)] = compute_edge_profile(strip)

    # 3. 全面比对所有可能的边缘配对
    face_pairs = [('1', '2'), ('1', '3'), ('2', '3')]
    all_matches = []

    for f1, f2 in face_pairs:
        for e1 in ['top', 'bottom', 'left', 'right']:
            for e2 in ['top', 'bottom', 'left', 'right']:
                # 图像相似度
                img1, _ = edge_strips.get((f1, e1), (None, None))
                img2, _ = edge_strips.get((f2, e2), (None, None))

                if img1 is None or img2 is None:
                    continue

                # 直接比较 + 翻转比较
                score_direct = compare_edge_images(img1, img2, flip=False)
                score_flipped = compare_edge_images(img1, img2, flip=True)

                # 特征曲线比较
                p1 = edge_profiles.get((f1, e1))
                p2 = edge_profiles.get((f2, e2))
                profile_score = compare_profiles(p1, p2, p2_reversed=False)
                profile_score_rev = compare_profiles(p1, p2, p2_reversed=True)

                best_img_score = max(score_direct, score_flipped)
                best_profile_score = max(profile_score, profile_score_rev)
                needs_flip = score_flipped > score_direct

                combined_score = 0.5 * best_img_score + 0.5 * best_profile_score

                all_matches.append({
                    'faces': (f1, f2),
                    'edges': (e1, e2),
                    'img_score': best_img_score,
                    'profile_score': best_profile_score,
                    'combined_score': combined_score,
                    'needs_flip': needs_flip,
                })

    # 4. 排序并输出结果
    all_matches.sort(key=lambda x: -x['combined_score'])

    print(f"\n  所有边缘配对评分（前20）:")
    print(f"  {'面1':>4} {'边1':>6} {'面2':>4} {'边2':>6} {'图像分':>8} {'曲线分':>8} {'综合分':>8} {'翻转':>4}")
    print(f"  {'-' * 52}")
    for m in all_matches[:20]:
        f1, f2 = m['faces']
        e1, e2 = m['edges']
        flip_mark = '是' if m['needs_flip'] else '否'
        print(f"  {f1:>4} {EDGE_CN[e1]:>6} {f2:>4} {EDGE_CN[e2]:>6} "
              f"{m['img_score']:>8.4f} {m['profile_score']:>8.4f} "
              f"{m['combined_score']:>8.4f} {flip_mark:>4}")

    # 5. 推测最佳的三面边缘对应关系
    # 面1-面2, 面1-面3, 面2-面3 各有一个最佳配对
    best_pairings = {}
    for (f1, f2) in face_pairs:
        pair_matches = [m for m in all_matches if m['faces'] == (f1, f2)]
        if pair_matches:
            best = pair_matches[0]
            best_pairings[(f1, f2)] = best

    print(f"\n  推测的边缘对应关系:")
    for (f1, f2), m in best_pairings.items():
        e1, e2 = m['edges']
        print(f"    面{f1}的{EDGE_CN[e1]} ↔ 面{f2}的{EDGE_CN[e2]} "
              f"(综合分: {m['combined_score']:.4f})")

    return faces, edge_strips, all_matches, best_pairings


# ==================== 可视化 ====================
def draw_annotations_on_block(block_img, annotations, color=(0, 255, 0), thickness=3):
    """在木块图像上绘制年轮标注线"""
    if annotations is None:
        return block_img.copy()
    vis = block_img.copy()
    for shape in annotations.get('shapes', []):
        if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.polylines(vis, [points], False, color, thickness)
    return vis


def draw_annotations_on_edge_strip(strip_img, blocks_list, edge,
                                    color=(0, 255, 0), thickness=3):
    """
    在边缘条带上绘制来自各个木块的年轮线
    blocks_list: 沿边缘方向的木块信息列表
    edge: 边缘方向
    """
    if strip_img is None:
        return None

    vis = strip_img.copy()

    if edge == 'top':
        for idx, block_info in enumerate(blocks_list):
            r, c = 1, idx + 1
            block = block_info
            if block and block['annotations']:
                x_offset = 0
                for prev_idx in range(idx):
                    prev_block = blocks_list[prev_idx]
                    if prev_block and prev_block['image'] is not None:
                        x_offset += prev_block['image'].shape[1]

                for shape in block['annotations'].get('shapes', []):
                    if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
                        points = np.array(shape['points'], dtype=np.int32)
                        # 只保留靠近上边缘的点
                        h = block['image'].shape[0]
                        points = points[points[:, 1] < EDGE_STRIP_WIDTH]
                        points[:, 0] += x_offset  # 偏移x坐标
                        if len(points) > 1:
                            cv2.polylines(vis, [points], False, color, thickness)

    elif edge == 'bottom':
        for idx, block_info in enumerate(blocks_list):
            block = block_info
            if block and block['annotations'] and block['image'] is not None:
                x_offset = 0
                for prev_idx in range(idx):
                    prev_block = blocks_list[prev_idx]
                    if prev_block and prev_block['image'] is not None:
                        x_offset += prev_block['image'].shape[1]

                img_h = block['image'].shape[0]
                for shape in block['annotations'].get('shapes', []):
                    if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
                        points = np.array(shape['points'], dtype=np.int32)
                        # 只保留靠近下边缘的点
                        points = points[points[:, 1] > img_h - EDGE_STRIP_WIDTH].copy()
                        if len(points) > 1:
                            points[:, 1] -= (img_h - EDGE_STRIP_WIDTH)
                            points[:, 0] += x_offset
                            cv2.polylines(vis, [points], False, color, thickness)

    elif edge == 'left':
        for idx, block_info in enumerate(blocks_list):
            r, c = idx + 1, 1
            block = block_info
            if block and block['annotations']:
                y_offset = 0
                for prev_idx in range(idx):
                    prev_block = blocks_list[prev_idx]
                    if prev_block and prev_block['image'] is not None:
                        y_offset += prev_block['image'].shape[0]

                for shape in block['annotations'].get('shapes', []):
                    if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
                        points = np.array(shape['points'], dtype=np.int32)
                        # 只保留靠近左边缘的点
                        points = points[points[:, 0] < EDGE_STRIP_WIDTH]
                        points[:, 1] += y_offset
                        if len(points) > 1:
                            cv2.polylines(vis, [points], False, color, thickness)

    elif edge == 'right':
        for idx, block_info in enumerate(blocks_list):
            block = block_info
            if block and block['annotations'] and block['image'] is not None:
                y_offset = 0
                for prev_idx in range(idx):
                    prev_block = blocks_list[prev_idx]
                    if prev_block and prev_block['image'] is not None:
                        y_offset += prev_block['image'].shape[0]

                img_w = block['image'].shape[1]
                for shape in block['annotations'].get('shapes', []):
                    if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
                        points = np.array(shape['points'], dtype=np.int32)
                        # 只保留靠近右边缘的点
                        points = points[points[:, 0] > img_w - EDGE_STRIP_WIDTH].copy()
                        if len(points) > 1:
                            points[:, 0] -= (img_w - EDGE_STRIP_WIDTH)
                            points[:, 1] += y_offset
                            cv2.polylines(vis, [points], False, color, thickness)

    return vis


def assemble_face_preview(blocks, max_dim=2000):
    """组装一个面的缩略图"""
    # 获取基准尺寸
    sample = next(b['image'] for b in blocks.values() if b['image'] is not None)
    bh, bw = sample.shape[:2]

    # 缩放
    scale = min(max_dim / (COLS * bw), max_dim / (ROWS * bh))
    new_bh, new_bw = int(bh * scale), int(bw * scale)

    canvas = np.zeros((ROWS * new_bh, COLS * new_bw, 3), dtype=np.uint8)

    for (r, c), block in blocks.items():
        if block['image'] is None:
            continue
        img = cv2.resize(block['image'], (new_bw, new_bh))
        y = (r - 1) * new_bh
        x = (c - 1) * new_bw
        canvas[y:y + new_bh, x:x + new_bw] = img

    return canvas


def create_edge_comparison_figure(cube, faces, edge_strips, best_pairings):
    """
    创建边缘对比图：
    展示推测的最佳边缘配对
    """
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(f'木块 {cube} — 边缘衔接检查', fontsize=18, fontweight='bold', y=0.98)

    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
                  top=0.92, bottom=0.05, left=0.04, right=0.96)

    face_pairs = [('1', '2'), ('1', '3'), ('2', '3')]

    for row_idx, (f1, f2) in enumerate(face_pairs):
        # 第一列：面f1的边缘
        ax_f1 = fig.add_subplot(gs[row_idx, 0])
        # 第二列：面f2的边缘
        ax_f2 = fig.add_subplot(gs[row_idx, 1])
        # 第三列：并排对比
        ax_compare = fig.add_subplot(gs[row_idx, 2:])

        # 获取最佳配对
        pair_key = (f1, f2)
        if pair_key in best_pairings:
            best = best_pairings[pair_key]
            e1, e2 = best['edges']
            needs_flip = best['needs_flip']
            score = best['combined_score']

            strip1, bounds1 = edge_strips.get((f1, e1), (None, None))
            strip2, bounds2 = edge_strips.get((f2, e2), (None, None))

            # 绘制面1的边缘
            if strip1 is not None:
                strip1_rgb = cv2.cvtColor(strip1, cv2.COLOR_BGR2RGB)
                ax_f1.imshow(strip1_rgb)
                ax_f1.set_title(f'面{f1} {EDGE_CN[e1]}', fontsize=12)
                ax_f1.axis('off')

            # 绘制面2的边缘
            if strip2 is not None:
                strip2_rgb = cv2.cvtColor(strip2, cv2.COLOR_BGR2RGB)
                if needs_flip:
                    strip2_rgb = np.flip(strip2_rgb, axis=(0 if e2 in ['left', 'right'] else 1))
                ax_f2.imshow(strip2_rgb)
                ax_f2.set_title(f'面{f2} {EDGE_CN[e2]}{"(翻转)" if needs_flip else ""}', fontsize=12)
                ax_f2.axis('off')

            # 并排对比图
            if strip1 is not None and strip2 is not None:
                if e1 in ['top', 'bottom'] and e2 in ['top', 'bottom']:
                    # 两个都是水平边缘 → 上下排列
                    h1, w1 = strip1.shape[:2]
                    h2, w2 = strip2.shape[:2]
                    target_w = max(w1, w2)

                    s1 = cv2.resize(strip1, (target_w, h1))
                    s2 = cv2.resize(strip2, (target_w, h2))
                    if needs_flip:
                        s2 = cv2.flip(s2, 1)

                    divider = np.ones((5, target_w, 3), dtype=np.uint8) * 128
                    combined = np.vstack([s1, divider, s2])

                elif e1 in ['left', 'right'] and e2 in ['left', 'right']:
                    # 两个都是垂直边缘 → 左右排列
                    h1, w1 = strip1.shape[:2]
                    h2, w2 = strip2.shape[:2]
                    target_h = max(h1, h2)

                    s1 = cv2.resize(strip1, (w1, target_h))
                    s2 = cv2.resize(strip2, (w2, target_h))
                    if needs_flip:
                        s2 = cv2.flip(s2, 0)

                    divider = np.ones((target_h, 5, 3), dtype=np.uint8) * 128
                    combined = np.hstack([s1, divider, s2])

                else:
                    # 一个水平一个垂直 → 并列
                    h1, w1 = strip1.shape[:2]
                    h2, w2 = strip2.shape[:2]
                    max_h = max(h1, h2)
                    s1_padded = np.zeros((max_h, w1, 3), dtype=np.uint8)
                    s2_padded = np.zeros((max_h, w2, 3), dtype=np.uint8)
                    s1_padded[:h1, :] = strip1
                    s2_padded[:h2, :] = strip2

                    divider = np.ones((max_h, 5, 3), dtype=np.uint8) * 128
                    combined = np.hstack([s1_padded, divider, s2_padded])

                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
                ax_compare.imshow(combined_rgb)
                ax_compare.set_title(
                    f'面{f1} {EDGE_CN[e1]} ↔ 面{f2} {EDGE_CN[e2]}  '
                    f'(相似度: {score:.3f})',
                    fontsize=12, color='green' if score > 0.3 else 'red'
                )
                ax_compare.axis('off')
        else:
            ax_compare.text(0.5, 0.5, '未找到对应边缘', ha='center', va='center',
                            transform=ax_compare.transAxes, fontsize=14)

    # 保存
    output_path = os.path.join(OUTPUT_DIR, f'{cube}_edge_comparison.png')
    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  边缘对比图已保存: {output_path}")
    return output_path


def create_face_overview(cube, faces, best_pairings):
    """
    创建三个面的概览图，标注各面的边缘对应关系
    """
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'木块 {cube} — 三面概览与边缘对应', fontsize=18, fontweight='bold')

    for idx, face_name in enumerate(['1', '2', '3']):
        ax = fig.add_subplot(1, 3, idx + 1)
        blocks = faces[face_name]

        # 组装面预览
        preview = assemble_face_preview(blocks)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        h, w = preview_rgb.shape[:2]
        ax.imshow(preview_rgb)

        # 标注四个边缘
        edge_positions = {
            'top': (w // 2, 0),
            'bottom': (w // 2, h),
            'left': (0, h // 2),
            'right': (w, h // 2),
        }
        edge_colors = {'top': 'red', 'bottom': 'orange', 'left': 'cyan', 'right': 'yellow'}

        for edge, (ex, ey) in edge_positions.items():
            ax.plot(ex, ey, 'o', color=edge_colors[edge], markersize=15,
                    markeredgecolor='black', markeredgewidth=2, zorder=10)
            # 偏移文字避免重叠
            offset_x = 30 if edge == 'left' else (-30 if edge == 'right' else 0)
            offset_y = 30 if edge == 'top' else (-30 if edge == 'bottom' else 0)
            ax.annotate(EDGE_CN[edge], (ex, ey),
                        textcoords="offset points", xytext=(offset_x, offset_y),
                        fontsize=11, color=edge_colors[edge], fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                        ha='center', va='center')

        ax.set_title(f'面 {face_name}', fontsize=14)
        ax.axis('off')

    # 在底部添加对应关系说明
    text_lines = []
    for (f1, f2), m in best_pairings.items():
        e1, e2 = m['edges']
        flip = '(翻转)' if m['needs_flip'] else ''
        text_lines.append(
            f'面{f1}{EDGE_CN[e1]} ↔ 面{f2}{EDGE_CN[e2]} {flip}  (相似度: {m["combined_score"]:.4f})'
        )

    fig.text(0.5, 0.02, '\n'.join(text_lines), ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

    output_path = os.path.join(OUTPUT_DIR, f'{cube}_face_overview.png')
    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  面概览图已保存: {output_path}")
    return output_path


def create_detailed_edge_report(cube, faces, edge_strips, best_pairings):
    """
    创建详细的边缘对比报告，包含原始边缘和标注叠加
    """
    n_pairs = len(best_pairings)
    if n_pairs == 0:
        return

    fig, axes = plt.subplots(n_pairs, 3, figsize=(24, 6 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'木块 {cube} — 详细边缘衔接报告', fontsize=16, fontweight='bold', y=1.01)

    for row_idx, ((f1, f2), best) in enumerate(best_pairings.items()):
        e1, e2 = best['edges']
        needs_flip = best['needs_flip']

        # 获取边缘条带
        strip1, bounds1 = edge_strips.get((f1, e1), (None, None))
        strip2, bounds2 = edge_strips.get((f2, e2), (None, None))

        # 列1: 面1边缘
        ax1 = axes[row_idx, 0]
        if strip1 is not None:
            ax1.imshow(cv2.cvtColor(strip1, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'面{f1} — {EDGE_CN[e1]} (原始)', fontsize=11)
        ax1.axis('off')

        # 列2: 面2边缘
        ax2 = axes[row_idx, 1]
        if strip2 is not None:
            s2 = strip2.copy()
            if needs_flip:
                s2 = np.flip(s2, axis=(0 if e2 in ['left', 'right'] else 1))
            ax2.imshow(cv2.cvtColor(s2, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'面{f2} — {EDGE_CN[e2]} (原始{" + 翻转" if needs_flip else ""})', fontsize=11)
        ax2.axis('off')

        # 列3: 叠加对比（半透明叠加）
        ax3 = axes[row_idx, 2]
        if strip1 is not None and strip2 is not None:
            # 缩放到相同尺寸
            s1_gray = cv2.cvtColor(strip1, cv2.COLOR_BGR2GRAY)
            s2 = strip2.copy()
            if needs_flip:
                s2 = np.flip(s2, axis=(0 if e2 in ['left', 'right'] else 1))
            s2_gray = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)

            target_h = min(s1_gray.shape[0], s2_gray.shape[0])
            target_w = min(s1_gray.shape[1], s2_gray.shape[1])
            s1_gray = cv2.resize(s1_gray, (target_w, target_h))
            s2_gray = cv2.resize(s2_gray, (target_w, target_h))

            # 叠加：面1=青色，面2=品红色
            overlay = np.zeros((target_h, target_w, 3), dtype=np.float32)
            overlay[:, :, 0] = s1_gray * 0.7  # 青色通道 (B)
            overlay[:, :, 1] = s1_gray * 0.7  # (G)
            overlay[:, :, 2] = s1_gray * 0.3  # (R) - 偏青
            overlay[:, :, 1] += s2_gray * 0.3  # (G) - 偏品红
            overlay[:, :, 2] += s2_gray * 0.7  # (R)

            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            ax3.imshow(overlay)
            ax3.set_title(
                f'叠加对比: 面{f1}{EDGE_CN[e1]}(青) + 面{f2}{EDGE_CN[e2]}(品红)\n'
                f'相似度: {best["combined_score"]:.3f}',
                fontsize=10
            )
        ax3.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'{cube}_detailed_report.png')
    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  详细报告已保存: {output_path}")
    return output_path


def create_summary_report(all_results):
    """
    创建所有木块的汇总报告
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('所有木块边缘衔接评分汇总', fontsize=18, fontweight='bold')

    for idx, (cube, results) in enumerate(all_results.items()):
        ax = axes[idx // 3, idx % 3]
        _, _, all_matches, best_pairings = results

        # 展示最佳配对的评分
        labels = []
        scores = []
        colors = []
        for (f1, f2), m in best_pairings.items():
            labels.append(f'面{f1}-面{f2}\n{m["edges"][0]}-{m["edges"][1]}')
            scores.append(m['combined_score'])
            colors.append('green' if m['combined_score'] > 0.3 else 'orange' if m['combined_score'] > 0.15 else 'red')

        bars = ax.bar(range(len(labels)), scores, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('相似度')
        ax.set_title(f'木块 {cube}', fontsize=13)
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='较好')
        ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='一般')

        # 在柱子上标数值
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', fontsize=9)

        ax.legend(fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'summary_report.png')
    fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n汇总报告已保存: {output_path}")


# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("  木块立方体边缘衔接检查工具")
    print("=" * 70)
    print(f"  数据目录: {BASE_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  待检查木块: {', '.join(CUBES)}")
    print(f"  边缘提取宽度: {EDGE_STRIP_WIDTH} 像素")

    all_results = {}

    for cube in CUBES:
        try:
            faces, edge_strips, all_matches, best_pairings = analyze_cube(cube)

            # 生成可视化
            create_edge_comparison_figure(cube, faces, edge_strips, best_pairings)
            create_face_overview(cube, faces, best_pairings)
            create_detailed_edge_report(cube, faces, edge_strips, best_pairings)

            all_results[cube] = (faces, edge_strips, all_matches, best_pairings)

        except Exception as e:
            print(f"  [错误] 处理木块 {cube} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 生成汇总报告
    if all_results:
        create_summary_report(all_results)

    print(f"\n{'=' * 70}")
    print(f"  检查完成！所有结果已保存到: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
