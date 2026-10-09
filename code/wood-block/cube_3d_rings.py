"""
立方体3D年轮线可视化
从6个面的年轮标注(linestrip)提取年轮线，映射到3D正方体的6个面上渲染。
- 无背景(透明PNG)、无照片，只画年轮线
- 每个面用不同明暗/色相，增强3D立体感(顶面最亮、侧面渐暗)

面映射（与十字展开图折叠关系一致，图片位置同2D展开图）：
  前面 A: z=+0.5   x = x_img/W-0.5,  y = 0.5-y_img/H
  顶面 D: y=+0.5   x = x_img/W-0.5,  z = y_img/H-0.5   (图上方=背面)
  底面 B: y=-0.5   x = x_img/W-0.5,  z = 0.5-y_img/H   (图上方=正面)
  左面 C: x=-0.5   z = x_img/W-0.5,  y = 0.5-y_img/H   (图右=正面)
  右面 E: x=+0.5   z = 0.5-x_img/W,  y = 0.5-y_img/H   (图左=正面)
  后面 F: z=-0.5   x = 0.5-x_img/W,  y = 0.5-y_img/H   (图左=右棱)

用法: python cube_3d_rings.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = r'D:\教务处实习\木材数据1\立方体'
OUTPUT_DIR = r'C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block\cube_unfold_output'

# 每个面的颜色：顶面最亮(受光)，前面次之，右/左/后渐暗，底面最暗(几乎不可见)
FACE_COLORS = {
    'A': '#FF7043',   # 前面 亮橙
    'D': '#FFC107',   # 顶面 金黄(最亮)
    'E': '#4A6FD4',   # 右面 中蓝(受光较弱)
    'C': '#8E5BC9',   # 左面 紫(背光)
    'F': '#2E9E8F',   # 后面 青绿(暗)
    'B': '#9E9E9E',   # 底面 灰(几乎不可见)
}

# 3个立方体：按12条棱衔接评分从高到低选
CUBES = [(4, 1), (1, 1), (4, 2)]

# ==================== 立方体棱线 ====================
# 8个顶点 (x,y,z ∈ ±0.5)
VERTS = np.array([
    [-0.5, -0.5, -0.5],  # 0 底后左
    [0.5, -0.5, -0.5],   # 1 底后右
    [0.5, 0.5, -0.5],    # 2 顶后右
    [-0.5, 0.5, -0.5],   # 3 顶后左
    [-0.5, -0.5, 0.5],   # 4 底前左
    [0.5, -0.5, 0.5],    # 5 底前右
    [0.5, 0.5, 0.5],     # 6 顶前右
    [-0.5, 0.5, 0.5],    # 7 顶前左
])
# 每个面的顶点(右手定则, 外法线朝外)
FACES = {
    'A': [4, 5, 6, 7],  # 前面 +z
    'D': [3, 7, 6, 2],  # 顶面 +y
    'E': [1, 2, 6, 5],  # 右面 +x
    'C': [0, 4, 7, 3],  # 左面 -x
    'B': [0, 1, 5, 4],  # 底面 -y
    'F': [0, 3, 2, 1],  # 后面 -z
}
# 12条棱: (顶点1, 顶点2, [相邻面, 相邻面])
EDGES = [
    (0, 1, ['B', 'F']), (0, 4, ['B', 'C']), (0, 3, ['C', 'F']),
    (1, 2, ['E', 'F']), (1, 5, ['B', 'E']), (2, 3, ['D', 'F']),
    (2, 6, ['D', 'E']), (3, 7, ['D', 'C']), (4, 5, ['A', 'B']),
    (4, 7, ['A', 'C']), (5, 6, ['A', 'E']), (6, 7, ['A', 'D']),
]


def to_plot(xw, yw, zw):
    """世界坐标(x右, y上, z前) → matplotlib绘图坐标。
    让世界'上'(y)显示在屏幕上方，且世界'右'(x)镜像到屏幕右
    (配合azim=122的左前上视角，得到前面居中偏左、右面在右的经典视图)。"""
    return -xw, zw, yw


def visible_faces(ax):
    """用投影矩阵做屏幕空间背面剔除：与A面(前面)同号的为可见面"""
    proj = ax.get_proj()

    def area2d(fidx):
        pts = []
        for i in fidx:
            p = proj @ np.append(to_plot(*VERTS[i]), 1.0)
            pts.append((p[0] / p[3], p[1] / p[3]))
        s = 0.0
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            s += x1 * y2 - x2 * y1
        return s

    a_sign = np.sign(area2d(FACES['A']))
    return {f for f, idx in FACES.items() if np.sign(area2d(idx)) == a_sign}


def draw_edges(ax, vis):
    """可见棱实线，被遮挡棱虚线"""
    for v1, v2, adj in EDGES:
        visible = any(f in vis for f in adj)
        xw = np.array([VERTS[v1][0], VERTS[v2][0]])
        yw = np.array([VERTS[v1][1], VERTS[v2][1]])
        zw = np.array([VERTS[v1][2], VERTS[v2][2]])
        x, y, z = to_plot(xw, yw, zw)
        if visible:
            ax.plot(x, y, z, color='#3A3A3A', linewidth=1.4, linestyle='-')
        else:
            ax.plot(x, y, z, color='#8A8A8A', linewidth=1.1,
                    linestyle='--', dashes=(5, 4), alpha=0.85)


def load_rings(letter, r, c):
    """读取某面年轮线linestrip点列"""
    path = os.path.join(BASE_DIR, f'{letter}_1_blocks', f'{letter}_1_{r}{c}.json')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        ann = json.load(f)
    rings = []
    for shape in ann.get('shapes', []):
        if shape.get('shape_type') == 'linestrip' and shape.get('label') == '年轮':
            pts = np.array(shape['points'], dtype=float)
            if len(pts) >= 2:
                rings.append(pts)
    return rings


def to_3d(letter, pts, W=1870.0, H=1870.0):
    """面内2D点 → 3D坐标。W/H用该面图像实际尺寸"""
    x = pts[:, 0] / W - 0.5
    y = 0.5 - pts[:, 1] / H
    if letter == 'A':
        return x, y, np.full_like(x, 0.5)
    if letter == 'D':   # 顶面
        return x, np.full_like(x, 0.5), pts[:, 1] / H - 0.5
    if letter == 'B':   # 底面
        return x, np.full_like(x, -0.5), 0.5 - pts[:, 1] / H
    if letter == 'C':   # 左面
        return np.full_like(x, -0.5), y, pts[:, 0] / W - 0.5
    if letter == 'E':   # 右面
        return np.full_like(x, 0.5), y, 0.5 - pts[:, 0] / W
    if letter == 'F':   # 后面
        return 0.5 - pts[:, 0] / W, y, np.full_like(x, -0.5)
    raise ValueError(letter)


def render_cube(r, c, out_path):
    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=122)  # 左前上经典视角：前面居中偏左、右面在右、顶面在上
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.62, 0.62)
    ax.set_zlim(-0.62, 0.62)

    n_lines = 0
    for letter in ['A', 'B', 'C', 'D', 'E', 'F']:
        color = FACE_COLORS[letter]
        # 用该面图像实际尺寸归一化
        img_path = os.path.join(BASE_DIR, f'{letter}_1_blocks', f'{letter}_1_{r}{c}.jpg')
        W = H = 1870.0
        if os.path.exists(img_path):
            import cv2
            im = cv2.imread(img_path)
            if im is not None:
                H, W = im.shape[:2]
        for pts in load_rings(letter, r, c):
            xw, yw, zw = to_3d(letter, pts, float(W), float(H))
            x3, y3, z3 = to_plot(xw, yw, zw)
            ax.plot(x3, y3, z3, color=color, linewidth=1.6, solid_capstyle='round')
            n_lines += 1

    # 画棱：可见面实线、被遮挡面虚线
    vis = visible_faces(ax)
    draw_edges(ax, vis)

    fig.patch.set_alpha(0)
    fig.savefig(out_path, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return n_lines


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('  立方体3D年轮线可视化 (无背景, 只画年轮线)')
    for r, c in CUBES:
        out = os.path.join(OUTPUT_DIR, f'cube_r{r}c{c}_3D_rings.png')
        n = render_cube(r, c, out)
        print(f'  ({r},{c}): {n}条年轮线 -> {out}')


if __name__ == '__main__':
    main()
