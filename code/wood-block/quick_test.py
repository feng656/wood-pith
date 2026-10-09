"""
快速测试：单张 UruDendro4 图片的年轮检测
"""
import os, sys, time, ssl
from pathlib import Path

# === 网络修复: 跳过 SSL 验证 + 禁用 HF hub ===
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_OFFLINE"] = "1"          # 禁止从 huggingface 下载
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# === 绕过 ImageNet 预训练权重下载 ===
# checkpoint 已经包含所有权重，不需要下载 ImageNet 预训练权重
import segmentation_models_pytorch as smp
_original_get_encoder = smp.encoders.get_encoder
def _patched_get_encoder(name, weights='imagenet', **kwargs):
    return _original_get_encoder(name, weights=None, **kwargs)  # 跳过下载
smp.encoders.get_encoder = _patched_get_encoder
import segmentation_models_pytorch.decoders.unet.model as unet_model
unet_model.get_encoder = _patched_get_encoder

# === 绕过 Shapely 2.x 子类不兼容 ===
# Shapely 2.x 不支持 LineString/Polygon 子类，但 CS-TRD 代码大量使用。
# 我们通过 monkey-patch shapely.lib 来支持包装类 (有 __geom__ 属性的对象)
import shapely
import shapely.lib as _lib

def _unwrap_geom(obj):
    """如果对象有 __geom__ 属性，返回原始 Shapely geometry"""
    if hasattr(obj, '__geom__'):
        return obj.__geom__
    return obj

def _make_unwrapper(func):
    """包装 shapely.lib 函数，自动解包 wrapper 对象"""
    def wrapper(*args, **kwargs):
        unwrapped = tuple(_unwrap_geom(a) for a in args)
        return func(*unwrapped, **kwargs)
    return wrapper

# Patch 关键的几何操作函数
_shapely_funcs = ['intersection', 'intersects', 'distance', 'touches', 'contains',
                  'within', 'crosses', 'overlaps', 'union', 'difference', 'buffer',
                  'convex_hull', 'envelope', 'simplify', 'boundary', 'centroid',
                  'relate', 'project', 'interpolate', 'hausdorff_distance',
                  'is_valid', 'is_empty', 'is_simple', 'is_ring', 'has_z',
                  'area', 'length', 'geom_type_id', 'get_coordinate_dimension',
                  'get_num_coordinates', 'get_num_geometries', 'get_num_interior_rings',
                  'get_num_points', 'get_num_rings', 'get_point', 'get_geometry_n',
                  'get_interior_ring_n', 'get_exterior_ring', 'get_x', 'get_y',
                  'set_coordinates', 'get_coordinates', 'normalize', 'make_valid',
                  'clip_by_rect', 'reverse', 'segmentize', 'line_merge', 'shared_paths',
                  'split', 'voronoi_polygons', 'delaunay_triangles', 'concave_hull',
                  'oriented_envelope', 'minimum_rotated_rectangle', 'minimum_bounding_circle',
                  'minimum_bounding_radius', 'minimum_clearance', 'polylabel',
                  'polygonize', 'polygonize_full', 'transform', 'node',
                  'point_on_surface', 'representative_point', 'shortest_line',
                  'snap', 'extract_unique_points', 'offset_curve', 'set_precision',
                  'is_closed', 'count_coordinates', 'get_parts', 'get_rings',
                  'is_prepared', 'prepare', 'destroy_prepared',
                  ]

for name in _shapely_funcs:
    if hasattr(_lib, name):
        orig = getattr(_lib, name)
        setattr(_lib, name, _make_unwrapper(orig))

WORK_DIR = r"C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block"
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main")

import pandas as pd
from urudendro.image import load_image
from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection
from cross_section_tree_ring_detection.utils import saving_results

# === 配置 ===
MODEL = r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main\models\deep_cstrd\256_pinus_v1_1504.pth"
IMG = r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\images_no_background\T0_B1_N27_A.jpg"
OUTPUT = Path(WORK_DIR) / "output_test" / "T0_B1_N27_A"
OUTPUT.mkdir(exist_ok=True, parents=True)

# 读 pith
pith_df = pd.read_csv(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4\pith_location.txt")
row = pith_df[pith_df["Code"] == "T0_B1_N27_A"]
cx, cy = int(row.iloc[0]["cx"]), int(row.iloc[0]["cy"])
print(f"样本: T0_B1_N27_A, pith=({cx},{cy}), 图像={IMG}")

# 检测
print("开始检测...")
t0 = time.time()
im_in = load_image(IMG)
res = DeepTreeRingDetection(
    im_in, cy=cy, cx=cx,
    height=1504, width=1504,
    alpha=45, nr=360, mc=2,
    weights_path=MODEL,
    total_rotations=4,
    prediction_map_threshold=0.2,
    tile_size=256, batch_size=1,
    encoder='resnet18',  # 模型实际用 resnet18 训练
    debug_output_dir=str(OUTPUT),
    debug_image_input_path=IMG,
    debug=False,
)
saving_results(res, str(OUTPUT), 1)
print(f"✅ 检测完成! 用时 {time.time()-t0:.1f}s")
print(f"输出: {OUTPUT}")
print(f"文件: {list(OUTPUT.glob('*'))}")
