"""
================================================================================
DeepCS-TRD 年轮检测 — 用户自定义数据运行脚本
==============================================
使用 DeepCS-TRD（U-Net + CS-TRD几何流水线）对你自己的木材图像进行
年轮检测，并可选用你的LabelMe标注进行评估。

完整流程:
  1. 读取 pith.csv（每张图对应的髓心坐标）
  2. 逐张图像调用 DeepTreeRingDetection() → U-Net推理 + 几何后处理
  3. 输出 LabelMe JSON（矢量年轮轮廓）+ 可视化图像
  4. [可选] 与标注对比评估（Precision/Recall/F1/RMSE）

使用前准备:
  1. 将图像文件放入 my_data/ 目录
  2. 填写 my_data/pith.csv（Code, cx, cy 三列）
  3. [可选] 将标注JSON放入 my_data/annotations/

用法:
  python run_on_my_data.py
================================================================================
"""
import os, sys, ssl, time, json, cv2
from pathlib import Path
import pandas as pd
import numpy as np

# ======================================================================
# 环境兼容性修复（解决DeepCS-TRD在Windows上的依赖问题）
# ======================================================================

# ★ SSL证书验证绕过 — 解决部分环境下urllib下载ImageNet权重时的SSL错误
ssl._create_default_https_context = ssl._create_unverified_context

# ★ 禁用HuggingFace Hub在线检查 — 防止segmentation_models_pytorch尝试从HF下载
os.environ["HF_HUB_OFFLINE"] = "1"

# ★ SMP绕过ImageNet预训练权重下载
# segmentation_models_pytorch库默认会尝试从网上下载ImageNet预训练权重
# 在离线或网络受限环境下会失败。这里通过monkey-patch强制使用本地权重
# （若本地也没有则使用随机初始化的权重，迁移学习效果会下降）
import segmentation_models_pytorch as smp
_orig_get_encoder = smp.encoders.get_encoder
smp.encoders.get_encoder = lambda name, weights='imagenet', **kw: _orig_get_encoder(name, weights=None, **kw)

# ★ Shapely 2.x 兼容性修复
# Shapely 2.0+改变了内部几何对象的表示方式（使用__geom__包装），
# 但CS-TRD的底层代码仍然期望Shapely 1.x的接口。
# 以下monkey-patch为所有常用的几何函数添加自动unwrap逻辑
import shapely.lib as _lib

def _unwrap_geom(obj):
    """从Shapely 2.x的包装对象中提取底层GEOS几何对象"""
    if hasattr(obj, '__geom__'):
        return obj.__geom__
    return obj

def _make_unwrapper(func):
    """为几何函数创建一个自动unwrap所有参数的包装器"""
    def wrapper(*args, **kwargs):
        return func(*[_unwrap_geom(a) for a in args], **kwargs)
    return wrapper

# ★ 为Shapely的所有常用几何函数添加unwrap包装
for _name in ['intersection', 'intersects', 'distance', 'touches', 'contains', 'within',
              'crosses', 'overlaps', 'union', 'difference', 'buffer', 'length', 'area',
              'convex_hull', 'boundary', 'centroid', 'is_valid', 'is_empty', 'simplify',
              'hausdorff_distance', 'project', 'interpolate', 'relate', 'shortest_line',
              'point_on_surface', 'snap', 'extract_unique_points']:
    if hasattr(_lib, _name):
        setattr(_lib, _name, _make_unwrapper(getattr(_lib, _name)))


# ======================================================================
# 路径配置
# ======================================================================
WORK_DIR = Path(r"C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block")
sys.path.insert(0, str(WORK_DIR))
sys.path.insert(0, r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main")

from urudendro.image import load_image
from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection  # ★ 年轮检测主函数
from cross_section_tree_ring_detection.utils import saving_results      # 结果保存工具


# ======================================================================
# 用户可自定义的配置
# ======================================================================

# --- 数据目录 ---
# 放置你的木材图像（已去除背景的 jpg 或 png，白色背景）
DATA_DIR = WORK_DIR / "my_data"
DATA_DIR.mkdir(exist_ok=True)

# --- 髓心坐标CSV ---
# 格式: Code,cx,cy
#   Code = 图像文件名（不含扩展名）
#   cx = 髓心X坐标（像素）
#   cy = 髓心Y坐标（像素）
PITH_CSV = DATA_DIR / "pith.csv"

# --- U-Net预训练模型权重 ---
# 推荐选择:
#   256_pinus_v1_1504.pth  = 火炬松v1 + 256×256分块推理（推荐）
#   0_pinus_v1_1504.pth    = 火炬松v1 + 整图推理（不分块，但需要更多GPU显存）
#   0_all_1504.pth         = 通用多树种模型
MODEL_PATH = r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main\models\deep_cstrd\256_pinus_v1_1504.pth"

# --- 输出目录 ---
OUTPUT_DIR = WORK_DIR / "my_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ======================================================================
# DeepCS-TRD 检测参数
# ======================================================================
DETECTION_PARAMS = dict(
    height=1504,        # 预处理高度（0=不缩放，1504=论文推荐的分辨率）
    width=1504,         # 预处理宽度
    alpha=45,           # 边缘过滤角度阈值（度），越大越宽松（保留更多边缘）
                        #   默认30°，这里用45°以适应你的数据
    nr=360,             # ★ 射线数量（360条 = 每1°一条射线）
    mc=2,               # 最小链长度（至少2个交点才能形成一条链）
    weights_path=MODEL_PATH,  # U-Net预训练权重路径
    total_rotations=4,  # ★ TTA旋转推理次数（4=0°/90°/180°/270°）
    prediction_map_threshold=0.2,  # ★ 二值化阈值（概率>0.2→年轮边界）
    tile_size=256,      # ★ 分块推理大小（256=256×256的小块拼接）
                        #   使用 256_ 模型时必须设为256
                        #   使用 0_ 模型时设为0（不分块）
    batch_size=1,       # U-Net推理批次大小（GPU显存小就设为1）
    encoder='resnet18', # U-Net编码器（对应权重文件的训练配置，不要改）
    debug=False,        # True=保存中间结果图像（概率图、法向量、Chain等）
)


def detect_single(image_path, cx, cy, output_subdir):
    """
    ====================================================================
    对单张图像执行DeepCS-TRD年轮检测
    ====================================================================
    调用 DeepTreeRingDetection() 的完整流水线:
      U-Net语义分割 → TTA多角度融合 → 二值化 → 骨架化
      → 轮廓提取 → 法向量计算 → 边缘方向过滤
      → 射线采样 → 链合并 → 后处理 → LabelMe JSON输出

    参数:
        image_path:    输入图像路径
        cx, cy:        髓心坐标
        output_subdir: 输出子目录名（通常用样本名）
    返回:
        sample_output: 输出目录（包含 labelme.json + 可视化图像）
    """
    sample_output = OUTPUT_DIR / output_subdir
    sample_output.mkdir(exist_ok=True, parents=True)

    print(f"  检测: {image_path}")
    print(f"  树髓: cx={cx}, cy={cy}")

    # ★ 加载图像
    im_in = load_image(str(image_path))

    # ★★★ 核心调用：DeepTreeRingDetection ★★★
    # 此处是UNet推理 + 完整几何后处理的入口
    # 返回: (im_in, im_pre, m_ch_e, l_ch_f, l_ch_s, l_ch_c, l_ch_p, l_rings)
    #   l_rings = 最终的LabelMe JSON格式年轮轮廓
    res = DeepTreeRingDetection(
        im_in, cy=cy, cx=cx,
        debug_output_dir=str(sample_output),      # 中间结果保存目录
        debug_image_input_path=str(image_path),   # 输入图像路径（写入JSON）
        **DETECTION_PARAMS                        # 展开所有检测参数
    )

    # ★ 保存结果（LabelMe JSON + 可视化图像）
    # saving_results 内部调用 chain_2_labelme_json + write_image
    saving_results(res, str(sample_output), 1)

    return sample_output


def evaluate_single(dt_json_path, gt_json_path, img_path, cx, cy):
    """
    ====================================================================
    评估检测结果 vs 标注真值
    ====================================================================
    使用UruDendro的compute_metrics计算:
      - Precision: 检测到的环中有多少是正确的
      - Recall:    标注的环中有多少被检测到
      - F1:         Precision和Recall的调和平均
      - RMSE:       环位置的均方根误差
      - TP/FP/TN/FN: 混淆矩阵

    参数:
        dt_json_path: 检测结果JSON（DeepTreeRingDetection的输出）
        gt_json_path: 标注真值JSON（你的LabelMe标注）
        img_path:     原始图像路径
        cx, cy:       髓心坐标
    返回:
        评估指标字典
    """
    import urudendro

    eval_dir = Path(dt_json_path).parent / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    # ★ UruDendro的compute_metrics使用影响区域法（metric of influence area）
    # 对比检测环和标注环在极坐标空间中的一致性
    precision, recall, f_score, rmse, tp, fp, tn, fn = urudendro.compute_metrics(
        str(dt_json_path), str(gt_json_path), str(img_path),
        cx=cx, cy=cy, threshold=0.5,          # IoU匹配阈值=0.5
        output_dir=str(eval_dir)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "rmse": rmse,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def main():
    """
    ====================================================================
    主函数：批量处理 + 汇总评估
    ====================================================================
    """
    print("=" * 60)
    print("deepCS-TRD 年轮检测 + urudendro 评估")
    print("=" * 60)

    # ====== 步骤1: 读取髓心坐标 ======
    if not PITH_CSV.exists():
        # 首次运行 → 创建示例CSV文件
        print(f"\n⚠️ 请先创建树髓坐标文件: {PITH_CSV}")
        print("格式: CSV 文件，包含三列: Code, cx, cy")
        print("示例:")
        print("  Code,cx,cy")
        print("  T0_B1_N27_A,1841,2002")

        sample_df = pd.DataFrame({"Code": ["sample1"], "cx": [0], "cy": [0]})
        sample_df.to_csv(PITH_CSV, index=False)
        print(f"\n已创建示例文件: {PITH_CSV}")
        print("请编辑后重新运行")
        return

    pith_df = pd.read_csv(PITH_CSV)
    print(f"\n共 {len(pith_df)} 个样本待处理")

    # ====== 步骤2: 逐样本处理 ======
    results = []
    for idx, row in pith_df.iterrows():
        code = row["Code"]
        cx, cy = int(row["cx"]), int(row["cy"])

        print(f"\n--- [{idx+1}/{len(pith_df)}] {code} ---")

        # --- 查找图像文件（支持 .jpg/.jpeg/.png 扩展名）---
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = DATA_DIR / f"{code}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # 模糊匹配（图像文件名包含code字符串）
            for f in DATA_DIR.glob("*"):
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and code in f.stem:
                    img_path = f
                    break

        if img_path is None:
            print(f"  ❌ 找不到图像文件 {code}")
            continue

        # --- 执行检测 ---
        t0 = time.time()
        output_dir = detect_single(img_path, cx, cy, code)
        elapsed = time.time() - t0
        print(f"  ✅ 检测完成 ({elapsed:.1f}s)")

        # --- 读取检测到的年轮数量 ---
        labelme_json = output_dir / "labelme.json"
        if labelme_json.exists():
            with open(labelme_json, 'r') as f:
                data = json.load(f)
            # ★ 从JSON中提取所有polygon类型的shape（每个polygon = 一个年轮环）
            rings = [s for s in data['shapes'] if s['shape_type'] == 'polygon']
            print(f"  年轮数: {len(rings)}")

        # --- [可选] 评估：对比你的标注 ---
        gt_json = DATA_DIR / "annotations" / f"{code}.json"
        if gt_json.exists():
            print(f"  评估中...")
            try:
                metrics = evaluate_single(labelme_json, gt_json, img_path, cx, cy)
                metrics["sample"] = code
                metrics["time"] = elapsed
                results.append(metrics)
                print(f"  P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                      f"F1={metrics['f_score']:.3f} RMSE={metrics['rmse']:.2f}")
            except Exception as e:
                print(f"  ⚠️ 评估失败: {e}")

    # ====== 步骤3: 汇总统计 ======
    print("\n" + "=" * 60)
    print("处理完成!")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

        print(f"平均 Precision: {df['precision'].mean():.3f}")
        print(f"平均 Recall:    {df['recall'].mean():.3f}")
        print(f"平均 F-score:   {df['f_score'].mean():.3f}")
        print(f"平均 RMSE:      {df['rmse'].mean():.2f}")
        print(f"\n详细结果: {OUTPUT_DIR / 'summary.csv'}")

    print(f"\n所有输出: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
