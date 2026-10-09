"""
测试单张图片的完整流程：背景去除 → 年轮检测 → 可视化
使用 deepCS-TRD + uruDendro
"""
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# 添加工作目录到 Python 路径
WORK_DIR = r"C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block"
sys.path.insert(0, WORK_DIR)

# ==================== 配置 ====================
# 模型路径
MODEL_PATH = r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main\models\deep_cstrd\256_pinus_v1_1504.pth"

# 数据集路径
DATASET_DIR = r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
IMAGES_NO_BG_DIR = os.path.join(DATASET_DIR, "images_no_background")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations", "annual_rings")
PITH_FILE = os.path.join(DATASET_DIR, "pith_location.txt")

# 输出目录
OUTPUT_DIR = os.path.join(WORK_DIR, "output_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 选择测试样本 ====================
# 选一个有 ADAP (共识标注) 的样本
TEST_SAMPLE = "T0_B1_N27_A"

print(f"测试样本: {TEST_SAMPLE}")
print("=" * 60)

# ==================== 1. 加载树髓坐标 ====================
import pandas as pd
pith_df = pd.read_csv(PITH_FILE)

# 查找 pith 坐标 (去掉variant后缀来匹配)
# e.g., T0_B1_N27_A -> T0_B1_N27
sample_base = "_".join(TEST_SAMPLE.split("_")[:4])
variant = TEST_SAMPLE.split("_")[4] if len(TEST_SAMPLE.split("_")) > 4 else ""

# 直接匹配
row = pith_df[pith_df["Code"] == TEST_SAMPLE]
if row.empty:
    # 尝试去掉 variant
    row = pith_df[pith_df["Code"].str.startswith(sample_base)]

if row.empty:
    print(f"❌ 找不到样本 {TEST_SAMPLE} 的树髓坐标!")
    print(f"pith_location.txt 中的前 5 行:")
    print(pith_df.head())
    sys.exit(1)

cx = int(row.iloc[0]["cx"])
cy = int(row.iloc[0]["cy"])
print(f"树髓坐标: cx={cx}, cy={cy}")

# ==================== 2. 加载图像 ====================
# 使用已去除背景的图像
img_path = os.path.join(IMAGES_NO_BG_DIR, f"{TEST_SAMPLE}.jpg")

if not os.path.exists(img_path):
    # 回退到原始图像
    img_path = os.path.join(IMAGES_DIR, f"{TEST_SAMPLE}.png")
    print("使用原始图像（无已去除背景版本）")

if not os.path.exists(img_path):
    print(f"❌ 找不到图像文件!")
    sys.exit(1)

print(f"图像: {img_path}")

# ==================== 3. 背景去除（如果需要） ====================
# 如果使用原始图像，先去除背景
if "_no_background" not in img_path and not img_path.endswith(".jpg"):
    print("正在去除背景...")
    import urudendro
    bg_removed_path = os.path.join(OUTPUT_DIR, f"{TEST_SAMPLE}_no_bg.jpg")
    urudendro.remove_salient_object(img_path, bg_removed_path)
    img_path = bg_removed_path
    print(f"背景已去除: {img_path}")

# ==================== 4. 年轮检测 ====================
print(f"\n正在运行 deepCS-TRD 年轮检测...")
from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection

sample_output_dir = os.path.join(OUTPUT_DIR, TEST_SAMPLE)
os.makedirs(sample_output_dir, exist_ok=True)

# DeepTreeRingDetection 参数
result = DeepTreeRingDetection(
    im_in=img_path,
    cx=cx, cy=cy,
    height=1504,        # 调整后图像高度
    width=1504,         # 调整后图像宽度
    alpha=30,           # 边缘过滤共线性阈值
    nr=360,             # 射线数量
    mc=2,               # 最小链长度
    weights_path=MODEL_PATH,
    total_rotations=4,  # 多次旋转推理
    prediction_map_threshold=0.2,
    tile_size=256,      # 与模型匹配
    batch_size=1,
    encoder='resnet34',
    output_dir=sample_output_dir,
    save_imgs=True,
    debug=True,
)

print(f"\n检测完成! 结果保存在: {sample_output_dir}")
print(f"检测到的年轮数: {len(result.get('rings', []))}")

# ==================== 5. 可视化对比 ====================
print(f"\n正在生成对比图...")
import urudendro

gt_json = os.path.join(ANNOTATIONS_DIR, f"{TEST_SAMPLE}.json")
if os.path.exists(gt_json):
    urudendro.visualize_annotation(gt_json, img_path, sample_output_dir)
    print(f"Ground truth 可视化已保存")
else:
    print(f"⚠️ 找不到标注文件: {gt_json}")

print(f"\n✅ 单张测试完成!")
print(f"输出目录: {sample_output_dir}")
print(f"  - output.png: 年轮检测结果")
print(f"  - chains.png: 原始边缘链")
print(f"  - connect.png: 连接后的链")
print(f"  - postprocessing.png: 后处理结果")
print(f"  - labelme.json: 年轮坐标 (LabelMe 格式)")
