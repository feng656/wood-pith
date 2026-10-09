"""
UruDendro4 数据集批量年轮检测 + 评估脚本
使用 deepCS-TRD (Pinus V1 model, tile=256)

流程:
  1. 读取 pith_location.txt 获取树髓坐标
  2. 对每个样本运行 DeepTreeRingDetection
  3. 用 urudendro 评估检测结果 vs ground truth
"""
import os
import sys
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==================== 路径配置 ====================
DEEPCSTRD_DIR = r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main"
WORK_DIR = r"C:\Users\江卓峰\PycharmProjects\pythonProject4\wood block"
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, DEEPCSTRD_DIR)

DATASET_DIR = Path(r"D:\教务处实习\木材数据1\UruDendro4\UruDendro4")
IMAGES_NO_BG_DIR = DATASET_DIR / "images_no_background"
ANNOTATIONS_DIR = DATASET_DIR / "annotations" / "annual_rings"
PITH_FILE = DATASET_DIR / "pith_location.txt"

# 模型权重
MODELS = {
    "256_pinus_v1": r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main\models\deep_cstrd\256_pinus_v1_1504.pth",
    "0_pinus_v1": r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main\models\deep_cstrd\0_pinus_v1_1504.pth",
    "256_pinus_v2": r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main\models\deep_cstrd\256_pinus_v2_1504.pth",
    "0_all": r"D:\教务处实习\木材数据1\deepcstrd-main\deepcstrd-main\models\deep_cstrd\0_all_1504.pth",
}

# 输出目录
OUTPUT_BASE = Path(WORK_DIR) / "uru4_results"
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)

# ==================== 检测参数 ====================
DETECTION_PARAMS = dict(
    height=1504,
    width=1504,
    alpha=45,              # 边缘过滤共线性阈值 (论文推荐 45)
    nr=360,                # 射线数量
    mc=2,                  # 最小链长度
    prediction_map_threshold=0.2,
    total_rotations=4,     # 多次旋转推理
    tile_size=256,         # 256 tile 模型
    batch_size=1,
    encoder='resnet34',
    debug=False,           # 设为 True 可保存中间结果
)


def run_detection_for_sample(sample_code, cx, cy, model_path, output_dir):
    """对单个样本运行年轮检测"""
    from urudendro.image import load_image
    from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection
    from cross_section_tree_ring_detection.utils import saving_results

    # 查找图像文件
    img_path = IMAGES_NO_BG_DIR / f"{sample_code}.jpg"
    if not img_path.exists():
        # 尝试不加 variant
        for ext in ['.jpg', '.png']:
            candidates = list(IMAGES_NO_BG_DIR.glob(f"{sample_code.split('_')[0]}_{sample_code.split('_')[1]}_{sample_code.split('_')[2]}_*{ext}"))
            if candidates:
                # 取第一个匹配的
                img_path = candidates[0]
                break

    if not img_path.exists():
        print(f"  ⚠️ 找不到图像: {sample_code}")
        return None

    sample_output_dir = output_dir / sample_code
    sample_output_dir.mkdir(exist_ok=True, parents=True)

    # 跳过已处理的
    if (sample_output_dir / "labelme.json").exists():
        print(f"  ⏭️ 已有结果，跳过")
        return str(sample_output_dir)

    try:
        im_in = load_image(str(img_path))
        args = dict(cy=cy, cx=cx, weights_path=model_path,
                    debug_image_input_path=str(img_path),
                    debug_output_dir=str(sample_output_dir),
                    **DETECTION_PARAMS)
        res = DeepTreeRingDetection(im_in, **args)
        saving_results(res, str(sample_output_dir), 1)
        return str(sample_output_dir)
    except Exception as e:
        print(f"  ❌ 检测失败: {e}")
        return None


def evaluate_sample(sample_code, results_dir):
    """评估单个样本的检测结果"""
    import urudendro

    gt_path = ANNOTATIONS_DIR / f"{sample_code}.json"
    dt_path = Path(results_dir) / "labelme.json"
    img_path = IMAGES_NO_BG_DIR / f"{sample_code}.jpg"

    if not gt_path.exists():
        print(f"  ⚠️ 找不到 ground truth: {gt_path.name}")
        return None
    if not dt_path.exists():
        print(f"  ⚠️ 找不到检测结果: {dt_path}")
        return None
    if not img_path.exists():
        print(f"  ⚠️ 找不到图像: {img_path.name}")
        return None

    # 读取 pith 坐标
    cx, cy = None, None
    pith_df = pd.read_csv(PITH_FILE)
    row = pith_df[pith_df["Code"] == sample_code]
    if not row.empty:
        cx = int(row.iloc[0]["cx"])
        cy = int(row.iloc[0]["cy"])

    if cx is None:
        print(f"  ⚠️ 无 pith 坐标，跳过评估")
        return None

    eval_dir = Path(results_dir) / "evaluation"
    eval_dir.mkdir(exist_ok=True, parents=True)

    try:
        precision, recall, f_score, rmse, tp, fp, tn, fn = urudendro.compute_metrics(
            str(dt_path), str(gt_path), str(img_path),
            cx=cx, cy=cy, threshold=0.5, output_dir=str(eval_dir)
        )
        return {
            "sample": sample_code,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "rmse": rmse,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
    except Exception as e:
        print(f"  ❌ 评估失败: {e}")
        return None


def main():
    # 读取 pith 坐标
    pith_df = pd.read_csv(PITH_FILE)
    print(f"共 {len(pith_df)} 个样本")

    # 选择模型
    model_path = MODELS["256_pinus_v1"]
    print(f"使用模型: 256_pinus_v1")

    # 询问处理范围
    total = len(pith_df)
    print(f"\n准备处理 {total} 个样本")

    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / f"detection_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # ==================== 批量检测 ====================
    print("\n" + "=" * 60)
    print("阶段 1: 年轮检测")
    print("=" * 60)

    results_dirs = {}
    for idx, row in tqdm(pith_df.iterrows(), total=total, desc="检测进度"):
        sample_code = row["Code"]
        cx = int(row["cx"])
        cy = int(row["cy"])
        result_dir = run_detection_for_sample(sample_code, cx, cy, model_path, output_dir)
        if result_dir:
            results_dirs[sample_code] = result_dir

    print(f"\n检测完成: {len(results_dirs)}/{total} 个样本成功")

    # ==================== 批量评估 ====================
    print("\n" + "=" * 60)
    print("阶段 2: 评估")
    print("=" * 60)

    all_metrics = []
    for sample_code, result_dir in tqdm(results_dirs.items(), desc="评估进度"):
        metrics = evaluate_sample(sample_code, result_dir)
        if metrics:
            all_metrics.append(metrics)

    # ==================== 汇总结果 ====================
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)

        print("\n" + "=" * 60)
        print("汇总结果")
        print("=" * 60)
        print(f"评估样本数: {len(metrics_df)}")
        print(f"平均 Precision: {metrics_df['precision'].mean():.4f}")
        print(f"平均 Recall:    {metrics_df['recall'].mean():.4f}")
        print(f"平均 F-score:   {metrics_df['f_score'].mean():.4f}")
        print(f"平均 RMSE:      {metrics_df['rmse'].mean():.4f}")
        print(f"\n结果保存在: {output_dir}")
        print(f"详细指标: {output_dir / 'all_metrics.csv'}")
    else:
        print("\n⚠️ 没有成功的评估结果")

    print("\n✅ 全部完成!")


if __name__ == "__main__":
    main()
