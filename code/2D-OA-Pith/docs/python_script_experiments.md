# Python Script 实验运行说明

本文把仓库原有的训练、多人标注共识、单图推理/弧段贡献分析和分组
conformal 校准从“命令行参数驱动”改成“Python 脚本配置 + Python API 调用”。
四个脚本都不解析命令行参数，也不调用 `subprocess`；实验参数直接在脚本顶部编辑。

> 科学实现边界保持不变：`docs/experiments.md` 中 E0–E10 是预注册式实验协议，
> 不是仓库已经实现完毕的 benchmark。这里脚本化的是当前源码真实具备的可执行主链，
> 不虚构尚不存在的外部基线 adapter、统一 evaluator 或真实数据结果。

## 1. 脚本与原入口对应关系

| Python 脚本 | 作用 | 主要产物 |
|---|---|---|
| `scripts/01_consensus.py` | 多标注者匹配弧的亚像素共识 | `data/consensus.jsonl` |
| `scripts/02_train.py` | 训练密集/概率网络 | `runs/oa_pith/best.pt`、`last.pt`、`metrics.jsonl` |
| `scripts/03_infer.py` | tiled 密集预测、几何后验、贡献度、bootstrap、可选 conformal | `prediction.json`、`contribution.png` |
| `scripts/04_calibrate.py` | 按独立组拟合 split-conformal 阈值 | `conformal_95.json` |

程序化入口统一暴露在 `oapith.workflows`：

```python
from oapith.workflows import (
    run_calibration,
    run_consensus,
    run_inference,
    run_training,
)
```

原来的 console entry point 仍保留，仅用于向后兼容；下面的实验流程不依赖它们。

## 2. 安装

在项目根目录创建环境并安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

脚本会把工作目录切换到项目根目录，因此可从任意目录执行；`configs/base.yaml` 中的
`data.manifest`、`data.root` 等相对路径仍按项目根目录解释。

## 3. 准备数据

主数据清单遵守 `docs/data_schema.md` 的 JSONL 格式。至少检查：

- `group_id` 使用最高统计依赖层级，通常是树，而不是 crop 或 disc；同组不得跨 split。
- `pith_px` 保留原始像素坐标，允许在图外，禁止裁到图框边缘。
- `ring_id` 当前必须可转换为整数；曲线使用亚像素 `points_px`。
- 若提供 `mm_per_pixel`，当前实现要求 X/Y 两轴间距相同；否则先重采样为方形物理像素。
- split 应先按独立组划分，再生成同树的裁剪、旋转和尺度副本。

### 3.1 可选：先做多人标注共识

编辑 `scripts/01_consensus.py`：

```python
INPUT_MANIFEST = PROJECT_ROOT / "data/raw_multi_annotator.jsonl"
OUTPUT_MANIFEST = PROJECT_ROOT / "data/consensus.jsonl"
```

然后运行：

```bash
python scripts/01_consensus.py
```

它只对显式匹配的 `(ring_id, arc_id)` 多标注弧求稳健共识；拓扑不同的弧不会被强行平均。
如果训练要使用共识结果，再把 `configs/base.yaml -> data.manifest` 指向输出清单。

## 4. 训练

先编辑 `configs/base.yaml` 的数据路径、网络、训练和输出设置，再检查
`scripts/02_train.py` 顶部：

```python
CONFIG_PATH = PROJECT_ROOT / "configs/base.yaml"
RESUME_CHECKPOINT = None
DEVICE = None
ALLOW_LEGACY_CHECKPOINT = False
```

运行：

```bash
python scripts/02_train.py
```

`DEVICE=None` 时自动优先 CUDA；也可在脚本中写成 `"cuda:0"` 或 `"cpu"`。
断点继续时把 `RESUME_CHECKPOINT` 指向完整 trainer checkpoint。不要为了加载语义不明的旧权重
随意开启 `ALLOW_LEGACY_CHECKPOINT`；当前 checkpoint 会校验坐标、预处理、模型构造和
`switch_radius` 契约。

主要输出位于 `configs/base.yaml -> output_directory`：

- `best.pt`：验证损失最优权重；
- `last.pt`：最后一轮；
- `epoch_XXXX.pt`：按 `checkpoint_every` 保存；
- `metrics.jsonl`：每轮训练/验证指标。

## 5. 单图推理、可行度与年轮弧贡献

编辑 `scripts/03_infer.py` 的必需路径：

```python
CONFIG_PATH = PROJECT_ROOT / "configs/base.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "runs/oa_pith/best.pt"
IMAGE_PATH = PROJECT_ROOT / "sample.png"
OUTPUT_JSON = PROJECT_ROOT / "runs/oa_pith/prediction.json"
OVERLAY_PNG = PROJECT_ROOT / "runs/oa_pith/contribution.png"
```

物理尺度未知时保持：

```python
MM_PER_PIXEL = None
```

已知方形像素标定时才写实际数值，例如：

```python
MM_PER_PIXEL = (0.042, 0.042)
```

贡献/标注误差审计默认配置为：

```python
EXACT_CONTRIBUTIONS = True
BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_CORRELATION = 0.92
```

运行：

```bash
python scripts/03_infer.py
```

这三个审计参数都可能很慢。`EXACT_CONTRIBUTIONS=True` 会对每条已提取弧执行一次
leave-one-arc-out 几何再优化；`BOOTSTRAP_REPLICATES=100` 会额外重复 100 次相关弧误差扰动。
快速检查可暂时设置 `EXACT_CONTRIBUTIONS=False`、`BOOTSTRAP_REPLICATES=0`，正式贡献实验再恢复。

`prediction.json` 中重点字段：

- `state_decision`：`near/far/infinity/null`；这是联合神经先验和几何模态证据后的状态。
- `output_type`：有限多模态、无界/方向型或 null/unresolved 等输出拓扑。
- `modes[*].center_pixel_unclipped`：不截边的原像素髓心；可在图外。
- `modes[*].data_information`：由图像弧数据支持的信息秩、条件数、协方差等，可用于可辨识性审计。
- `modes[*].posterior_information`：加入先验后的后验信息；不能与 data-only 信息混为一谈。
- `arc_contributions[*].information_gain`：删除该弧前后的 data pseudo-logdet 信息变化；模型切换或非有界情形可能为 `null`。
- `arc_contributions[*].impact_vector` / `far_chart_impact`：删弧后位置或远场方向/逆距离变化。
- `arc_contributions[*].structural_required`：删弧导致 data rank 下降，说明该弧具有结构性必要性。
- `arc_contributions[*].model_switched`：删弧导致近/远/无穷或扰动模型分支切换，此时不宜把标量信息增益作普通可比量。
- `annotation_bootstrap`：按弧相关标注误差的重复推断结果。
- `conformal.prediction`：供独立校准集拟合阈值的 mixture prediction。

`contribution.png` 中年轮弧按贡献着色，结构必需弧为紫红色；黄色标记/箭头表示当前几何
解。图中的局部 Gaussian 95% 椭圆不是 conformal 集。

### 5.1 贡献度结论的边界

当前 exact LOAO 固定 CNN 密集预测、弧提取和当前分组，只在删除一条弧后重跑几何权重、
形状 nuisance、模型分支与 Hessian。因此它回答的是“在当前已提取弧条件下，该弧对几何后验
有多重要”。它不是删除原始像素证据后重跑 CNN—连接—分组—几何全过程的全管线反事实。
`docs/experiments.md` 的 E9 明确把后者列为待实现实验。

## 6. 分组 conformal 校准

### 6.1 先生成独立 calibration split 的预测

对 calibration split 中的每个样本，用 `scripts/03_infer.py` 生成独立的 `prediction.json`。
不要用训练或 validation 样本代替独立校准组。每个用于校准的预测必须包含
`conformal.prediction`；如果样本因“无可用年轮弧”被直接拒识，当前实现没有为该特殊分支
构造 conformal 集，不能静默丢掉后再宣称无条件覆盖率。

然后建立 `data/calibration_predictions.jsonl`。每行恰好包含：

```json
{"prediction_json":"runs/oa_pith/calibration/tree03_crop01.json","truth_normalized":[-0.37,1.82],"group_id":"tree03"}
```

其中 `truth_normalized` 必须按实际图像尺寸从原像素真值计算，不能裁边：

```text
L = max(W, H) / 2
x_norm = (x_px - (W - 1) / 2) / L
y_norm = (y_px - (H - 1) / 2) / L
```

同一树的多个 crop 使用相同 `group_id`。默认 `GROUP_REDUCTION="max"` 会先取每组最不利
nonconformity score，再对独立组做有限样本分位数。

### 6.2 拟合阈值

编辑 `scripts/04_calibrate.py`：

```python
CALIBRATION_MANIFEST = PROJECT_ROOT / "data/calibration_predictions.jsonl"
OUTPUT_CALIBRATOR = PROJECT_ROOT / "runs/oa_pith/conformal_95.json"
ALPHA = 0.05
GROUP_REDUCTION = "max"
```

运行：

```bash
python scripts/04_calibrate.py
```

对 95% split conformal，如果独立可交换校准组少于 19 个，正确有限样本阈值可能为无穷大；
脚本会如实保存/打印该状态，不应用最大观测 score 偷换无穷阈值。

### 6.3 在新图上携带校准器推理

把 `scripts/03_infer.py` 改为：

```python
CALIBRATOR_PATH = PROJECT_ROOT / "runs/oa_pith/conformal_95.json"
```

再运行推理。当前源码会把 `alpha/threshold/groups` 写入 `prediction.json`，但按项目文档，
尚未把集合 `A(I,c) <= q` 反演/渲染成 conformal 椭圆、多峰区域或无界扇区；overlay 的
Gaussian 椭圆不能当作校准集合。

## 7. 与 E0–E10 实验协议的关系

| 协议 | 当前脚本可直接提供 | 仍需另行实现/组织 |
|---|---|---|
| E0 可辨识性相图 | `data_information`、真实定位误差所需基础量 | 真值弧扫描器、参数网格、统一相图 evaluator |
| E1 端到端分层定位 | 训练 + 推理主链 | 按连续距离/缺陷等分层的批量 evaluator 与真实数据结果 |
| E2/E3 坐标、参数化、模型复杂度 | 现有坐标/近远场/多几何分支实现 | 成套 ablation 配置、配对统计 |
| E4 细线标注误差 | 共识工具、`sigma_px`、相关 bootstrap | 完整多标注误差实验矩阵 |
| E5 降权/剪裁 | refiner 可靠度与 Student-t/EM 基础 | 系统假边/漏段/共同形变消融 |
| E6 旋转等变 | 训练中的 C4 一致性 | 文档要求的完整后验/贡献/置信集 evaluator |
| E7 同截面图外扰动 | 图外几何与扰动分支 | 配对裁剪数据构建与距离扫描 |
| E8 置信度 | 联合状态、Hessian 后验、conformal 阈值 | 50/80/90/95% 全评测、NLL/energy/AURC 批量报告 |
| E9 弧段解释 | 当前条件几何层 exact LOAO | 全管线 LOAO、完整删除/异常 AUPRC 实验 |
| E10 跨域泛化 | 同一训练/推理接口 | 跨树种/设备/表面处理 split 与统计报告 |

因此，这四个脚本是“当前工程可执行实验链”的完整 Python-script 等价入口；如果论文目标是
宣称 E0–E10 已全部完成，还必须按照 `docs/experiments.md` 补数据构建、基线 adapter、批量
evaluator、组级统计和锁定测试结果。

## 8. 建议的正式复现顺序

1. 锁定树级 split 和随机种子；若需要，运行 `01_consensus.py`。
2. 配置 `configs/base.yaml`，运行 `02_train.py`，保存配置和 checkpoint。
3. 在 validation/calibration/test 上分别生成预测；贡献实验打开 exact LOAO，速度实验关闭。
4. 只用 calibration 独立组拟合 `04_calibrate.py`，固定阈值后再查看 locked test 覆盖。
5. locked test 只做预注册指标/子组统计，不重新调阈值或超参数。
6. 报告树级样本数、运行配置、硬件/软件版本、随机种子、运行时/显存，以及文档列出的失败条件。

## 9. 开发验证

改动源码后至少运行：

```bash
python -m compileall -q src scripts tests
pytest -m "not slow"
```

优化较重的合成几何测试另跑：

```bash
pytest -m slow
```
