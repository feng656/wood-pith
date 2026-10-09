# 2D-OA-Pith

面向**单张局部二维木材横切面**的可观测性自适应髓心反演工程。髓心可以位于图内、图外，或远到当前弧段只能支持方向。工程不把远图外坐标裁到图框，也不把近平行短弧强行解释成高置信二维点。

## 当前实现状态

当前仓库是一个**可运行的研究原型**。它已把密集预测、弧提取和概率几何串成可直接调用的 Python 工作流；项目同时保留原 CLI 作为向后兼容入口，但并不等于文档中所有研究协议都已经训练、校准和真实数据验证。

**已接入训练或推理主链路：**

- `align_corners=False` 像素中心约定、等比例方形坐标及不裁剪的图外坐标；
- 亚像素 polyline 和逐点 `sigma_px` 的软栅格化，包括距离场、双角方向场和实例索引；
- U-Net 类多头前端、近场 Gaussian mixture、远场方向—逆距离 mixture 和 `near/far/infinity/null` 状态头；
- 密集年轮/距离/方差/方向/实例损失，以及前端和概率头的 C4 输出一致性损失；
- 重叠 tile 密集推理、亚像素 ridge/骨架路径、实例嵌入阈值聚类和弧观测生成；
- 训练端混合全图 context 与直接从原始矩形图像一次采样的原生尺度随机 tile；tile 方差按二阶矩融合并换算到整图规范坐标；
- APD 式法线交会、成对交会、神经候选和对数距离初值；
- 近场严格正增量嵌套星形模型、远场/平行层正间隔模型、可位于图外的二维逆向形态校正场；
- 多起点 EM–Student-t 优化、边缘似然 Hessian、Schur 补、秩诊断和模态权重；
- 神经状态先验、几何模态证据与未解析质量的联合状态后验；最终状态由联合后验决定，而不是由最高权重几何解或原始状态头单独决定；
- 条件于当前弧提取/分组的几何层 leave-one-arc-out，以及分组 conformal 分数和阈值拟合；
- unknown/遮挡 mask、匹配多标注者的中心线共识/MAD 方差工具，以及沿弧相关误差 bootstrap 模块；
- 规范/原像素/方形像素毫米坐标 JSON、Hessian Gaussian 椭圆、图外方向箭头和弧段贡献 overlay。

**已定义模块或研究协议，但尚未接入完整主链路：**

- `ArcGraphNet` 和弧图构造已有实现，但尚无训练器、checkpoint 和推理调用；当前年轮分组只用固定嵌入阈值；
- defect logits 已定义并参与 C4 一致性，但缺陷监督和 top-K 拓扑分组尚未接线；unknown mask 与多标注共识已接入数据工具；
- conformal 校准入口支持在独立校准数据上拟合和保存阈值，但推理入口尚不构造/渲染椭圆、多峰或无界扇区的校准集；本工程未附真实数据覆盖率结果；
- Python 推理脚本可通过 `MM_PER_PIXEL=(X, Y)` 输出毫米坐标，但要求两个间距相等；各向异性物理像素必须先重采样为方形像素；
- `escnn` 只是可选依赖占位，尚没有可选编码器或可运行消融；
- 训练已混合全图等比例 context 与覆盖原图不同位置的原生尺度视图，且后者不会先缩小整图再放大；推理使用原分辨率重叠 tile 加全局 context。两者的数值 parity、拼接误差和跨分辨率校准仍需在真实数据上配对实测；
- `docs/experiments.md` 是预注册式验证协议；仓库尚未包含 E0–E10 的完整基线适配器、真实数据结果或性能承诺。

方法细节见 [`docs/method.md`](docs/method.md)，数据规范见 [`docs/data_schema.md`](docs/data_schema.md)，实验设计见 [`docs/experiments.md`](docs/experiments.md)，Python 脚本实验说明见 [`docs/python_script_experiments.md`](docs/python_script_experiments.md)，开源基线和许可证见 [`docs/open_source.md`](docs/open_source.md)。
本次交付实际执行和未能执行的检查见 [`VERIFICATION.md`](VERIFICATION.md)。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

为未来严格群等变编码器消融预装可选依赖：

```bash
pip install -e ".[equivariant]"
```

默认使用 C4 输出一致性。`escnn` 的表示类型和 U-Net skip 需要单独设计；安装该 extra 不会自动启用群等变模型。

## 数据

主清单是 JSONL。所有几何标注保留在**原始图像像素坐标**中；髓心允许远在图外：

```json
{
  "sample_id": "disc07_crop_014",
  "group_id": "tree03",
  "split": "train",
  "image": "images/disc07_crop_014.png",
  "metadata": {"tree_id": "tree03", "disc_id": "disc07"},
  "mm_per_pixel": [0.042, 0.042],
  "pith_px": [-1834.4, 706.2],
  "pith_cov_px": [[2.1, 0.2], [0.2, 3.4]],
  "pith_state": 1,
  "rings_annotated": true,
  "curves": [
    {
      "arc_id": "r18_a",
      "ring_id": 18,
      "points_px": [[14.2, 91.8], [33.7, 89.1], [55.3, 84.4]],
      "sigma_px": [0.7, 0.9, 1.2],
      "visibility": 0.95,
      "annotator_id": "expert_2"
    }
  ]
}
```

`group_id` 必须取最高依赖层级；若同一棵树包含多个圆盘，应像示例一样使用树级 `group_id`，并把 `disc_id` 单独保存在 `metadata`。校准器只按完整 `group_id` 字符串聚合，不会自动解析 `tree/disc` 路径层级。

`pith_state` 是当前观测支持的状态：`0=near, 1=far, 2=infinity, 3=null`。它不是“真实髓心是否有限”的物理标签。只有通过可辨识性实验或严格合成得到的 `infinity/null` 才应监督状态头。推理 JSON 中 `neural_state_probabilities` 是图像网络给出的先验，`joint_state_probabilities` 再融合几何模态证据；无法形成有效局部后验的质量计入 unresolved/null，最终有限、无界或拒识状态依据联合结果，而不是直接阈值化神经先验。

## Python 脚本训练、推理和校准

```bash
python scripts/01_consensus.py   # 仅多标注共识数据需要
python scripts/02_train.py
python scripts/03_infer.py
python scripts/04_calibrate.py
```

所有实验参数都直接在对应脚本顶部编辑，不解析命令行参数。`scripts/03_infer.py` 中 `EXACT_CONTRIBUTIONS=True` 会在保持当前密集预测、弧提取和年轮分组不变的条件下，对每一段弧重跑几何优化；它不是对整条 CNN—连接—模型混合管线的反事实重训练。数据划分必须以最高依赖层级（通常是树）为独立组，再生成圆盘、裁剪、尺度与旋转副本。完整字段说明、calibration JSONL 格式和 E0–E10 实现边界见 [`docs/python_script_experiments.md`](docs/python_script_experiments.md)。

## 当前计算边界

- 软标签栅格器在分块控制内存的同时，仍会让每个输出像素搜索全体年轮线段；计算量近似随“输出像素数 × 线段数”增长，不适合未经评测直接用于超大画布或极密集标注。
- data-only/posterior 信息使用包含全部干扰参数的自动微分全 Hessian；精确 LOAO 会按弧段数重复几何优化，bootstrap 又按重复次数成倍增加代价。它们目前更适合作为离线审计，而非低延迟在线默认项。
- 全图/原生尺度混合训练与 tiled inference 使用同一直接原图采样和坐标约定，但这不等于已经证明数值 parity；正式使用前须按分辨率、tile 大小、重叠率和年轮尺度做配对验证。

## 验证

```bash
pytest -m "not slow"
pytest -m slow
python -m compileall -q src
```

`tests/` 覆盖图外坐标往返、协方差/曲率尺度、远场逆距离趋零连续性、Schur 补、秩退化、亚像素软曲线、模型输出契约、分组 conformal 及合成图外髓心优化。几何优化与 Hessian 强制 `float64`；CNN 可用 AMP。

## 科学边界

本工程只推断二维截面内的髓心和二维潜在形态扰动。没有 CT 或连续轴向切片时，不输出三维节轴，不把未知形变强称为隐藏节，也不声称恢复真实应力—应变场。近平行短弧的联合后验可以退化为无界方向信息或拒识；校准扇区的显式集合构造与渲染仍是未接线部分。
