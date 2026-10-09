# 数据与标注规范

## 1. 独立统计单位

最高层级至少保存 `tree_id/log_id/disc_id/full_image_id/crop_id`。`group_id` 必须直接取数据中最高的统计依赖层级，通常是 `tree_id`；若同一棵树有多个圆盘，不能使用 `tree_id/disc_id` 或 `disc_id` 作为不同组。`disc_id`、`full_image_id` 和 `crop_id` 单独保存在 `metadata`。校准与 split 检查只比较完整 `group_id` 字符串，不会自动解析层级路径。先划分 `train/validation/calibration/test`，再从完整截面生成圆盘裁剪、旋转和尺度副本；同一母组的所有派生图像必须在同一集合。

没有注册到完整截面或其他可靠测量的孤立局部图像，不能用于远图外髓心精度或覆盖率验证。它们仍可用于年轮分割、方向场、实例连接、旋转一致性和域适应。

## 2. 几何坐标

原始标注使用浮点像素中心坐标，左上角像素中心为 `(0,0)`。图像连续边界位于 `-0.5` 与 `W-0.5/H-0.5`。网络变换为：

\[
L=\frac{\max(W,H)}2,\qquad
\widetilde{\mathbf x}=\frac{\mathbf x-((W-1)/2,(H-1)/2)}L.
\]

这一约定等价于 `grid_sample(..., align_corners=False)`。图外点不截断；例如 `(7.3,-2.1)` 是有效规范坐标。

当前工程若检测到两个轴的毫米间距不同，会显式拒绝样本；必须先在物理坐标中重采样为方形像素，再写入相等的 `mm_per_pixel`。保存完整仿射与诱导度量属于未来扩展，当前 schema、损失和几何 refiner 均未消费这类字段。不要在没有同步更新欧氏度量的情况下分别把图像 x/y 拉伸到 `[-1,1]`，否则圆、角度、法线和曲率都会被改变。

## 3. 髓心字段

- `pith_px`: 来自完整图/外部测量的原始像素坐标，可图外；
- `pith_cov_px`: 标注和配准误差的 2x2 协方差；
- `pith_source`: `full_disc/manual_visible/external_measurement`；
- `pith_state`: 可观测性状态，不等同于物理距离类别；
- `mm_per_pixel`: 当前实现要求两个轴数值相等；物理标定来源另存于 `metadata`；
- 派生量：有符号框外距离、`d/L`、方向、逆距离。

## 4. 年轮字段

每条原始年轮保存亚像素 polyline，而非仅保存 1 px mask：

- `arc_id`, `ring_id/year`；
- `points_px`；
- 法向和切向标注标准差 `sigma_n_px(s), sigma_t_px(s)`；当前实现的 `sigma_px` 是保守的法向尺度；
- `visibility`, `unknown_intervals`, `occlusion_reason`；
- 可选 `unknown_mask` 灰度图路径；代码将归一化灰度值 `m` 作为软未知度，使有效监督权重乘以 `1-m`：255 完全排除，0 完全保留，中间值部分降权；
- `open/closed`, `missing/wedging`；
- `annotator_id`, 标注版本及复核状态；
- 同环、邻环、不相容和可能跨缺口连接边。

年轮编号必须说明方向：建议由髓心向外递增。若年份未知但相对顺序可靠，使用连续局部整数编号。**当前 `RingCurve.ring_id` 必须是可转为 `int` 的值，不能为 null。** 完全未知的拓扑应保留在原始标注/候选假设中，待显式选择一个整数分组后再进入当前几何管线；nullable `ring_id` 和由弧段图保留 top-K 分组是假定的未来协议，尚未接线。

## 5. 多标注共识

代表性子集建议至少由 3 位标注者独立描线，并包含部分隔期重复标注与资深专家仲裁。先做年轮身份匹配，再沿共识法向用含标注者偏差的层次模型估计局部偏差、相关长度和 `sigma_ann(s)`。拓扑分歧（某轮是否存在、是否断裂、两段是否相连）必须保存为多个假设/频率，不能压成一个位置方差。

软目标为：

\[
y(\mathbf x)=\exp\left[-\frac{d(\mathbf x,\Gamma)^2}
{2(\sigma_{ann}^2+\sigma_{pix}^2)}\right].
\]

在相邻轮的 Voronoi 中线处截断概率管，防止两条极细近邻年轮粘连。遮挡和无法判断区域使用 unknown mask，不当作背景。

显式共享 `(ring_id, arc_id)` 且具有不同 `annotator_id` 的曲线可先运行 `python scripts/01_consensus.py`；该工具不会把拓扑不同的弧强行平均。

## 6. 缺陷与潜在扰动

节、裂纹、锯痕、污渍、反应木证据使用独立字段，不能与“错误分割”共享一个异常标签。只有同一完整二维截面上有可见节核真值的匹配裁剪，才可标为同平面图外节；其他情况称为未归因二维形态扰动。
