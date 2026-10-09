# RAC-Pith v2：局部年轮弧树髓定位与反事实子弧贡献分析

> 面向“局部木材横切面图像 + 已标注年轮线”的最终综合方案  
> 修订日期：2026-09-06  
> 融合材料：`pith_localization_urudendro_synthesis.md` 与 `ArcPith_GTArc_SplitRole_Frozen_Final_Plan_v4.md`  
> 核心目标：可靠定位可能位于视野外的生物学树髓，并说明每条年轮弧或子弧使结果趋近真值还是偏离真值。

---

## 0. 最终结论

推荐采用 **RAC-Pith v2（Robust Arc-Consensus Pith）**。它不是一个高自由度的椭圆、Eikonal 或深度网络模型，而是一个低自由度、分层且可审计的几何方法：

```text
连续年轮弧与图像质量
        ↓
半径无关约束的凸共同中心初值
        ↓
二维鲁棒共同中心 VarPro 精化
        ↓
远场方向快速扫描；仅在退化时运行弱方向一维逆距离剖面
        ↓
POINT / RANGE / RAY(AXIS) / MULTIMODAL / REJECT
        ↓
冻结定位器后的 ring / arc / subarc 完整删除重求解
        ↓
有 GT：真实正负贡献
无 GT：交叉拟合支持/冲突贡献及校准概率
```

最终设计遵循八项决定：

1. **有限髓心不被限制在裁窗内**，也不把搜索边界上的点伪装成远处髓心。
2. 离散标注点只用于恢复连续曲线；插值加密、顶点复制和人为切段不能增加证据量。
3. 用一个可凸求解的共同中心线性化得到稳定初值，再用仅含二维中心变量的鲁棒几何目标精化。
4. 共同圆心模型**不会自动补足短弧的绝对距离**；距离弱时必须输出范围或方向，而不是伪精确点。
5. 图像主要用于评估标注边界是否受支持、局部是否异常；不再训练一个重复检测年轮或直接回归髓心的大网络。
6. 拟合权重始终非负；“负贡献”是估计后的反事实影响，绝不是负损失权重。
7. 正式贡献来自删除整组证据后对**同一冻结管线完整重求解**；高残差、高杠杆或低权重本身都不等于负贡献。
8. 第一版的贡献模块只负责解释和质检，**不反馈改变最终坐标**。自动删弧只能作为以后经独立树验证的条件模块。

相较原综合稿，本版最重要的新增是：

- 一个不需要切向微分、由半径无关恒等式构造的凸共同中心初值；
- 一个低成本远场方向扫描，以及只在几何退化时触发的弱方向逆距离剖面；
- `ring → visible arc → subarc` 的证据守恒和多尺度贡献协议；
- 无真值时的交叉拟合支持/冲突贡献，而不是用残差或杠杆武断地产生正负号；
- 生物学髓心与局部径向汇聚中心的明确区分。

---

## 1. 任务、目标与适用边界

### 1.1 输入

一个样本至少包含：

- 局部横切面图像 \(I\)，视野为 \(\Omega\)；
- 有序年轮线 \(\{\Gamma_r\}_{r=1}^{R}\)，知道哪些点属于同一条 parent ring；
- 裁切后每条 ring 的一个或多个可见连通弧；
- source image 与 crop 坐标变换；
- 像素—毫米标定，若可获得；
- 训练/评测阶段可选的髓心真值 \(c^*\) 及其标注不确定度。

推荐的证据层级为：

```text
tree
└── section
    └── crop
        └── parent ring
            └── visible arc / fragment
                └── contiguous subarc
                    └── quadrature node
```

### 1.2 输出

主输出包括：

1. 原始最优候选 `pith_raw`；
2. 可使用的定位结果：有限坐标、方向或距离区间；
3. 几何状态 `POINT / RANGE / RAY / AXIS / MULTIMODAL / REJECT`；
4. 各向异性不确定性，而不是只有一个标量置信度；
5. ring、visible arc 和 subarc 三层贡献；
6. 每组证据的方向、距离、排模态、杠杆和冲突角色；
7. 图像—标注质量、模型风险、搜索稳定性及失败原因。

`pith_raw` 便于调试和可视化，但当状态不是 `POINT` 时，不得把它当作可靠有限髓心使用。

### 1.3 需要区分的两个中心

方法的几何模型首先估计局部年轮的**径向汇聚中心** \(c_{\mathrm{rc}}\)。研究目标则是生物学髓心 \(c^*\)。偏心生长、反应木、干燥畸变或共同标注偏差可能导致：

\[
c_{\mathrm{rc}} \ne c^*.
\]

在有完整截面的开发数据上，定义：

\[
B_{\mathrm{target}}
=
\frac{\|c_{\mathrm{rc}}^{\mathrm{full}}-c^*\|}
{D_{\mathrm{section}}}.
\]

据此将数据域标为：

- `TARGET_ALIGNED`：共同汇聚中心与生物学髓心在应用容差内一致；
- `ECCENTRIC_GT`：髓心真值可靠，但共同中心存在系统偏差；
- `TARGET_UNKNOWN`：局部数据不足以判断两者关系。

主结果必须同时报告 aligned-domain 与 full-domain。不能通过删除 `ECCENTRIC_GT` 样本虚增精度，也不能把局部内部拟合良好直接解释为生物学髓心正确。

### 1.4 一个必须接受的不可辨识事实

短弧接近直线时，髓心方向通常比髓心距离容易确定。设弧的半张角为 \(\alpha\ll1\)，常见噪声模型下有量级关系：

\[
\operatorname{Std}(\hat c_{\mathrm{radial}})
=O\!\left(\frac{\sigma}{\sqrt n\,\alpha^2}\right),
\qquad
\operatorname{Std}(\hat c_{\mathrm{tangential}})
=O\!\left(\frac{\sigma}{\sqrt n\,\alpha}\right).
\]

因此，本任务不可能对所有局部裁窗都可靠输出有限点。允许输出 `RANGE` 或 `RAY/AXIS` 不是算法失败，而是对观测信息边界的正确表达。

---

## 2. 连续弧、图像证据与证据守恒

### 2.1 从离散折线恢复连续弧

对每个 visible arc：

1. 以弦长参数拟合轻度平滑的三次 B-spline 或局部二次曲线；
2. 平滑尺度在开发集上按重复标注稳定性冻结，不按当前样本 GT 调节；
3. 去除端点导数不稳定的短保护区；
4. 按物理弧长进行数值积分或等弧长重采样；
5. 真实缺口必须拆成两个 fragment，不能用样条跨越；
6. 自交、错误回折、错误 ring ID 进入数据错误诊断。

只有在切向诊断或方向贡献中需要：

\[
t(s)=\frac{\gamma'(s)}{\|\gamma'(s)\|},
\qquad t\sim -t.
\]

核心有限中心初值不依赖切向微分，因此对粗折线更稳。

### 2.2 Parent-ring 证据预算

设第 \(r\) 条 parent ring 在完整输入中的有效总长度为 \(L_r^0\)，默认每条合格 ring 的总预算相同：

\[
\alpha_r=\frac1R,
\qquad
d\mu_r(s)=\alpha_r\frac{ds}{L_r^0}.
\]

于是：

- 点采样加密不会增大该 ring 的权重；
- 同一 ring 被裁成多个 fragment 不会增加总预算；
- 子弧越切越细不会凭空产生更多信息；
- 对子弧做删除贡献时，删除该段对应的预算份额，**不把这部分预算重新分给相邻子弧**。

删除后，半径和均值等数据依赖量在剩余证据上重新估计，但原始 \(L_r^0\) 与证据密度保持冻结。这样测得的是“拿掉该段证据”的影响，而不是“拿掉该段并同时上调其兄弟段”的混合效应。

默认采用 equal-parent budget。作为消融，可以比较带长度饱和与质量上限的预算，但任何 ring 都不应因点多或弧长极大而无上限地主导结果。

### 2.3 图像怎样进入模型

既然年轮线已经标注，图像只提取少量候选无关特征：

1. `boundary_support`：沿标注法线两侧窄带的梯度或对比度；
2. `orientation_agreement`：结构张量主方向与标注切向的一致性；
3. `texture_anomaly`：节疤、裂纹、锯痕、污渍、模糊等局部异常；
4. `polarity_consistency`：数据集边界定义是否一致；
5. 可选 `ray_support`：只有在木射线方向能被高置信地区分时才作为辅助方向证据。

这些特征优先映射为候选无关的测量尺度 \(\sigma_r(s)\)，并设置训练集校准的上下限。不要同时把同一个质量量重复用于权重、残差尺度和后验删弧。

以下做法明确禁止：

- 用灰度极性直接规定贡献正负；
- 因为某段残差大或杠杆高，就先验赋负权；
- 用一个大网络重新覆盖人工年轮标注几何；
- 用当前候选的残差反向估计其误差尺度，从而让候选通过“放大方差”降低损失。

---

## 3. 定位方法：一个凸初值、一个二维精化、两个轻量远场检查

### 3.1 坐标规范

以 crop 中心 \(o\) 为原点，以 crop 对角线或长边 \(D_{\mathrm{FOV}}\) 为统一尺度：

\[
\tilde x=\frac{x-o}{D_{\mathrm{FOV}}}.
\]

所有拟合在归一化坐标中完成，最终再还原为像素和毫米。图外坐标不得被 clip 到图像边界。

下文定位公式中的 \(x,c,s\) 默认均为归一化坐标，因此 crop 尺度为 1；只有带 \(\mathrm{mm}\) 下标或明确写出 \(D_{\mathrm{FOV}}\) 时才使用物理坐标。

### 3.2 由精确的半径无关约束构造凸共同中心初值

对第 \(r\) 条 ring 的连续弧，定义剩余证据上的均值：

\[
\bar x_r=\frac{1}{\mu_r(E_r)}\int_{E_r}x\,d\mu_r,
\qquad
\overline{\|x\|^2}_r
=\frac{1}{\mu_r(E_r)}\int_{E_r}\|x\|^2\,d\mu_r.
\]

若该 ring 属于以 \(c\) 为圆心的圆，则：

\[
\|x-c\|^2=R_r^2.
\]

将该式减去本 ring 的弧长平均，可以在理想圆约束中精确消去 \(R_r^2\) 与 \(\|c\|^2\)：

\[
\boxed{
2(x-\bar x_r)^Tc
=
\|x\|^2-\overline{\|x\|^2}_r
}.
\]

令：

\[
a_r(x)=2(x-\bar x_r),
\qquad
b_r(x)=\|x\|^2-\overline{\|x\|^2}_r.
\]

用这些半径无关的线性约束构造凸鲁棒初值：

\[
\boxed{
\hat c_0
=
\arg\min_c
\sum_r\int_{E_r}
\rho_H\!\left(
\frac{a_r(x)^Tc-b_r(x)}{\sigma_{\mathrm{alg},r}(x)}
\right)d\mu_r(x)
}.
\]

它具有四个优点：

- 不需要逐条短弧独立拟圆；
- 不需要切向或二阶曲率；
- 目标对 \(c\) 是凸的，IRLS 每步只解 \(2\times2\) 线性系统；
- 可直接得到弱方向与病态程度。

这里精确的是半径无关的线性恒等式；异方差尺度和 pseudo-Huber 聚合得到的是凸初始化 surrogate，并不等价于对原始圆模型做一次精确鲁棒 profile。代数残差仅用于初值和几何诊断，最终物理估计仍由几何 VarPro 给出。

全文的 \(\rho_H\) 默认表示平滑 pseudo-Huber：

\[
\rho_H(z)=\delta^2\left(\sqrt{1+(z/\delta)^2}-1\right).
\]

\(\sigma_{\mathrm{alg}}\) 的单位是归一化长度的平方；它由位置标注误差经过上述代数式的一阶传播，或在独立开发数据上单独校准，不能直接复用单位为长度的 \(\sigma_x\)。

IRLS 最终信息矩阵记为：

\[
H_0=\sum_r\int w_r(x)a_r(x)a_r(x)^T\,d\mu_r(x).
\]

设特征值 \(\lambda_s\ge\lambda_w\)，对应特征向量 \(v_s,v_w\)。\(\lambda_w/\lambda_s\) 小表示沿 \(v_w\) 的髓心距离弱可辨。

### 3.3 切向交会只作第二初值和一致性检查

用正确的切向零投影约束：

\[
t(s)^T(c-\gamma(s))=0.
\]

可再求一个 pseudo-Huber/IRLS 初值：

\[
\hat c_t
=
\arg\min_c
\sum_r\int
\rho_H\!\left(
\frac{t(s)^T(c-\gamma(s))}{\sigma_{\perp,r}(s)}
\right)d\mu_r(s).
\]

\(\hat c_t\) 不替代凸共同中心初值；两者明显分歧时触发扩大剖面或模型风险诊断。这里不能写成 \(n^T(c-q)=0\)，因为若 \(n\) 是法向，该式会错误地把髓心放在切线上。

### 3.4 二维鲁棒共同中心 VarPro 精化

给定中心 \(c\)，第 \(r\) 条 ring 的半径由鲁棒位置量得到：

\[
\hat R_r(c)
=
\arg\min_R
\int_{E_r}
\rho_H\!\left(
\frac{\|x-c\|-R}{\sigma_{x,r}(x)}
\right)d\mu_r(x).
\]

代回后仅优化二维中心：

\[
\boxed{
J_{\mathrm{geo}}(c)
=
\sum_r\int_{E_r}
\rho_H\!\left(
\frac{\|x-c\|-\hat R_r(c)}{\sigma_{x,r}(x)}
\right)d\mu_r(x)
}.
\]

从 \(\hat c_0\)、\(\hat c_t\) 以及条件式弱轴剖面得到的少量候选启动二维优化。所有候选必须用同一个 \(J_{\mathrm{geo}}\) 重评分。

尺度 \(\sigma_x\) 来自重复标注、标注扰动或训练集 pooled robust scale，并设物理噪声下限；不能用单条 ring 当前残差独立估计，否则完美拟合的短弧会产生尺度塌缩。

为保证完整删除重求解可复现，工程注册表还必须冻结：每条 ring 的最小剩余证据质量、内层半径求解容差、外层梯度/步长容差、最大迭代数和失败处理。平滑 pseudo-Huber 使有效证据下的一维半径目标严格凸；若证据质量不足或内层/外层未收敛，分别返回 `INSUFFICIENT_RING_EVIDENCE` 或 `SOLVER_UNSTABLE`，不得沿用上一次半径或坐标。

### 3.5 为什么 VarPro 仍不能保证有限距离

令 \(c=Ru\)，\(R\to\infty\)。对有界裁窗中的点 \(x\)：

\[
\|x-Ru\|=R-u^Tx+O(R^{-1}).
\]

消去每轮半径后，残差趋向该 ring 内 \(u^Tx\) 的中心化值。若多条可见弧几乎是相互平行的直线，取 \(u\) 为其法向时，这个极限可以很小。因此：

> “共同中心 + 每轮独立半径”利用了距离信息，但不能从近直线短弧中创造不存在的绝对距离信息。

这也是原方案必须修正的关键点。

### 3.6 先做低成本的全方向远场极限扫描

只检查 \(v_w\) 方向可能漏掉另一个远场方向。因此，对每个样本都计算共同中心目标在无穷远处的解析极限。令 \(u=(\cos\phi,\sin\phi)\)，并对每条 ring profile 掉投影常数：

\[
m_r(u)
=
\arg\min_m
\int_{E_r}
\rho_H\!\left(
\frac{u^Tx-m}{\sigma_{x,r}(x)}
\right)d\mu_r(x),
\]

\[
\boxed{
J_\infty(u)
=
\sum_r\int_{E_r}
\rho_H\!\left(
\frac{u^Tx-m_r(u)}{\sigma_{x,r}(x)}
\right)d\mu_r(x)
}.
\]

因为 pseudo-Huber 对符号对称，\(u\) 与 \(-u\) 在纯远场损失上等价，所以只需在 \(\phi\in[0,\pi)\) 做一个便宜的一维角度网格并精化低谷。这个量正是逐 ring profile 半径后的径向距离残差远场极限，不是用“大半径近似点”代替无穷远。

任何拟输出 `POINT` 的样本都必须满足：

- 最优远场方向与有限最优解之间有经校准的目标间隔；
- 加密角度网格后最小远场方向和间隔稳定；
- 不存在与有限谷等价但方向分离的远场低谷。

若不满足，不能签发 `POINT`。

### 3.7 仅在退化时运行弱方向或竞争方向的一维距离剖面

当出现下列任一条件时触发：

- \(\lambda_w/\lambda_s\) 很小；
- \(\hat c_0\) 与 \(\hat c_t\) 明显分歧；
- VarPro 解远离裁窗或对初值敏感；
- 远场扫描与有限解的目标间隔不足；
- 最优解靠近预设的宽松物理边界。

以最佳有限 VarPro 解 \(\hat c_f\) 为锚点，先沿弱特征方向计算：

\[
P_w(s)
=
\min_{\eta\in I_s}
J_{\mathrm{geo}}
\big(\hat c_f+s\,v_w+\eta v_s\big),
\]

其中 \(I_s\) 是由强方向局部支持区给出的有限区间，防止 \(\eta\) 随 \(s\) 同比例发散、把“一维弱轴剖面”悄然变成任意远场方向搜索。

使用紧致坐标：

\[
z=\frac{s}{1+|s|}
\in(-1,1),
\qquad
s(z)=\frac{z}{1-|z|},
\qquad
P_z(z)=P_w(s(z)).
\]

\(z=\pm1\) 表示沿 \(\pm v_w\) 的两个远场端点，并用上一节的解析极限计算。如果 \(J_\infty(u)\) 发现与 \(v_w\) 不一致的竞争远场方向，则沿每个竞争方向补做同样的一维距离剖面；仍无法覆盖时，强制进入小型二维紧致方向—逆距离审计。

支持集：

\[
\mathcal S_\Delta
=
\{z:P_z(z)\le P_{\min}+\Delta\}.
\]

\(\Delta\) 由重复标注扰动和独立校准树冻结；数据不足时只称“稳定域”，不能宣称有严格覆盖概率。

完整 bootstrap 若在后续发现新的长尾或模态，必须回到本阶段扩大剖面并重新判定状态；最终状态以重跑结果为准。

### 3.8 少量多初值与搜索充分性检查

默认候选来自三个互补来源：

1. 凸共同中心 \(\hat c_0\)；
2. 切向交会 \(\hat c_t\)；
3. \(J_\infty\) 的方向低谷及相应距离剖面的有限局部谷底。

整弧圆拟合可以作为调试 seed，不进入最终评分。下列情况强制触发小型确定性二维紧致方向—逆距离网格，而不是继续相信局部结果：

- 有限候选、弱轴剖面与远场扫描不一致；
- 出现多谷、数值贴边或拟输出 `POINT` 但远场目标间隔不足；
- 竞争远场方向不能被一维剖面覆盖；
- 删除重求解产生新的未搜索区域。

检查内容包括：

- 不同启动是否收敛到相同或明确分离的谷底；
- 角度扫描和距离剖面分辨率加倍后，谷底、端点和支持区是否稳定；
- 解是否由数值边界人为截断；
- ring 顺序是否出现大面积违反；
- 插值加密、fragment 重切和坐标变换是否改变结果。

二维审计仍不充分时，结果降为 `REJECT: SEARCH_INADEQUATE`，不得输出 `POINT`。这应命名为 `search_adequacy_check`，而不是没有全局下界证明的“全局最优证书”。

### 3.9 年轮顺序只作低容量结构门

若 `ring_order` 可靠，候选中心下的半径应大体满足：

\[
\hat R_{r+1}(c) > \hat R_r(c).
\]

考虑偏心和标注误差后，只把它用于：

- 排除明显镜像方向；
- 在近等价候选间做 tie-break；
- 将大面积次序违反标记为 `RING_ASSOCIATION_RISK`；
- 配合边界梯度极性把无向 `AXIS` 升为有向 `RAY`。

不假设轮宽相等，也不把强单调罚项用于硬拉动坐标。

### 3.10 几何状态

| 状态 | 操作定义 | 可使用输出 |
|---|---|---|
| `POINT` | 单一有限紧致谷底；方向、距离、重采样和多初值均稳定 | 有限髓心与不确定区 |
| `RANGE` | 方向稳定，有限距离区间闭合但过宽 | 方向 + 距离区间；原始点仅供参考 |
| `RAY` | 支持集触及一个远场端点，内外方向已定 | 有向髓心方向和距离下界 |
| `AXIS` | 支持集触及远场且正反方向未消歧 | 无向轴，不输出可靠有限距离 |
| `MULTIMODAL` | 存在两个以上稳定、分离的谷底 | 候选集合及各自支持区 |
| `REJECT` | 数据非法，或搜索/数值结果不稳定 | 失败原因与修复建议 |

状态按以下互斥优先级确定：

```text
1. 数据、数值或搜索不充分
   -> REJECT
2. 紧致方向—距离空间中有两个以上分离的持久支持分量
   （包括“有限分量 + 分离的远场分量”）
   -> MULTIMODAL
3. 仅一个连通分量，但触及两个远场端点
   -> 稳定为同一轴时 AXIS；轴本身也不稳定时 REJECT
4. 仅一个连通分量并触及一个远场端点
   -> 可由 ring order/极性定向时 RAY，否则 AXIS
5. 不触及远场，有限距离支持闭合但过宽
   -> RANGE
6. 单一、紧致、有限且通过几何与搜索稳定门
   -> POINT
```

有限谷与远场谷若属于不同支持分量，必须判为 `MULTIMODAL`；不能因为有限谷的点估计更方便就忽略远场分量。模型风险是与上述几何状态并列的字段：几何上可以是 `POINT`，同时标为 `MODEL_RISK_HIGH / TARGET_UNKNOWN` 并禁止生产使用；不能用 `POINT_BLOCKING` 等风险标签悄然改写几何状态。

近裁窗中心的有限解使用 \(x,y\) 支持椭圆判断，避免因极角在原点附近无定义而误判；只有图外和远场解才使用方向—逆距离宽度。

---

## 4. 不确定性与模型风险

### 4.1 三类不确定性不能混成一个分数

1. **测量不确定性**：重复标注或连续移动块扰动曲线位置和切向；
2. **几何可辨识性**：二维局部曲率、弱轴剖面和是否触及远场；
3. **模型风险**：偏心生长、共同变形或生物学髓心与径向中心不一致。

建议输出：

- `POINT`：bootstrap 样本、95% 椭圆、长短轴及覆盖级别；
- `RANGE`：方向区间、有限距离区间和 log-range 宽度；
- `RAY/AXIS`：方向区间、距离下界、是否无有限上界；
- `MULTIMODAL`：所有持久模态，而不是压成一个巨大椭圆。

### 4.2 重采样层级

- 点级样本相邻性很强，不做独立点 bootstrap；
- 曲线测量误差用连续移动块或重复标注扰动；
- parent-ring 数量足够时做 ring-level bootstrap；
- ring 少时优先使用弱轴 profile 和 leave-one-ring-out 列表；
- benchmark 的统计区间最终按 biological tree 聚类。

### 4.3 模型风险只诊断，不用高自由度模型掩盖

检查：

- 不同 ring 的残差是否有一致的低频方向趋势；
- leave-one-ring-out 是否产生异常散布或状态翻转；
- 完整截面的 \(B_{\mathrm{target}}\) 是否偏大；
- 图像是否显示反应木、节疤、裂纹或全局收缩迹象；
- ring-order 是否系统违反。

第一版不默认拟合共同椭圆、逐轮中心漂移、一般生长场或神经隐式场。只有在独立树验证中明确改善髓心 P90/P95 和灾难误差，且不把 `RANGE/AXIS` 伪升级为 `POINT` 时，才考虑一个低容量、有收缩的偏差修正器。

---

## 5. 弧段正负贡献：定义、计算与解释

### 5.1 先冻结估计器，再谈贡献

令完整定位管线为：

\[
\mathcal M:\mathcal E\mapsto
(\hat c,\mathcal S,\text{state},\text{modes}).
\]

产生正式贡献标签前，必须冻结：

- 曲线平滑与端点规则；
- 图像质量到测量尺度的映射；
- Huber 核、尺度下限与 parent budget；
- 初值、剖面和搜索策略；
- 状态与支持区规则；
- 可选模型风险处理。

对证据组 \(G\)：

\[
M_{\mathrm{all}}=\mathcal M(\mathcal E),
\qquad
M_{-G}=\mathcal M(\mathcal E\setminus G).
\]

删除后必须重新计算数据依赖的均值、半径、鲁棒权重、二维精化、弱轴剖面、模态和状态。完整解可以作 warm start，但不能省略其余固定搜索路径。

### 5.2 三层贡献分别计算，不能相加

正式层级：

1. `leave-one-ring-out`：最稳定，必须优先；
2. `leave-one-visible-arc-out`：识别裁断、裂纹或局部异常；
3. `leave-one-subarc-out`：生成细粒度贡献图。

子弧采用预注册的两到三个物理尺度和两个切分相位。一个实用起点是 5/10/20 mm；没有物理标定时可用 visible arc 长度的 5%/10%/20%。最终尺度必须结合图像分辨率、标注相关长度和验证集稳定性冻结。

重要性质：

- LOO 是相对于“当前其他弧”的条件边际作用；
- 相邻子弧高度相关；
- 切得越细，单块作用一般越小；
- 子弧 LOO 之和不等于整弧或整轮 LOO；
- 符号随尺度或相位翻转时，细粒度结果应标为 `SCALE_UNSTABLE`，回退到更高层级解释。

### 5.3 有 GT 时的真实正负贡献

只有当完整与删除后结果都通过状态门并被判为 `POINT`（或预注册的等价有限点状态）时，主指标才使用直观的物理误差变化。`RANGE` 的 raw finite candidate 不满足这一条件：

\[
\boxed{
C_{G,\mathrm{mm}}^{GT}
=
\|\hat c_{-G}-c^*\|_{\mathrm{mm}}
-
\|\hat c-c^*\|_{\mathrm{mm}}
}.
\]

- \(C_{G,\mathrm{mm}}^{GT}>0\)：保留 \(G\) 使结果更接近真值，正贡献；
- \(C_{G,\mathrm{mm}}^{GT}<0\)：保留 \(G\) 把结果推离真值，负贡献；
- 接近 0：冗余或当前分辨率下不可区分。

若缺少毫米标定，同式先以像素和 \(D_{\mathrm{FOV}}\) 归一化单位报告，不能把像素值误写成物理误差。

同时报告平方误差贡献：

\[
C_{G,\mathrm{sq}}^{GT}
=
\|\hat c_{-G}-c^*\|^2
-
\|\hat c-c^*\|^2.
\]

令该组引入的中心位移为：

\[
\delta_G=\hat c-\hat c_{-G}.
\]

则：

\[
\boxed{
C_{G,\mathrm{sq}}^{GT}
=
2\delta_G^T(c^*-\hat c_{-G})-\|\delta_G\|^2
}.
\]

这个恒等式直接对应用户要求：第一项衡量该弧的拉动是否朝向真值，第二项惩罚过度拉动。朝对方向但严重越过真值，也可能成为负贡献。

跨尺度汇总可另报：

\[
C_{G,\mathrm{rel}}^{GT}
=
\frac{
\|\hat c_{-G}-c^*\|^2-\|\hat c-c^*\|^2
}{
\|\hat c_{-G}-c^*\|^2+\|\hat c-c^*\|^2+\tau_C^2
}.
\]

\(\tau_C\) 来自 GT 标注误差、重复标注传播误差和求解数值误差，不是任意极小常数。

### 5.4 符号死区与贡献置信区间

通过 GT 扰动、曲线移动块扰动和求解 replay 得到 \(C_G^{GT}\) 的区间。标签规则：

```text
lowerCI(C) > +epsilon_C  -> BENEFICIAL_GT
upperCI(C) < -epsilon_C  -> HARMFUL_GT
区间位于死区内          -> NEUTRAL
区间跨越边界             -> UNCERTAIN
```

\(\epsilon_C\) 至少覆盖应用上无意义的微小位移以及 GT/solver 误差。

负贡献仅表示“在当前裁窗、其他证据和冻结估计器的上下文中有害”，不自动证明该标注错误；它也可能是真实偏心生长或少数正确证据对抗多数共同偏差。

### 5.5 删除后不可辨识时，状态优先于毫米数

若删除 \(G\) 导致：

- `POINT → RANGE/RAY/AXIS`：标为 `IDENTIFIABILITY_CRITICAL`；
- 新增稳定模态：标为 `MODE_EXCLUSION`；
- 方向区间显著变宽：标为 `DIRECTION_CRITICAL`；
- 距离区间显著变宽：标为 `RANGE_CRITICAL`。

此时不应用某个任意远场原始点制造巨大 \(C_{\mathrm{mm}}\)。如果研究需要在所有状态上比较概率预测，可以把 profile/bootstrap 结果映射到紧致单位圆盘：

\[
T(\tilde c)=\frac{\tilde c}{1+\|\tilde c\|},
\]

其中 \(\tilde c=(c-o)/D_{\mathrm{FOV}}\)。再对其预测分布使用 proper energy score。该分布贡献只作为审计指标，有限 `POINT` 的主解释仍使用 \(C_{\mathrm{mm}}^{GT}\)。

### 5.6 信息、杠杆、冲突与正负号必须分开

对子弧 \(G\) 可报告方向多样性信息：

\[
A_G=\int_G t(s)t(s)^T\,d\mu(s),
\]

全证据信息矩阵直接定义为：

\[
A=\int_{\mathcal E}t(s)t(s)^T\,d\mu(s).
\]

只有在**一个固定尺度、一个固定相位的非重叠完整分割**中，才有 \(A=\sum_G A_G\)。跨层级、跨尺度和 shifted blocks 仅用于稳定性审计，不能同时相加，否则会重复计数并破坏 \(A-A_G\succeq0\) 的语义。\(\varepsilon\) 按归一化坐标尺度和 \(\operatorname{tr}(A)\) 预先固定，而不是为每个删除结果重新调节。

\[
I_G^{\mathrm{dir}}
=
\log\det(A+\varepsilon I)
-
\log\det(A-A_G+\varepsilon I)
\ge0.
\]

该量只能叫 `directional_info_gain`，不能冒充最终 VarPro 的全部信息。最终距离信息用删除前后的 profile 宽度、局部 Schur/Hessian 或状态变化表达。

定义：

\[
C_G^{\mathrm{shift}}=\|\hat c-\hat c_{-G}\|,
\]

\[
K_G^{\mathrm{conflict}}
=
J_{-G}(\hat c)-J_{-G}(\hat c_{-G})
\ge0.
\]

- `shift` 大只表示高杠杆，不说明正负；
- `conflict` 大表示 \(G\) 迫使其余证据牺牲拟合，也不必然说明 \(G\) 错；
- 信息高、杠杆高且 GT 为负是完全可能的：该段既是关键距离证据，又带有系统偏差。

因此角色允许多标签：

```text
BENEFICIAL_GT / HARMFUL_GT / NEUTRAL / UNCERTAIN
DIRECTION_CRITICAL / RANGE_CRITICAL / MODE_EXCLUSION
HIGH_LEVERAGE / CONFLICTING / REDUNDANT
IMAGE_POOR / ASSOCIATION_RISK / SCALE_UNSTABLE
```

### 5.7 无 GT 时：交叉拟合支持/冲突贡献

没有 \(c^*\) 时，从同一观测内部严格判断“是否更接近生物学髓心”在信息上通常不可能。尤其当所有弧共同偏移时，内部一致性可以很好而中心仍然错误。

比“残差大就是负”“杠杆大就是负”更严谨的运行时定义是**交叉拟合预测贡献**。

对目标组 \(G\)，把其他 parent rings 分成若干 witness folds \(V_f\)。令：

\[
T_f=\mathcal E\setminus(G\cup V_f),
\]

\[
\hat c_f^-=\mathcal M(T_f),
\qquad
\hat c_f^+=\mathcal M(T_f\cup G).
\]

在完全未参与两次拟合的 witness rings 上计算相同的几何预测损失：

\[
\boxed{
C_G^{CF}
=
\operatorname{median}_f
\left[
L(V_f;\hat c_f^-)-L(V_f;\hat c_f^+)
\right]
}.
\]

其中每条 witness ring 在给定中心下重新 profile 自己的半径。

具体地：

\[
L(V;c)
=
\sum_{r\in V}
\min_R
\int_{E_r}
\rho_H\!\left(
\frac{\|x-c\|-R}{\sigma_{x,r}(x)}
\right)d\mu_r(x).
\]

Witness 使用原始冻结的尺度、Huber 参数和证据预算；\(V_f\) 中的整条 parent ring 不得进入两次训练。对 subarc 级条件贡献，其同 ring 的兄弟段可以按预注册规则保留在训练上下文中，但绝不能进入 witness；对 ring 级贡献则整条目标 ring 都从训练与 witness 中排除。

- \(C_G^{CF}>0\)：加入 \(G\) 改善对独立年轮的预测，模型相对支持；
- \(C_G^{CF}<0\)：加入 \(G\) 损害对独立年轮的预测，模型相对冲突；
- 区间跨 0：不确定。

这仍不是因果真值贡献。以下情况必须 `ABSTAIN`：

- 独立 parent rings/证据块不足；
- 两次训练结果自身为 `AXIS/MULTIMODAL/REJECT`；
- witness 几何方向单一；
- 符号对 fold、尺度或切分相位不稳定。

### 5.8 从支持/冲突到部署期正负概率

在有 GT 的开发树上，先冻结定位器并生成 out-of-fold 的 \(C_G^{GT}\)。再用一个小型正则化三分类/有序回归校准器，输入仅无 GT 特征：

- \(C_G^{CF}\)；
- \(K_G^{\mathrm{conflict}}\)；
- LOO shift；
- 方向/距离区间和状态变化；
- `directional_info_gain`；
- ring-order 违反；
- 图像—标注质量；
- 尺度、相位与扰动稳定性。

输出：

```text
P(positive), P(neutral), P(negative)
expected_contribution
prediction_interval
abstain_reason
```

高杠杆和高冲突不预设符号，由独立树数据学习其组合关系。校准器必须按 tree 做 out-of-fold 训练和独立概率校准；同一树派生的 crops 不能跨集合。

### 5.9 第一版禁止贡献反馈

主版本固定：

```text
final coordinate = All-Arc RAC-Pith v2
contribution = explanation / quality-control only
```

以后若研究 Safe-Prune，只允许一次删除，并同时满足：

- 高置信负贡献；
- 有候选无关的图像或标注异常证据；
- 不是方向、距离或排模态关键弧；
- 删除后状态不被虚假升级；
- 独立树上 median、P90/P95、灾难误差和 False-POINT 均不恶化。

否则始终回退 All-Arc。不得建立“预测负贡献 → 删除 → 重新产生更有利贡献标签”的自强化循环。

---

## 6. 完整算法伪代码

```text
Algorithm RAC-Pith-v2
Input:
    crop image I
    labeled parent-ring curves {Gamma_r}
    coordinate transform and optional pixel-to-mm scale
    optional pith GT c_star (train/evaluation only)

Output:
    raw candidate, usable geometry result and state
    uncertainty/profile/modes
    ring/arc/subarc contribution records

A. Validate and normalize
    validate source↔crop transform, ring IDs and physical scale
    normalize coordinates by crop center and D_FOV
    never clip an outside pith or candidate to the crop

B. Recover continuous evidence
    split true gaps into visible fragments
    fit lightly smoothed cubic curves
    set endpoint guards and arc-length quadrature
    assign one frozen evidence budget to each parent ring
    compute candidate-independent image/annotation quality and sigma

C. Build attribution hierarchy
    keep ring -> arc -> subarc lineage
    form primary non-overlapping blocks at registered physical scales
    form shifted blocks only for boundary-stability analysis

D. Convex common-center initialization
    for each parent ring:
        compute x_bar_r and mean_squared_norm_r
        build 2(x-x_bar_r)^T c = ||x||^2-mean_squared_norm_r
    solve parent-balanced pseudo-Huber problem -> c0, H0, weak direction

E. Secondary tangent seed
    estimate stable tangents from the annotation
    solve pseudo-Huber tangent-zero-projection problem -> c_t

F. Robust finite refinement
    from c0 and c_t:
        profile one robust radius R_r(c) per ring
        minimize 2-D geometric VarPro objective
    keep distinct stable minima

G. Mandatory far-direction audit
    evaluate analytic J_infinity(phi) on [0, pi)
    refine its low valleys and compare them with the best finite loss
    no tentative POINT may skip this step

H. Conditional distance profiles and state
    if H0 is ill-conditioned, solutions disagree/are far, or far gap is small:
        profile distance along the weak and all competitive far directions
        include analytic far endpoints and densify once
    if one-dimensional coverage is insufficient:
        run compact 2-D direction/inverse-distance audit
    apply the mutually exclusive state decision tree
    inadequate search -> REJECT, never POINT

I. Structural risk and uncertainty
    check ring order, image anomalies and target-domain registry
    perturb continuous annotation by moving blocks / replicate labels
    use ring bootstrap when enough rings exist
    combine with weak-axis profile and mode structure
    if bootstrap reveals a new long tail or mode:
        rerun H with expanded profiles and update the final state
    invalid data/association -> REJECT
    high target/model risk -> block production use without rewriting geometry state

J. Exact hierarchical contributions
    freeze all estimator settings
    for every registered ring/arc/subarc in audit mode:
        remove its frozen evidence mass
        rerun steps D-I completely
        record state change, support change, shift, conflict and role
        if GT and both estimates pass the POINT gate:
            compute C_GT_mm, C_GT_sq and C_GT_rel
        otherwise:
            prioritize state/identifiability role

K. Runtime signed diagnostics
    where enough independent rings exist:
        compute cross-fitted witness contribution C_CF
        apply independently trained sign calibrator
    otherwise:
        abstain from runtime positive/negative claim

Return all results; do not change the All-Arc coordinate from contribution output.
```

---

## 7. 建议的输出结构

### 7.1 定位结果

```text
PithResult
├── estimate
│   ├── pith_raw_xy
│   ├── pith_usable_xy
│   ├── direction_or_axis
│   ├── finite_range_interval
│   └── state
├── support
│   ├── xy_region_or_ellipse
│   ├── weak_axis_profile
│   ├── direction_interval
│   ├── range_interval
│   └── modes
├── diagnostics
│   ├── condition_ratio
│   ├── search_adequacy
│   ├── ring_order_risk
│   ├── target_domain
│   ├── model_risk
│   └── reason_codes
└── contributions[]
```

### 7.2 每个贡献记录

| 字段 | 含义 |
|---|---|
| `level / ring_id / arc_id / subarc_id` | 证据层级与 lineage |
| `arc_interval_mm` | 在原弧上的物理位置 |
| `contrib_gt_mm/sq/rel` | 有 GT 时的正式反事实贡献 |
| `contrib_gt_label/CI` | 正、负、中性或不确定 |
| `contrib_cf` | 无 GT 交叉拟合支持/冲突分数 |
| `P_positive/P_neutral/P_negative` | 独立校准后的符号概率 |
| `directional_info_gain` | 非负方向多样性信息 |
| `loo_shift_vector/mm` | 该组引入的中心移动 |
| `conflict_cost` | 对其余证据造成的拟合牺牲 |
| `direction/range_change` | 删除后支持区变化 |
| `state_change/new_modes` | 可辨识性与排模态作用 |
| `image_annotation_quality` | 图像与标注质量 |
| `scale_phase_stability` | 子弧粒度稳定性 |
| `roles / abstain_reason` | 多角色标签或不出符号原因 |

贡献热图应同时显示符号、置信度和层级。仅用红绿颜色而不显示 `UNCERTAIN`、尺度和状态变化，会制造过度解释。

### 7.3 产物内容绑定

路径和文件名不足以证明同一运行的 lineage。已物化 crop 的正式 manifest 应逐条保存图像与 GT-free crop annotation 的 SHA-256，evidence 构建时重新计算并拒绝内容漂移；未物化候选不能伪造为已有产物哈希。定位、贡献、不确定性、统计和图形都以各自 index 登记的配置、tree/section/crop lineage、source result index 与内容哈希为权威，目录扫描只用于发现并拒绝未注册旧文件。

贡献统计必须显式接收同一 `crop_manifest` 作为完整分母，并输出 `contribution_evaluation_index.json`；贡献图输出 `contribution_visualization_index.json`。运行时分析用 `runtime_denominator.json` 登记所有运行时/搜索 CSV 的哈希。最终 Markdown 报告旁必须有同名 `.md.index.json`，同时绑定报告自身和所有实际输入 index/denominator 的 SHA-256；报告与 sidecar 任一缺失或哈希不一致时，都不能视为封存结果。

正式报告还必须在生成前通过三条相互独立的审计作用域：主定位/正式贡献、分区敏感性贡献、完整截面目标域。三者都输出 `racpith.combined_audit.v1` 且为 `PASS` 后，`10_build_report.py` 才允许生成封存报告；脚本还要求三份审计中恰有一份含非空 `target_domain_records`，防止用普通 PASS 审计替代完整截面目标域门。报告正文列出三份审计摘要，sidecar 逐份绑定内容哈希。审计与绘图都拒绝已漂移的 crop 输入、未注册或已改变的不确定性 sidecar、minus 结果和 PNG，避免“表图正确但来源已被替换”。

---

## 8. UruDendro 裁窗实验设计

> **统一路径配置**：`configs/racpith_v1.json` 的 `paths.dataset_root` 固定为
> `/root/2026/dataset/UruDendro4/UruDendro4/`，`paths.output_root` 固定为
> `./outputs`（相对于项目根目录）。`scripts/manual/run_all_development.sh` 与
> `scripts/manual/00_prepare_data.sh` 均从该配置解析路径；默认产物分别进入
> `./outputs/prepared`、`./outputs/development` 和 `./outputs/sealed`。
> 总控脚本不接受位置路径参数，标准命令固定为
> `bash scripts/manual/run_all_development.sh`。具体运行说明见
> `RAC_PITH_IMPLEMENTATION_AND_MANUAL_RUN.md`。
> 冻结与 sealed 手动入口也从所给配置解析 prepared/output 路径，不再在命令中重复
> 传入这些目录。

UruDendro4 的部分 LabelMe 文件将 `imageWidth`、`imageHeight`，以及可能的
`imagePath` 保存为空字符串。空白值只表示这部分元数据未提供：尺寸以实际解码图像为准，
section 身份以严格的图像—标注文件 stem join 为准。任何非空尺寸或 `imagePath` 声明仍须
与实际图像和 section ID 一致；所有年轮点也始终按实际图像尺寸检查边界。

### 8.1 数据生成

对每个完整截面：

1. 读取图像、年轮边界和 \(c^*\)；
2. 先按 biological tree 划分 train/validation/test；
3. 只在各 split 内生成局部 crops；
4. 用**年轮边界曲线**与 crop 相交，不能把填充 polygon 与矩形相交后误保留 ROI 边框；
5. 保留 parent ring ID、ring order、visible fragment 和原始弧长参数；
6. 将髓心转换为 crop 坐标，即使为负或远大于图像宽高也原样保存；
7. 保存 tree、section、crop、标定和变换 provenance。

UruDendro 提供完整截面年轮标注和髓心坐标，适合构造这种可控的“完整真值 → 局部反问题”基准。

### 8.2 难度分层

至少按以下量正交分层：

- 髓心在图内、近图外、中距图外、远图外；
- \(d(c^*,\Omega)/D_{\mathrm{FOV}}\)；
- parent-ring 数；
- 总有效弧长和总转角；
- \(H_0\) 的特征值比；
- 单侧弧与多方位弧；
- 图像质量、裂纹、节疤、染色和模糊；
- `TARGET_ALIGNED / ECCENTRIC_GT`；
- crop 物理尺寸与图像分辨率。

### 8.3 基线

几何输入公平基线：

- B0：每 ring 独立圆拟合后平均/中位圆心；
- B1：切向法线束 WLS/Huber；
- B2：凸共同中心初值；
- B3：普通共同中心 VarPro；
- B4：B2 + 鲁棒 VarPro；
- B5：B4 + 全方向远场扫描 + 条件式距离 profile；
- RAC-Pith v2：B5 + 图像质量、状态、不确定性与贡献层。

图像输入基线（APD/LFSA/ACO 类）应单独成组，因为它们不使用人工年轮 GT，不能与使用标注曲线的方法混成一个完全公平的排行榜。

### 8.4 定位指标

- 有限 `POINT` 的误差：px、mm、相对 \(D_{\mathrm{FOV}}\)；
- tree-balanced median、IQR、P90、P95、max 和灾难误差率；
- 图外样本的方向误差与 log-range 误差；
- `POINT/RANGE/RAY/AXIS/MULTIMODAL` 状态正确性；
- `False-POINT`：输出 POINT 但物理误差超过应用容差；
- POINT coverage 与 usable coverage 的风险—覆盖曲线；
- 支持区对 GT 的覆盖率及宽度；
- solver failure、搜索贴边率和多初值分歧率。

### 8.5 贡献指标

GT exact contribution：

- \(C^{GT}\) 的置信区间与稳定性；
- 正/中/负三分类分布；
- ring、arc、subarc 各层级一致性；
- 两到三个尺度、两个相位的符号稳定率；
- 有害段 top-k 捕获率；
- `DIRECTION/RANGE/MODE` 关键弧保护率。

无 GT 预测：

- macro-F1、balanced accuracy、负类 PR-AUC；
- \(P(+/0/-)\) 的 Brier score 与 ECE；
- 预测值与 \(C^{GT}\) 的 Spearman 相关；
- precision@k 和 abstention-risk curve；
- GT-oracle 单删、预测单删与 All-Arc 的差距。

不要要求“按初始贡献排序连续删除多个段后误差必然单调下降”，因为 LOO 具有交互，删除一个段后其余段的贡献会改变。

### 8.6 必做消融

1. 点数加权 vs parent-ring 证据守恒；
2. 切向初值 vs 凸共同中心初值；
3. 代数初值 alone vs `+ VarPro`；
4. L2 vs Huber/pseudo-Huber；
5. 固定外部搜索框 vs 弱轴逆距离 profile；
6. 无/有 ring-order gate；
7. geometry only vs `+ image/annotation quality`；
8. 单点 bootstrap vs 正确的移动块/ring 层级不确定性；
9. in-sample residual heuristic vs 交叉拟合 \(C^{CF}\)；
10. exact delete-refit vs influence 近似；
11. 5/10/20 mm 与双相位子弧；
12. All-Arc vs GT-oracle single delete；
13. aligned-domain vs full-domain；
14. 无/有低容量偏差校正（仅研究版）。

### 8.7 压力测试

合成与真实扰动必须覆盖：

- 直线极限和极远髓心；
- 只有 1–2 条 ring；
- 极短弧和不同重采样密度；
- 同 ring 多 fragment；
- 错误 ring association/order；
- 连续低频偏心变形；
- 多条 ring 的共同切向或法向偏差；
- 局部标注平移、断裂和伪弧；
- 节疤、裂纹、锯痕和模糊；
- 分辨率、裁窗尺度和坐标变换变化。

人工扰动段不必然产生负 GT 贡献：它可能偶然抵消原有偏差。因此应分别评价“异常检测是否正确”和“反事实贡献符号是否正确”。

---

## 9. 两份附件的客观融合与取舍

| 内容 | 最终处理 | 理由 |
|---|---|---|
| 共同中心 VarPro | 保留为有限中心精化 | 低自由度、解释清楚、实现成熟 |
| 法线/切向交会 | 调整为第二初值与诊断 | 公式正确但切向微分会放大标注噪声 |
| 半径无关的共同中心线性化约束 | 新增为主初值 | 凸、快速、无需曲率，直接利用 ring ID |
| 连续弧与 parent budget | 完整吸收 | 防止点密度、插值和人为切段制造伪证据 |
| 有限/无穷远统一 | 吸收思想，简化实现 | 用全方向远场极限扫描 + 条件式距离剖面处理退化，无需默认全 RP² 网格 |
| 六源冷启动 | 缩减为三类候选 | 多数来源重复，首版阈值和成本过高 |
| 双图表全局 mesh 与 seam replay | 降为异常样本审计 fallback | 不是每个 crop 都需要，且不能构成数学全局证书 |
| Search Certificate | 改称搜索充分性检查 | 预算加倍稳定不等于严格全局最优证明 |
| POINT/RANGE/AXIS/MULTIMODAL | 保留并加入 RAY | 准确表达方向—距离分离和定向信息 |
| 生物学髓心 vs 汇聚中心 | 完整吸收 | 防止低内部 loss 被误当作真实髓心正确 |
| exact delete-refit | 完整吸收 | 是正负贡献最可靠的操作定义 |
| 方向/距离/排模态多角色 | 完整吸收 | 单一正负标签无法表达关键弧的双重作用 |
| 高残差/高杠杆判负 | 舍弃 | 二者都没有真值方向；关键有益弧也可能高杠杆 |
| 贡献驱动循环降权 | 从主线删除 | 会改变估计器并造成自强化、标签循环 |
| Safe-Prune | 保留为未来单删实验，默认 OFF | 贡献解释不等于自动删弧；需独立树证明尾部安全 |
| Shapley 全量归因 | 舍弃首版 | 成本高、相邻子弧相关，LOO 已直接对应任务 |
| 曲率/密切圆心 | 仅作条件式基线或 seed | 二阶导数对短弧噪声过敏 |
| 椭圆、Eikonal、神经隐式场 | 降为后续研究 | 自由度远高于单局部裁窗可支持的信息 |
| 图像几乎不进入 ArcPith | 补充轻量质量证据 | 满足本任务的图像输入，同时避免重复造大模型 |
| ring order 未用于定位 | 新增轻量顺序门 | 可低成本消除镜像方向和发现错误 association |

ArcPith 最有价值的部分是“远场不伪有限、连续证据守恒、状态化输出和 exact 反事实贡献”；它更适合作为审计规范，而不是整套作为首版在线算法。原综合稿的 VarPro 主线更接近工程实现，但必须补上无限远退化、贡献层级、冻结估计器和无 GT 正负号不可识别这四个缺口。

---

## 10. 实施顺序与成熟度

### V1：必须先完成

- crop/坐标/lineage 验证；
- 连续弧和 parent-ring 证据守恒；
- 凸共同中心初值；
- 二维 pseudo-Huber VarPro；
- 必做远场方向扫描、条件式弱轴/竞争方向 profile 与状态输出；
- 必要时触发二维紧致方向—逆距离审计；
- 基础图像—标注质量诊断；
- 主定位的 profile 与最小必要测量不确定性；
- ring LOO，以及一个预注册非重叠主尺度的 subarc GT exact delete-refit；
- 按树分组的定位与贡献评测。

V1 已能回答：局部年轮弧能否定位树髓、何时只能给方向，以及在有可靠 GT 且完整/删除结果均为 `POINT` 时，哪个主尺度子弧把结果拉近或推离真值。

### V2：有验证收益再加入

- 图像—标注质量到测量尺度的校准；
- ring-order 定向与 association 风险；
- visible-arc 层级、多个子弧尺度、双相位和贡献区间；
- 完整移动块/ring bootstrap 与尺度稳定性；
- 无 GT 交叉拟合贡献；
- 小型三分类/有序贡献校准器；
- 更高预算的二维紧致审计 replay。

### V3：研究扩展

- 一个受几何支持区约束的低容量偏差修正器；
- 经独立树冻结的单次 Safe-Prune；
- 多窗口主动选取；
- 木射线高置信方向融合；
- 跨截面髓线联合估计。

暂不建议进入主线：全量 Shapley、默认椭圆族、逐轮中心漂移、Eikonal 反演、SIREN/隐式场、端到端大网络和递归删弧。

### 运行模式与计算预算

`CORE` 每个 crop 固定运行：连续曲线统计、一个二维凸初值、少量二维 VarPro 启动和一维远场角度扫描。若共有 \(N_q\) 个求积节点、\(M_\phi\) 个远场方向，主要代价近似为 \(O(N_q)+O(M_\phi N_q)\)，二维线性系统本身可忽略。

`PROFILE` 只在退化或竞争方向出现时运行若干一维距离剖面；`AUDIT` 才运行二维紧致网格、多尺度贡献和高预算 replay。若登记 \(K\) 个删除组，exact contribution 的代价约为 \(K\) 次 Core/Profile 重求解，因此应离线并行。V1 不要求在每个删除重求解内部再做完整贡献 bootstrap，避免指数式审计成本。

当前生产实现只允许用户在 conda `py311` / Python 3.11 环境中手动运行，几何主线固定为 NumPy/SciPy float64 CPU，并以 crop 级 worker 并行。GPU 当前禁用，不得按硬件可用性自动切换。只有未来确认批量特征/profile/bootstrap 为瓶颈，并在同一生产环境完成 CPU/cuda:0 对中心、目标/profile、状态、支持区和贡献标签的等价 replay 后，才允许增加显式 opt-in 的同算法 GPU0 加速；GPU 不得改变目标函数、默认精度或状态规则。

---

## 11. 可主张的创新点

1. **凸共同中心初始化**：由理想圆的精确半径无关恒等式构造二维凸鲁棒 surrogate，代替逐弧圆拟合和大规模全局冷启动。
2. **有限—远场轻量推理**：先扫描全方向远场极限，再只对弱方向和竞争方向做距离剖面，允许结果自然落到无穷远端点。
3. **证据守恒的层级归因**：ring、visible arc、subarc 使用冻结弧长预算，贡献不受插值密度与人为切段数量操纵。
4. **任务一致的反事实正负贡献**：用完整删除重求解直接测量该弧引起的位移是否朝向真实髓心，并通过死区和区间避免伪符号。
5. **无 GT 的外部化贡献**：用未参与拟合的 witness rings 判断目标弧是否改善跨年轮预测，再校准为可拒答的正/中/负概率。
6. **贡献与修正解耦**：贡献先作为解释和质检，不让预测结果反向改变产生标签的估计器，从设计上避免循环。

IRLS、Huber、VarPro、bootstrap 本身是成熟工具，不应单独包装成创新；真正的创新在于上述问题定义、组合方式、退化处理和贡献语义。

---

## 12. 最终建议

首版应冻结为：

```text
RAC-Pith-v2-Core
= 连续弧证据守恒
+ 凸共同中心初值
+ 二维鲁棒 VarPro
+ 全方向远场极限扫描
+ 条件式弱轴/竞争方向逆距离 profile
+ 状态化不确定性
+ 层级 exact delete-refit contribution
```

它保留了两个附件真正有价值的部分，同时避免了以下常见误区：

- 把复杂模型当作精度来源；
- 把搜索上界当成髓心；
- 把高残差或高杠杆当成负贡献；
- 把贡献预测反向用作负权重；
- 把子弧 LOO 当作可加账本；
- 把几何汇聚中心无条件等同于生物学髓心。

这条路线的计算主体只有一个凸二维初值、一个二维鲁棒精化、一个廉价远场角度扫描，以及按需触发的少量一维距离剖面；二维紧致网格只在覆盖不足时兜底。它足够简洁，可在小样本下实现和验证；同时在“视野外髓心、距离不可辨识和带符号子弧贡献”三个真正困难的问题上，比单纯圆拟合或复杂黑箱模型更完整。

---

## 参考资料

1. Marichal, H., Passarella, D., Lucas, C., Profumo, L., et al. (2025). *UruDendro, a public dataset of 64 cross-section images and manual annual ring delineations of Pinus taeda L.* Annals of Forest Science, 82, 25. [DOI: 10.1186/s13595-025-01296-5](https://doi.org/10.1186/s13595-025-01296-5).
2. Schraml, R., & Uhl, A. (2013). *Pith Estimation on Rough Log End Images using Local Fourier Spectrum Analysis.* [ACTA Press abstract and DOI](https://review.actapress.com/Abstract.aspx?paperId=454987).
3. Decelle, R., Ngo, P., Debled-Rennesson, I., Mothe, F., & Longuetaud, F. (2022). *Ant Colony Optimization for Estimating Pith Position on Images of Tree Log Ends.* Image Processing On Line, 12, 558–581. [Article and implementation](https://www.ipol.im/pub/art/2022/338/).
4. Marichal, H., Passarella, D., & Randall, G. (ICPR 2024; proceedings 2025). *Automatic Wood Pith Detector: Local Orientation Estimation and Robust Accumulation.* [arXiv:2404.01952](https://arxiv.org/abs/2404.01952).
5. Duncan, R. P. (1989). Increment-core pith estimation using incomplete ring geometry; retained here only as a historical curvature/arc baseline.
6. Groover, A. T. (2016). *Gravitropisms and reaction woods of forest trees—evolution, functions and mechanisms.* New Phytologist, 211(3), 790–802. [DOI: 10.1111/nph.13968](https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.13968).

---

## 文档来源说明

本稿把两份附件视为待审查的方法材料，而非执行指令。原附件未被改写：

- `pith_localization_urudendro_synthesis.md`
- `ArcPith_GTArc_SplitRole_Frozen_Final_Plan_v4.md`

最终取舍以用户目标、数学一致性、局部短弧的可辨识性、贡献定义的非循环性及首版工程成熟度为准。
