# 基于木材横切面局部图像与年轮线标注的树髓定位及年轮弧段贡献分析> 方法学报告 · 数学建模与算法设计
> 关键词：树髓定位、年轮、生长场等值线、同心圆拟合、反问题、贡献分析、Shapley 值、鲁棒估计

---

## 0. 摘要（TL;DR）

给定木材横切面的**局部**图像与已标注的年轮弧段，树髓（髓心）通常位于视野之外，定位本质上是一个**由等值线弧段反演"源点"的病态反问题**。本文：

1. 将树木生长的生物学事实（形成层平行扩张、早晚材、木射线径向分布、偏心生长）抽象为三个递进的数学模型：**同心圆模型 → 星形极坐标曲线族模型 → 一般生长势场等值线模型**；
2. 给出以"法线投票（闭式线性解）→ 可分离非线性最小二乘精化 → 鲁棒重加权 → 贝叶斯/协方差不确定性输出"为主干的求解管线；
3. 推导核心可辨识性结论：单窗口弧段对髓心的定位误差沿"指向髓心方向"按 **1/α²** 放大（α 为弧段对髓心的半张角），并给出互补性补救（多角度窗口、木射线方位观测、轮宽-曲率外推）；
4. 把"每个（子）弧段的贡献（可正可负）"严格定义为**信息贡献（Fisher信息/log-det，闭式秩更新）**与**偏差贡献（影响函数/留一法）**的带符号组合，并给出 Shapley 值汇总方案；指出一个关键的统计学事实：**在模型设定正确时增加数据不会降低精度，负贡献本身就是模型失配（节疤、伪年轮、标注偏差、应压木）的诊断信号**；
5. 以发散清单给出 15 种候选方法，并设计仿真+真值裁窗的验证方案。

---

## 1. 问题陈述

**输入**
- 木材横切面**局部**数字图像 $I$（视野通常不含髓心，例如板材的一个端面区域、钻芯样条）；
- 人工/半自动标注的 $m$ 条年轮线 $\{\Gamma_i\}_{i=1}^{m}$（折线或样条，含早材/晚材边界者更佳）；
- （可选）木射线、裂纹等辐射状纹理；图像标定参数。

**输出**
1. 髓心估计 $\hat c \in \mathbb{R}^2$ 及不确定性（协方差 $\Sigma_c$、95% 置信椭圆）；
2. 每个年轮弧段——同一年轮弧允许进一步细分为**子弧段** $\{s_k\}_{k=1}^{K}$——对定位的**带符号贡献** $\varphi_k \in \mathbb{R}$（正：提高精度/稳健性；负：引入偏差/不一致）；
3. 诊断信息：异常子弧候选（节疤、伪年轮、应压木区）、可辨识性等级。

**难点**
- 观测窗口小 ⇒ 弧段短、近似直线，圆心远在视野外 ⇒反问题**病态**；
- 生物性异常：偏心生长、应压木/应拉木、节子、伪年轮（年内间断）、缺轮、干燥收缩畸变；
- 标注噪声与系统性偏差。

---

## 2. 相关工作与文献调研

> **说明**：本节基于已有知识整理，无法实时检索；标注 ※ 者请务必按检索式核实卷期信息。
> 推荐检索库：Google Scholar / Web of Science；期刊：*Dendrochronologia*、*IAWA Journal*、*Holzforschung*、*Computers and Electronics in Agriculture*、*Forest Products Journal*、*IEEE T-IP/T-PAMI*。
> 检索式示例：`"pith detection" wood cross-section`、`"pith localization" CT log`、`"tree ring" center estimation curvature`、`"increment core" missing rings pith geometric`、`concentric circle fitting arcs`、`tree ring detection deep learning`。

**(a) 树轮图像分析（软件与算法）**
- 商业/半自动系统：WinDENDRO、CooRecorder/CDendro、LIGNOVISION——交互式年轮宽度测量，圆心多靠人工或模板。
- 开源：`measuRing`（R 包，Lara, Bravo & Sierra, 2015, *Dendrochronologia*）；`ROXAS`（von Arx & Carrer, 2014，针叶材管胞解剖，证明"图像→年轮解剖结构"自动化可行）。
- 深度学习树轮检测：基于 U-Net/Mask R-CNN 的年轮边界检测与实例分割近年已有公开工作（如 Thünen 研究所 Gillert 等关于针叶材扫描图的年轮自动识别，※2023 年前后）。这些方法产出**年轮线**——正是本报告的输入上游。

**(b) 树木年代学中的"缺髓心"实践**
- 钻芯未达髓心时，行业惯例用**同心圆弧模板（pith locator）**贴合最内可见年轮，估计到髓心的距离与缺失轮数（几何曲率法）；Duncan(1989)※、Norton 等※ 讨论过此类年龄估计误差。这为本报告"同心圆模型作为基线"提供了学科共识依据。

**(c) 工业 CT 髓心检测**
- 锯材优化中需在原木 CT 序列中检测髓心/髓线并沿长度追踪，常见做法是霍夫圆/圆拟合与三维追踪（如 Longuetaud、Boukadida、Mothe等※，以及瑞典/芬兰锯材 CT 团队※）。与本任务区别：CT 可看整截面（非局部），但算法思想（圆拟合、鲁棒投票）可迁移。

**(d) 几何拟合与鲁棒估计（数学工具）**
- 圆拟合：Kåsa(1976)、Pratt(1987)、Taubin(1991)；统计效率分析 Chernov & Lesort(2005)、Al-Sharadqah & Chernov(2009)；椭圆拟合 Fitzgibbon et al.(1999)。
- 霍夫变换 Duda & Hart(1972)；RANSAC Fischler & Bolles(1981)；M-估计 Huber(1981)。
- 微分几何（平行曲线、渐屈线）：do Carmo(1976)；中轴/草地火烧模型 Blum(1967)；水平集与快速行进法 Osher & Sethian(1988)、Sethian(1999)。

**(e) 生长建模（生物学先验）**
- 树轮-气候过程模型 VS模型（Vaganov–Shashkin；Anchukaitis et al. 2006※）；Fritts(1976)《Tree Rings and Climate》；应压木巨著 Timell(1986)。

**(f) 贡献/影响分析**
- Cook 距离(1977)；影响函数 Hampel(1974)；Shapley 值(1953) 与其多项式/采样计算 Castro et al.(2009,2017)；机器学习归因 SHAP（Lundberg & Lee, 2017）；最优实验设计（D-最优，Fedorov/Pukelsheim）；GNSS 中 GDOP 概念（几何精度因子，与本问题"弧段几何互补性"同构）。

**研究空白（本报告切入点）**：面向"局部窗口 + 已标注弧段 + 髓心在视野外"的树髓定位，缺乏①病态性定量分析、②多线索（年轮几何+木射线+轮宽场）融合、③弧段级带符号贡献度量的系统化工作。

---

## 3. 物理—生物—数学基础

### 3.1 生物学事实（建模依据）

1. **形成层平行扩张**：维管形成层沿既有表面近似平行地向外增生 ⇒ 年轮边界近似构成一族**平行曲线**（offset curves）。
2. **早晚材**：早材胞腔大、壁薄、色浅；晚材胞腔小、壁厚、色深；一年一轮。晚材率、过渡锐度与当年生长速率相关 ⇒ 提供**生长速率场**的纹理证据。
3. **木射线**：射线原始细胞沿**径向**排列，横切面上呈细辐射线（栎木等宽射线材尤其明显）⇒ 其延长线近似过髓心——**强方向先验**。
4. **偏心生长**：坡地/风/树冠偏冠及应压木（针叶）与应拉木（阔叶）使轮宽沿方位角系统性变化 ⇒ 同心圆只是零阶模型，需要偏心扩展。
5. **异常**：节子（局部纹理绕流）、伪年轮（年内假轮）、缺轮（局部不连续）、髓心开裂、干燥收缩（弦向收缩率 ≈ 2× 径向 ⇒ 圆盘系统性变为椭圆）。

### 3.2 几何学事实（算法之源）

- **同心圆**：任意点处法线过圆心；曲率 $\kappa = 1/r$ 为常数。
- **星形曲线族**：以髓心为极点，每条年轮可写为 $r_i(\theta)$，且单调 $r_{i+1}(\theta) > r_i(\theta)$（每轮必在前轮之外）。
- **平行曲线与渐屈线**：$\Gamma_{i+1} \approx \Gamma_i + \delta_i(\cdot)\,\mathbf n_i$；法线族的包络为渐屈线；髓心是法线族的"汇聚核"（同心圆时退化为一点）。
- **草地火烧类比（Blum）**：把每轮看作火前线以"每年一格"向内传播，**淬灭点即髓心**——髓心 = 轮族内偏移的割迹/中轴的终点。
- **正交轨线**：同心圆族的正交轨线是过圆心的直线——与木射线方向场**同构**，给出几何与生物两种观测的天然融合点。

### 3.3 生长场观点（统一框架）

定义**沉积时间场** $T(\mathbf x)$：细胞在 $\mathbf x$ 处被形成层沉积的时间。则- 年轮线 = $T$ 的整数等值线（等时线）；
- $|\nabla T| = 1/v(\mathbf x)$，$v$ 为径向生长速率（eikonal 型方程）；
- **髓心 = $T$ 的唯一全局极小点（梯度场的源/奇点）**。

于是本问题与两个经典反问题同构：
1. **地震震源定位**（波前=年轮等时线，震源=髓心，速度场=生长速率场，Geiger 法可借鉴）；
2. **势论反问题**（圆域格林函数 $\log\|\mathbf x - c\|$ 的等值线即同心圆，估计极点 $c$）。

### 3.4 可用信息线索清单

| # | 线索 | 类型 | 约束强度 |
|---|------|------|----------|
| 1 | 弧段法向（切向） | 几何 | 方向：强（局部）；距离：无 |
| 2 | 弧段曲率 $\kappa$ | 几何 | 距离：$r=1/\kappa$，但短弧病态 |
| 3 | 多轮间距（轮宽）序列 | 几何+生物 | 沿法向的绝对测距基准 |
| 4 | 曲率随轮次漂移 $\kappa_i$ | 几何 | 外推 $\kappa\to\infty$ 得髓心距离 |
| 5 | 轮宽/晚材率沿弧的梯度 | 生物物理 | 偏心方向（软先验） |
| 6 | 木射线方向场 | 生物 | 指向髓心的方位线（强） |
| 7 | 径向裂纹（心裂） | 生物/力学 | 弱方位线 |
| 8 | 干燥收缩各向异性 | 物理 | 仿射畸变校正（否则系统偏差） |

---

## 4. 数学抽象与建模

### 4.1 记号

- 子弧段 $s_k$ 上重采样点 $\{q_j\}$，单位法向 $n_j$（指向髓心一侧），曲率 $\kappa_j$（稳健微分几何估计，见 §6.1）；
- 待估髓心 $c\in\mathbb R^2$；第 $i$ 轮（局部）半径参数 $r_i$。

### 4.2 模型 M0：同心圆模型（基线）

$$\min_{c,\{r_i\}}\ \sum_{i=1}^{m}\sum_{p\in\Gamma_i}\big(\|p-c\|-r_i\big)^2 .$$

**可分离结构（VarPro）**：给定 $c$，最优 $r_i$ 为均值 $\bar r_i(c)=\frac1{n_i}\sum_{p}\|p-c\|$，代入得仅 2 维的目标$$F(c)=\sum_i\sum_p\big(\|p-c\|-\bar r_i(c)\big)^2 .$$

用 Levenberg–Marquardt 求解，初值由 §5.1 给出。

### 4.3 模型 M1：法线交汇线性模型（闭式解）

圆上任一点法线过圆心 $\Rightarrow n_j^\top(c-q_j)=0$。加权最小二乘：

$$\hat c=\Big(\sum_j w_j\, n_j n_j^\top\Big)^{-1}\Big(\sum_j w_j\, n_j n_j^\top q_j\Big),\qquad
\Sigma_{\hat c}\approx \sigma_{\rm eff}^2\Big(\sum_j w_j n_j n_j^\top\Big)^{-1}.$$

秩-1 结构使**条件数完全由法向方向的张开程度决定**——这是后文贡献分析的几何核心（类比 GNSS 的 GDOP）。

### 4.4 模型 M2：星形极坐标曲线族（偏心扩展）

$$r_i(\theta)=\rho_i\Big(1+e_1\cos(\theta-\varphi)+\sum_{\ell\ge2}e_{i,\ell}\cos(\ell\theta-\varphi_{i,\ell})\Big),\quad \rho_{i+1}>\rho_i .$$

- 一阶谐波项 = 圆心随半径线性漂移（偏心生长锥）：$c_i=c_0+e_1\rho_i(\cos\varphi,\sin\varphi)$，**髓心 $=c_0$**；
- 层级贝叶斯：$c_i\sim\mathcal N(c_0,\lambda^2\rho_i^2 I)$，向同心模型收缩，BIC/交叉验证选阶。

### 4.5 模型 M3：一般生长势场（最一般）

参数化 $T(\mathbf x;\Theta)$（如 $T=\log\|\mathbf x-c\|_G$、或薄板样条场），令标注曲线 $\Gamma_i$ 对齐 $T$ 的等值线 $\{T=t_i\}$，反演 $\Theta$ 并取 $T$ 的极小点为髓心。可与 eikonal 正则 $\|\nabla T\|=1/v$、平滑正则联合。计算量大，作为精修/研究选项。

### 4.6 可辨识性与病态分析（关键理论结果）

**单弧 1/α² 定律**（推导见附录 A）：设弧段对髓心的半张角为 $\alpha$、点噪声 $\sigma$、点数 $n$，半径未知时$$\operatorname{Std}(\hat c_{\text{径向}})\ \approx\ \frac{\sqrt{45}\,\sigma}{\sqrt n\,\alpha^{2}},\qquad
\operatorname{Std}(\hat c_{\text{切向}})\ \approx\ \frac{\sqrt{3}\,\sigma}{\sqrt n\,\alpha}.$$

**数值例**：$r=120$ mm、张角 $10^\circ$（$\alpha=0.0873$）、$n=200$、$\sigma=0.03$ mm ⇒
径向（指向髓心方向）标准差 $\approx 1.9$ mm，切向 $\approx 0.04$ mm——**各向异性比约 45 倍**。

**结论与对策**

| 病态来源 | 对策 |
|---|---|
| 单窗口径向弱可辨识（1/α²） | ①对向/多角度窗口（法向互补，误差立即降到 1/α 量级）；②木射线方位观测（方向强约束）；③轮宽-曲率外推（把半径从"未知参数"变为"可估参数"，见 §5.3） |
| 同窗口多轮仅降 $\sigma$为 $1/\sqrt m$ | 接受有限增益；主要靠几何互补 |
| 系统性畸变（干燥收缩、镜头） | 仿射/椭圆模型校正 |

**输出要求**：方法必须输出**各向异性协方差**；当长轴不确定度超阈值时，应诚实报告"方向可信、距离弱约束"的降级结论。

---

## 5. 求解算法体系

### 5.1 初值：法线投票 /木射线融合

- 由 M1 闭式解得 $c_0$；近秩亏时给出狭长协方差并提示数据不足；
- 若有木射线方位线 $\{\ell_j\}$，将其作为附加"过心线"加入同一投票框架（贝叶斯上相当于独立证据，与年轮几何互补：方位 vs 距离）。

### 5.2 精化：可分离 NLLS + 鲁棒损失

$$\hat c=\arg\min_c\sum_{i,p}\rho_H\!\big(\|p-c\|-\bar r_i(c)\big),$$

Huber 损失 $\rho_H$；IRLS 实现；残差按方位角做傅里叶检验，一阶谐波显著则升级 M2 偏心模型（BIC 选择）。

### 5.3 曲率-轮宽外推（把"数年轮"变成线性回归）

同心模型下 $1/\kappa_i$ = 第 $i$ 轮到髓心的（局部法向）距离，且相邻轮距离差=轮宽 $w_i$。记内起累计轮宽 $W_i=\sum_{\ell\le i}w_\ell$，则

$$\frac{1}{\kappa_i}=\rho_0+W_i+\varepsilon_i,$$

对每束局部弧做线性回归得 $\rho_0$（最内轮到髓心的法向距离），从而给出**绝对测距** $\hat c_{\text{法向}}=q_{\text{内}}+\rho_0\,n_{\text{内}}$。多轮、多角度联合即可在单窗口内缓解1/α² 病态。

### 5.4 密切圆聚类（曲率投票）

每个采样点给候选圆心 $o_j=q_j+\kappa_j^{-1}n_j$；全体 $\{o_j\}$ 做稳健聚类（mean-shift / 霍夫累加），簇心即髓心候选，簇协方差即不确定度初值。

### 5.5 鲁棒包装

- 弧段级 RANSAC：随机子集拟合→一致性集最大化；
- IRLS（Huber/Tukey）；
- 异常弧诊断（残差按轮/按方位分组），与 §6 的负贡献机制联动。

### 5.6 不确定性量化

线性化协方差 $\Sigma_c=\hat\sigma^2(J^\top WJ)^{-1}$（模型失配时用 sandwich 估计）；或残差 bootstrap / 贝叶斯后验（M2 的层级先验下 MCMC/Laplace）。输出 95% 椭圆 $(c-\hat c)^\top\Sigma_c^{-1}(c-\hat c)\le\chi^2_{2,0.95}$。

---

## 6. 年轮（子）弧段的贡献分析（核心）

### 6.1 子弧段划分

- **玩家定义**：将每条年轮弧 $\Gamma_i$ 细分为子弧 $\{s_k\}$：等弧长（如 5–10 mm）或等圆心角增量（如 2–5°），或按曲率拐点自然切分；可形成"整轮→半轮→子弧"的**贡献层级树**。
- 预处理：等弧长重采样；局部二次多项式（Savitzky–Golay 型）估计 $q_j,n_j,\kappa_j$；高频残差估计噪声 $\hat\sigma_k$（每个子弧可有不同噪声水平）。

### 6.2 价值泛函与三种贡献定义

设估计器由子集 $S$ 给出 $(\hat c(S),\Sigma(S))$。取价值

$$v(S)=\underbrace{-\log\det \Sigma(S)}_{\text{精度（越大越好）}}\;-\;\lambda\cdot d^2\!\big(\hat c(S),\tilde c\big),$$

其中 $\tilde c$ 为稳健共识中心（RANSAC/中位数），$d^2$ 为马氏距离，$\lambda$ 调节"精度—一致性"权衡。

1. **留一法（LOO）影响**：$\Delta_k=v(D)-v(D\setminus\{s_k\})$——线性模型下可**闭式精确计算**（附录 B 秩-1 更新）；
2. **影响函数（一阶灵敏度）**：M-估计渐近 $\hat c(D)-\hat c(D\setminus\{s_k\})\approx J^{-1}g_k$，$g_k$ 为子弧得分和；
3. **Shapley 值（满足效率性的总账）**：
$$\varphi_k=\sum_{T\subseteq D\setminus\{s_k\}}\frac{|T|!(K-|T|-1)!}{K!}\big[v(T\cup\{s_k\})-v(T)\big],\qquad \sum_k\varphi_k=v(D)-v(\varnothing).$$
$K\le20$ 可精确枚举（配增量 Cholesky），否则置换采样近似（Castro算法）。

### 6.3 信息贡献的闭式分解（线性/线性化模型）

Fisher 信息 $J=\sum_k J_k$，$J_k=\sum_{j\in s_k}w_j\,n_jn_j^\top$。由矩阵行列式引理：

$$\Delta_k^{\text{info}}=\log\det J-\log\det(J-J_k)=-\sum_{\ell}\log\big(1-\lambda_\ell^{(k)}\big)\ \ge 0,$$

$\lambda_\ell^{(k)}$ 为 $J^{-1/2}J_kJ^{-1/2}$ 的特征值。**每个子弧的信息贡献可精确、独立、$O(Kd^3)$ 地算出**（$d{=}2$，极廉价）。

### 6.4 贡献符号的机制分析

> **统计学事实**：模型设定正确且加权一致时，增加数据不会降低估计精度（$J$ 单调增）。
> **因此"负贡献"只可能来自模型失配**——这正是贡献分析的核心诊断价值。

| 情形 | 机制 | 贡献符号与量级 |
|---|---|---|
| 子弧张角大、曲率稳定 | $\Delta^{\text{info}}$ 随张角超线性增长 | **强正** |
| 与现有弧法向近似正交（新方向） | 抬升 $J$ 的最小特征值、改善条件数 | **强正**（互补红利） |
| 与已有弧同向重叠窗口 | 方向冗余，仅 $1/\sqrt n$ 增益 | 弱正 |
| 短弧+高噪声 | 信息≈0，无偏差时≈0 | ≈0 |
| 节疤/绕流/应压木局部变形 | 曲率系统偏差 ⇒ 拖偏 $\hat c$ | **负** |
| 伪年轮误标 | 轮序/轮宽先验错误 | **负** |
| 标注系统偏差（同一轮共享） | 同轮子弧偏差相关 | **负**（且彼此相关） |
| 干燥收缩未校正 | 全局仿射失配 | 整体性负（宜先校正） |

**带符号贡献合成**（实现建议）：

$$\varphi_k \;=\; \Delta_k^{\text{info}}\;-\;\lambda\cdot\max\!\big(0,\ \Delta_k^{\text{bias}}\big),\qquad
\Delta_k^{\text{bias}}=\Big\|\,J^{-1}g_k\,\Big\|_{\tilde\Sigma^{-1}}\cdot\mathrm{sign}\!\big(\text{偏离共识}\big),$$

并给每个子弧打诊断标签：`正常 / 节疤嫌疑 / 伪年轮嫌疑 / 标注复核 / 高杠杆-低噪声`。

### 6.5贡献驱动的迭代重加权```
repeat 由当前权重估计 ĉ, Σ
    计算全部子弧 φ_k    对显著负贡献子弧降权/剔除（保守阈值，防多重检验误删）
until 收敛
```

负贡献子弧应**输出供人工复核**而非静默删除（可能是真生物信号，如应力木）。

### 6.6贡献输出表示

- **贡献热力弧带图**：在横切面图上沿各弧段以颜色映射 $\varphi_k$（红正蓝负），宽度映射 $|\varphi_k|$；
- **贡献表**（示例模式）：

| 子弧ID | 年轮# | 张角(°) | 半径(mm) | $\Delta^{\text{info}}$ | $\Delta^{\text{bias}}$ | $\varphi_k$ | 标签 |
|---|---|---|---|---|---|---|---|
| s03 | 5 | 6.2 | 118.4 | +0.41 | 0.00 | **+0.41** | 正常 |
| s07 | 7 | 5.8 | 96.1 | +0.38 | −0.52 | **−0.14** | 节疤嫌疑 |
| … | | | | | | | |

---

## 7. 总体管线与伪代码

```text
算法 PITHLOC
输入 : 标注弧段 {Γ_i}；(可选)木射线方位线 {ℓ_j}；细分参数；噪声先验 σ0
输出 : 髓心 ĉ、协方差 Σ_c、95% 置信椭圆、子弧贡献 {φ_k} 与诊断标签

1 预处理 : 等弧长重采样；局部多项式估计 (q, n, κ)；逐子弧噪声 σ̂_k
2  细分   : Γ_i → 子弧 {s_k}（等弧长/等圆心角；保留层级）
3  初值   : c0 ← M1 加权法线交点（闭式）；近秩亏时融合 {ℓ_j} 投票
 ρ̂0 ← 曲率-轮宽外推（§5.3）提供绝对距离初值
4  精化   : ĉ ← argmin Σ ρ_H(‖p−c‖ − r̄_i(c)) （LM + VarPro）
5 诊断   : 残差方位傅里叶检验；一次谐波显著 → M2 偏心模型重估（BIC 选阶）
 干燥收缩显著 → 仿射/椭圆校正后重估
6 不确定 : Σ_c ← sandwich 协方差（或 bootstrap / 后验）
7  贡献   : Δ^info ← −Σ log(1−λ_ℓ^{(k)})；Δ^bias ← 影响函数偏移；
 φ_k汇总；需要"总账"时置换采样 Shapley 近似
8  反馈   : 显著负 φ_k → 人工复核/降权 → 回 4
9  输出   : ĉ、置信椭圆、贡献热力图 + 贡献表、可辨识性等级
```

**复杂度**：线性化部分 $O(K)$ 次小矩阵谱分解；NLLS 每轮 $O(n_{\text{点}})$；Shapley 采样 $O(M K)$ 次秩更新（$M$ 为置换数）。整体对常规标注规模（数十弧、数千点）为秒级。

---

## 8. 发散思维：候选方法清单（15 种）

| # | 方法 | 核心思想 | 所需信息 | 优点 | 风险/代价 |
|---|---|---|---|---|---|
| 1 | 法线投票+M1 闭式解 | 圆法线过心 | 弧段切向 | 快、可分析 | 方向单一则秩亏 |
| 2 | 同心圆模板滑窗匹配 | 边缘图与模板族相关 | 仅图像 | 不依赖标注 | 偏心失效、算力 |
| 3 | 密切圆聚类 | 曲率倒数定位圆心 | 曲率 | 直观 | 短弧曲率噪声大 |
| 4 | 曲率-轮宽外推 | $1/\kappa_i=\rho_0+W_i$ 回归 | 多轮+轮宽 | 单窗口内测距 | 轮宽噪声、伪轮敏感 |
| 5 | **木射线方位投票** | 射线延长线过髓心 | 射线纹理 | 与年轮互补（方向强约束） | 散孔材射线细弱 |
| 6 | 径向裂纹/心裂 | 裂纹多径向 | 纹理 | 零成本附加 | 偏差大，仅软先验 |
| 7 | 轮宽/晚材率梯度场 | 偏心方向先验 | 轮宽沿弧变化 | 软约束破对称 | 树种依赖 |
| 8 | eikonal 反演（Geiger 类比） | 等时线+速率场→源点 | 轮序 | 物理优雅 | 联合反演病态 |
| 9 | 势论/格林函数拟合 | 轮=$\log\|x-c\|_G$ 等值线 | 弧段 | 理论美 | 一般域需数值解 |
| 10 | 正交轨线/共形中心 | 轮族+正交轨线=极坐标网 | 轮族 | 融合射线的几何化 | 实现复杂 |
| 11 | 因子图/图模型融合 | 线索为因子、中心为变量 | 全部 | 统一不确定度 | 工程量大 |
| 12 | 深度学习回归 | 图+标注→坐标+不确定度 | 合成数据 | 端到端 | 可解释性、泛化 |
| 13 | SHAP/积分梯度归因 | 学习模型的"贡献" | 配合 #12 | 与 §6 呼应 | 归因≠因果 |
| 14 | 主动学习选窗（D-最优） | 贡献分析→下一采集窗口 | 可交互采集 | 事半功倍 | 需采集回路 |
| 15 | 仿射/椭圆全局拟合 | 校正干燥收缩畸变 | 全截面或多窗 | 消除系统偏差 | 需足够张角 |

**重点推荐组合**：#1（几何主干）× #4（绝对测距）× #5（生物方向）——三者恰好分别约束"切向 / 径向 / 方位"，从机制上封闭 1/α² 病态。

---

## 9. 实验设计与验证

**仿真（可控真值）**
1. 参数化生成器：同心圆 + 偏心漂移 $e_1$ + 谐波扰动 + 各向异性收缩 + 噪声 + 节疤遮挡 + 伪年轮；
2. 随机裁剪观测窗口（张角 5°–180°、窗口数 1–4、弧数 3–30）；
3. 检验：①定位误差是否服从 1/α² 律；②多窗/射线融合的互补增益；③注入"坏弧"后 $\varphi_k$ 是否显著为负、排序是否正确（贡献度量的命中率/AUC）。

**真实数据**
- 完整圆盘图（髓心可见作真值）→ 裁窗实验；树种覆盖针叶/阔叶、环孔/散孔；
- 指标：RMSE、马氏距离覆盖率（95% 椭圆实际覆盖率）、贡献排序与专家一致性（Kendall τ）。

**消融**：木射线 on/off；鲁棒损失 on/off；模型阶 M0/M2；细分粒度对 Shapley 稳定性的影响。

---

## 10. 局限与风险

1. 强偏心/双髓/萌条更新等使"单心"假设失效 ⇒需多心混合模型扩展；
2. 干燥收缩与扫描畸变引入系统偏差，单窗口难以自校正；
3. 木射线在部分树种不可见；伪年轮在干旱/虫害年份频发；
4. 贡献分析基于局部线性化，强非线性区（极短弧）需 bootstrap 复核；
5. Shapley 对强相关玩家（同轮子弧）的归因解释需谨慎，可改用块级 Shapley。

---

## 11. 结论

- 树髓定位 = **生长等值线族的源点反演**；同心圆模型提供低自由度、可分析的基线，偏心/势场模型提供生物真实性升级路径；
- **1/α² 定律**量化了局部观测的本质病态，并指明三条互补出路：多角度弧段、木射线方位、轮宽-曲率绝对测距；
- 弧段贡献可分解为**闭式信息贡献 + 影响函数偏差贡献**，并用 Shapley 汇总成"总账"；**负贡献即模型失配诊断**，驱动"估计—诊断—重加权"闭环；
- 推荐落地路线：M1 闭式初值 → M0/VarPro+Huber 精化 → 诊断升级 M2 → 贡献闭环 → 各向异性不确定度输出。

---

## 12. 参考文献（※ 请核验）

1. Kåsa, I. (1976). A circle fitting procedure and its error analysis. *IEEE TIM*.
2. Pratt, V. (1987). Direct least-squares fitting of algebraic surfaces. *SIGGRAPH*.
3. Taubin, G. (1991). Estimation of planar curves… *IEEE TPAMI*.
4. Chernov, N., Lesort, C. (2005). Least squares fitting of circles. *J. Math. Imaging Vis.*
5. Al-Sharadqah, A., Chernov, N. (2009). Error analysis for circle fitting. *Electron. J. Stat.*
6. Fitzgibbon, A. et al. (1999). Direct least square fitting of ellipses. *IEEE TPAMI*.
7. Duda, R., Hart, P. (1972). Hough transform. *CACM*.
8. Fischler, M., Bolles, R. (1981). RANSAC. *CACM*.
9. Huber, P. (1981). *Robust Statistics*. Wiley.
10. do Carmo, M. (1976). *Differential Geometry of Curves and Surfaces*.
11. Blum, H. (1967). A transformation for extracting new descriptors of shape.
12. Osher, S., Sethian, J. (1988). Fronts propagating with curvature-dependent speed. *JCP*.
13. Sethian, J. (1999). *Level Set Methods and Fast Marching Methods*. CUP.
14. Geiger, L. (1912). Probability method for the determination of earthquake epicenters.
15. Fritts, H. (1976). *Tree Rings and Climate*. Academic Press.
16. Timell, T. (1986). *Compression Wood in Gymnosperms*. Springer.
17. Vaganov, E. et al. / Anchukaitis, K. et al. (2006)※. VS树轮过程模型。
18. von Arx, G., Carrer, M. (2014). ROXAS. *Dendrochronologia*.
19. Lara, W., Bravo, F., Sierra, C. (2015). measuRing. *Dendrochronologia*.
20. Gillert, A. et al. (2023)※. 深度学习树轮自动检测。
21. Longuetaud, F. / Boukadida, H. 等※. 原木 CT 髓心与节子自动检测系列。
22. Duncan, R. (1989)※. 钻芯缺髓心年龄估计误差分析。
23. Shapley, L. (1953). A value for n-person games.
24. Castro, J. et al. (2009, 2017). Polynomial calculation & sampling of Shapley values.
25. Lundberg, S., Lee, S.-I. (2017). SHAP. *NeurIPS*.
26. Cook, R. (1977). Detection of influential observations. *Technometrics*.
27. Hampel, F. (1974). The influence curve and its role in robust estimation. *JASA*.

---

## 附录 A：单弧 1/α² 定律推导

取弧上点 $p(\theta)=c+r(\cos\theta,\sin\theta)$，$\theta\in[-\alpha,\alpha]$，径向噪声 $\sigma$。线性化观测量$\|p-c\|\approx r_0+\delta r - \delta c_x\cos\theta-\delta c_y\sin\theta$。
消去未知半径 $\delta r$（减去均值）后：

- $\mathrm{Var}(\sin\theta)\approx \alpha^2/3 \Rightarrow \mathrm{Var}(\hat c_y)=\dfrac{3\sigma^2}{n\alpha^2}$；
- $\cos\theta\approx1-\theta^2/2$，$\mathrm{Var}(\theta^2/2)=\mathrm{Var}(\theta^2)/4=\dfrac{1}{4}\Big(\dfrac{\alpha^4}{5}-\dfrac{\alpha^4}{9}\Big)=\dfrac{\alpha^4}{45}$
$\Rightarrow \mathrm{Var}(\hat c_x)=\dfrac{45\sigma^2}{n\alpha^4}$。

故 $\mathrm{Std}(\hat c_x)\propto 1/\alpha^2$（径向/角平分线方向），$\mathrm{Std}(\hat c_y)\propto 1/\alpha$（切向）。

## 附录 B：线性模型的精确 LOO（秩更新）

$J=\sum_k J_k$，$b=\sum_k b_k$，$\hat c=J^{-1}b$。写 $J_k=U_kU_k^\top$，由 Woodbury：

$$\hat c_{-k}=(J-J_k)^{-1}(b-b_k),\quad
(J-UU^\top)^{-1}=J^{-1}+J^{-1}U\,(I-U^\top J^{-1}U)^{-1}\,U^\top J^{-1}.$$

配合 Cholesky 增量/降秩更新，所有 $k$ 的精确 $\Delta_k$、$\hat c_{-k}$ 可在 $O(K d^3+d^2 n)$ 内完成（$d{=}2$，极快）。

## 附录 C：Shapley 置换采样近似

```text
for m = 1..M:
    π ← 随机排列(1..K); S ← ∅; v_prev ← v(∅)
    for k in π:
        v_new ← v(S ∪ {s_k})          # 秩-1 更新 O(d²)
        φ_k += (v_new − v_prev)/M
 S ← S ∪ {s_k}; v_prev ← v_new
```

$M\sim10^3$–$10^4$ 即可获得稳定排序；$K\le20$ 时建议精确枚举。