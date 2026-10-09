# ArcPith v5-slim 验证报告

> 验证时间：2026-09-02  
> 文档：ARCPITH_v5-slim_sequential_rapid_validation_protocol-v2.md  
> 代码：OC_ArcPith_RG_72h_algorithm_package_adjusted 2

---

## 执行摘要

本报告对树髓定位算法 ArcPith 的快速验证方案文档与现有代码实现进行系统性对照验证。验证按照文档定义的 6 个阶段顺序进行，每个阶段检查：

1. **文档要求是否清晰**
2. **代码是否实现了相应功能**
3. **实现与要求的差距**
4. **需要补充或修正的内容**

---

## 阶段 0：数值正确性与证据链硬门

### 文档要求（第4章，374-479行）

#### 核心问题（4.2节）
> 同一几何与同一冻结证据，在合法等价变换、重复运行和 chart 切换下，是否给出相同的 loss、解、profile feature、route 与 reason code；任何 audit-only 信息是否都无法影响上游选择？

#### 必须验证的内容

1. **实验 0A：数据合同、lineage 与重放**
   - [ ] 完整 tree→ring→fragment→observation lineage
   - [ ] normalization 与 inverse round-trip
   - [ ] input/config/code/threshold/seed hashes
   - [ ] 连续运行三次结果一致性
   - [ ] 负控制（重复ID、错误frame、缺ring association）

2. **实验 0B：尺度、符号与 frame 不变量**
   - [ ] `λh` 尺度变换不变性
   - [ ] `-h` 符号变换不变性
   - [ ] `t→-t` 切向符号不变性
   - [ ] frame transform 往返一致性
   - [ ] `d_RP2`、loss、gradient、basin、state 一致性

3. **实验 0C：finite→infinity、seam 与 origin**
   - [ ] signed inverse range `κ_s` 扫描连续性
   - [ ] affine/far chart overlap 一致性
   - [ ] `φ≈0/π` seam 邻域连续性
   - [ ] exact AXIS fixture 不误识为极大有限圆

4. **实验 0D：near-point barrier 与搜索充分性**
   - [ ] `d_min` 相对 `r_hard/r_excl` 的 barrier 连续性
   - [ ] `τ_low/τ_high` 穿越的值和梯度连续性
   - [ ] B3 high-budget search 参考 basin
   - [ ] minimal starts 的 basin recovery

### 代码实现状态

#### ✅ 已实现的核心功能

1. **齐次坐标与投影几何** (`geometry.py:13-45`)
   ```python
   def canonical_h(h):  # 实现了 h/-h 规范化
   def h_from_point(p):  # finite → RP²
   def h_from_phi_kappa(phi,kappa):  # (φ,κ) 参数化，κ=0 表示 infinity
   def point_from_h(h,axis_eps=1e-8):  # RP² → finite，正确处理 infinity
   def projective_angle(h1,h2):  # d_RP2 距离
   ```

2. **基础测试** (`test_core.py:20-22`)
   ```python
   test_h_roundtrip()  # ✅ finite point 往返测试
   test_axis()  # ✅ infinity 测试
   ```

3. **审计防火墙测试** (`AUDIT_OUTPUT.json:1-9`)
   - ✅ 测试通过（tests_returncode: 0）
   - ✅ 7个核心测试全部通过

#### ⚠️ 部分实现或需要补充

1. **Lineage 追溯**
   - ⚠️ `io.py` 中有 Sample/Ring 结构，但未见完整的 lineage_hash
   - ⚠️ 缺少 observation_id / fragment_id 的完整追溯链
   - ⚠️ 缺少 input_hash 的明确实现

2. **等价变换测试**
   - ❌ 缺少 `λh` 尺度变换测试
   - ❌ 缺少 `t→-t` 切向符号测试
   - ❌ 缺少 frame 随机变换测试

3. **Chart seam 与连续性**
   - ❌ 缺少 `κ_s` 扫描连续性测试
   - ❌ 缺少 seam 跨越测试

4. **重放与确定性**
   - ✅ `test_candidate_library_deterministic` 验证候选库确定性
   - ⚠️ 但缺少跨运行的完整重放测试

5. **负控制测试**
   - ❌ 缺少错误 lineage 的拒识测试
   - ❌ 缺少 invalid frame 的检测测试

### 验证结论与建议

#### 通过项 ✅
- 投影几何基础实现正确
- Finite/Infinity 连续表达
- 基本往返测试通过

#### 需要补充 🔧

**优先级 P0（阻塞后续阶段）**
1. 补充完整的 lineage 追溯（tree_id → ring_id → fragment_id → observation_id）
2. 实现 input_hash / config_hash / code_hash
3. 添加 3次重复运行一致性测试

**优先级 P1（安全关键）**
4. 添加尺度/符号变换不变性测试套件
5. 添加 seam 连续性测试
6. 添加负控制测试（错误 lineage、frame、association）

**优先级 P2（可延后）**
7. Frame transform 往返测试
8. Chart overlap 一致性测试

---

## 阶段 1：Shared S0 与 Equal-parent 核心价值

### 文档要求（第5章，481-628行）

#### 核心问题（5.2节）
> 在相同观测和相同搜索预算下，equal-parent shared S0 是否比 one-ring pseudocenter 与 pooled micro-arcs 更稳定、更接近 oracle，并对纯复制/插值/fragment split 保持不变？

#### 必须验证的内容（5.1节）

被验证的 S0 目标函数：
```
L_r(h̄) = Σᵢ wᵣᵢ[ρ(eᵣᵢ/σᵣᵢ) + Bᵣᵢ] / Σᵢ wᵣᵢ
L_S0(h̄) = (1/|P_elig|) Σᵣ∈P_elig L_r(h̄)
```

#### 实验要求

1. **实验 1A：核心定位器配对比较**
   - [ ] B0: one-ring pseudocenters + projective medoid
   - [ ] B1: pooled micro-arcs S0，无 equal-parent budget
   - [ ] B2: equal-parent shared S0（核心候选）
   - [ ] B3: dense/offline high-budget search（参考解）
   - [ ] 必须使用相同 crop、observation、split、seed

2. **实验 1B：Equal-parent 预算消融**
   - [ ] 某一 ring 采样密度 ×2/×4
   - [ ] 同一 observations 复制 ×2/×4
   - [ ] fragment 拆分 2/4/8
   - [ ] 验证：纯复制/插值不移动 B2 结果

3. **实验 1C：相关噪声与最小有用域**
   - [ ] iid vs 相关噪声配对
   - [ ] 建立初始 feasibility map

### 代码实现状态

#### ✅ 已实现的核心功能

1. **Equal-parent objective** (`objectives.py:8-16`)
   ```python
   def m0_components(h,prep,cfg,exclude_arc_ids=None,exclude_ring_ids=None):
       for a in prep.arcs:
           if a.arc_id in excl or a.ring_id in exrings: continue
           e=absolute_residual(h,a.points,a.tangents)
           q=a.ds/(np.sum(a.ds)+1e-12)  # ✅ 弧长 quadrature
           total += a.omega*float(np.sum(q*student(...)))  # ✅ equal arc weight
   ```

2. **Robust loss** (`objectives.py:6`)
   ```python
   def student(z,nu=4.0): return 0.5*(nu+1.0)*np.log1p((z**2)/nu)  # ✅ Student-t
   ```

3. **测试** (`test_core.py:27-28`)
   ```python
   test_m0_circle_prefers_gt_to_large_shift()  # ✅ 验证 M0 偏好正确中心
   ```

4. **Ring association** (`test_core.py:31-36`)
   ```python
   test_multiple_fragments_are_retained_without_duplicate_order_or_arc_ids()
   test_missing_parent_ring_is_not_made_adjacent()  # ✅ 保留 ring order
   ```

#### ⚠️ 部分实现

1. **Equal-parent 预算**
   - ✅ 代码中 `a.omega` 实现了 per-arc 权重
   - ⚠️ 但未明确验证 `omega` 的计算是否为 equal-parent
   - ⚠️ 缺少文档说明 omega 如何从 ring_id 聚合

2. **Baseline 对照**
   - ❌ 缺少 B0 (one-ring pseudocenter) 实现
   - ❌ 缺少 B1 (pooled micro-arcs) 实现
   - ✅ 有 M0 (当前实现，疑似对应 B2)
   - ❌ 缺少 B3 (high-budget reference) 明确实现

#### ❌ 缺失的关键实验

1. **配对比较实验**
   - ❌ 无 B0 vs B1 vs B2 vs B3 的配对实验脚本
   - ❌ 无 distance × arc span × ring count × noise 的系统测试

2. **预算消融实验**
   - ❌ 无采样密度变化测试
   - ❌ 无复制不变性测试
   - ❌ 无 fragment split 测试

3. **相关噪声测试**
   - ❌ 无 iid vs 相关噪声的配对测试

### 验证结论与建议

#### 通过项 ✅
- S0 objective 的基本结构正确
- Robust loss 实现
- Ring order 保留机制

#### 关键缺失 ❌

**阻塞性问题（必须解决才能声称通过阶段1）**
1. **明确 equal-parent 实现**：需要验证 `a.omega` 是否真的实现了文档要求的 `1/|P_elig|` 等权重
2. **Baseline 缺失**：必须实现 B0/B1/B3 才能验证 B2 的独立增量价值
3. **配对实验缺失**：必须有同一 crop 上的 B0 vs B1 vs B2 比较

**需要补充的实验**
4. 复制/插值/fragment split 不变性测试
5. Partial arcs 分层测试
6. Easy case 不退化验证

**建议行动**
- 在 `experiments/` 下新建 `stage1_s0_value.py`
- 实现 B0/B1/B3 baseline
- 设计 synthetic fixtures 覆盖 distance/coverage/noise
- 运行配对比较并记录 `d_RP2`、state error

---

## 阶段 2：Preflight 与 Correlation Control 增量价值

### 文档要求（第6章，630-732行）

#### 核心问题（6.2节）
> 在冻结 S0 上游不变的前提下，P2/P3 是否比 P0/P1 更少危险放行、减少无用计算或提供更稳定的路由？

#### Preflight 层级（6.4节）

| ID | 系统 | 被验证内容 |
|---|---|---|
| P0 | ring count only | 最便宜 Preflight |
| P1 | ring count + arc span/coverage | 简单几何 |
| P2 | P1 + turn/angular coverage + F_geom | 几何信息候选 |
| P3 | P2 + correlation class + N_seg-equiv | 相关性修正 |

#### 必须验证的内容

1. **Preflight 不定位髓心**：只判断可行性
2. **两阶段接口**：
   - Preflight-0: candidate-free 的 split `E_fit | buffer | E_audit`
   - Preflight-1: 只用 E_fit 生成 pilot envelope
3. **硬映射**：INVALID/INSUFFICIENT → REJECT
4. **防火墙**：E_audit 不参与 pilot selection

### 代码实现状态

#### ✅ 已实现的功能

1. **Candidate screening** (`experiments/00_screen.py`)
   - ✅ 有 translation/direction/distance 候选筛选
   - ✅ 输出 `m0_screen.json` / `m1rpc_screen.json`

2. **Screen 判断** (基于 EXPERIMENT_GUIDE.md)
   - ✅ `NO_GO_GLOBAL_GEOMETRY` 判断
   - ✅ `BIAS_REPAIR_REQUIRED` 判断

#### ⚠️ 部分对应

1. **Preflight 层级**
   - 当前实现似乎是 screening（阶段3前的筛选）
   - ⚠️ 与文档定义的 Preflight（阶段2）概念不完全对应
   - 当前的 `candidate_screen` 更像是"候选筛选"而非"预飞行可行性判断"

2. **F_geom**
   - ❌ 代码中未见 Fisher information matrix 计算
   - ❌ 未见 weakest eigenvalue / condition number

3. **Correlation control**
   - ❌ 未见 `c_corr` correlation class
   - ❌ 未见 `N_seg-equiv` block-equivalent 支持量

#### ❌ 缺失的关键内容

1. **P0/P1/P2/P3 层级对照**
   - ❌ 无明确的 P0/P1/P2/P3 实现和比较

2. **Pilot envelope**
   - ❌ 未见 E_fit-only pilot envelope 生成

3. **F_geom 预测性验证**
   - ❌ 缺少 perturbation test
   - ❌ 缺少 envelope-worst vs optimum-only 比较

4. **防火墙测试**
   - ❌ 缺少 E_audit 不影响 pilot 的测试

### 验证结论

#### 概念对应问题 ⚠️

**当前代码的 `candidate_screen` 与文档的 Preflight 有概念差异：**

| 维度 | 文档 Preflight | 代码 candidate_screen |
|---|---|---|
| 时机 | 在 S0 搜索前判断可行性 | 在 M0/M1 运行后筛选候选 |
| 输入 | candidate-free 的 lineage/coverage | 需要已计算的候选 loss |
| 输出 | POINT_CANDIDATE / AXIS_ONLY / INSUFFICIENT | NO_GO_GLOBAL_GEOMETRY / BIAS_REPAIR_REQUIRED |
| 作用 | 防止不可辨识样本被强迫成有限点 | 判断 M0/M1 是否可信 |

**这可能意味着：**
1. 文档定义的 Preflight（阶段2）在当前代码中**尚未实现**
2. 或者 Preflight 功能被分散在不同模块中，需要重新整理

**建议行动：**
- 明确 Preflight 的实现位置和逻辑
- 如果未实现，需要补充 P0/P1/P2/P3 层级
- 如果已实现但命名不同，需要建立映射关系

---

## 阶段 3：Profile、Topology 与五态表达价值

### 文档要求（第7章，734-839行）

#### 核心问题（7.2节）
> E_fit-only profile 的 supported set 是否能在合理 grid/δ 小扰动下稳定恢复预期 topology，并在快速网格漏检关键模式前触发 refinement 或保守拒识？

#### 五态定义（文档 1.1节，183-189行）

| 状态 | 含义 |
|---|---|
| POINT | 证据支持唯一且有限的髓心位置 |
| AXIS | 只能稳定确定远场方向，不能确定距离 |
| RANGE_UNCERTAIN | 有限区域或距离范围，不足以压缩成点 |
| MULTIMODAL | 多个稳定且相互分离的候选区域/模式 |
| REJECT | 数据/几何/搜索/审计/偏差/域条件不足 |

#### Fixture library（7.3节）

| Fixture | 预期 provisional topology |
|---|---|
| F-POINT | single compact finite (FINITE_COMPACT) |
| F-AXIS | stable direction + infinity ridge (AXIS) |
| F-RANGE | bounded wide finite component (RANGE_UNCERTAIN) |
| F-MULTI | ≥2 persistent components (MULTIMODAL) |
| F-REJECT | unstable/unclosed (REJECT) |

### 代码实现状态

#### ✅ 已实现的功能

1. **Observability 模块** (`oc_arcpith_rg/observability.py`)
   - ✅ `experiments/03_observability.py` 存在
   - ✅ Profile 计算（基于 φ,κ）

2. **分类测试** (`test_core.py:43-45`)
   ```python
   test_observability_multimodal_requires_multiple_components()
   test_observability_components_wrap_phi()  # ✅ 跨 seam 连通
   ```

3. **Profile grid** (`test_core.py:46-49`)
   ```python
   test_vectorized_m0_profile_matches_scalar_objective()
   # ✅ 验证向量化 profile 与逐点计算一致
   ```

4. **Config 参数** (`config.yaml:52-59`)
   ```yaml
   profile:
     phi_bins: 72
     kappa_bins: 40
     kappa_max: 50.0
     support_delta: 0.75
     point_phi_width_deg: 15.0
     axis_phi_width_deg: 20.0
   ```

#### ⚠️ 部分实现

1. **五态映射**
   - ✅ 有 classify 函数
   - ⚠️ 但输出状态与文档定义的对应关系需确认
   - ⚠️ 缺少 RANGE_UNCERTAIN 的明确实现路径

2. **Refinement 机制**
   - ❌ 未见 two-level (coarse → refinement) 实现
   - ❌ 未见 escalation trigger

3. **Component 分析**
   - ✅ 有 `support_components` 函数
   - ⚠️ 但 minimum_component_separation 的实现需确认

#### ❌ 缺失的关键实验

1. **Fast vs Dense topology 对比**
   - ❌ 无 B3 dense profile 作为参考
   - ❌ 无 two-level fast 与 dense 的 agreement 测试

2. **危险漏检专测**
   - ❌ 无 narrow-deep second basin 测试
   - ❌ 无"B3 存在第二 component，但 fast path 输出 FINITE_COMPACT 且未触发 escalation"的红旗检测

3. **Fixture library**
   - ❌ 无 F-POINT/F-AXIS/F-RANGE/F-MULTI/F-REJECT 的标准 fixtures
   - ❌ 无每类 fixture 的系统测试

4. **状态硬路由测试**
   - ❌ 无 INVALID/INSUFFICIENT → REJECT 的确定性测试
   - ❌ 无 AXIS_ONLY ≠ POINT 的硬约束测试

### 验证结论

#### 通过项 ✅
- Profile 计算框架存在
- 基本的 component 检测
- φ seam 跨越处理

#### 关键缺失 ❌

**阻塞性问题**
1. **Refinement 机制缺失**：文档要求 two-level escalation，代码未见实现
2. **Fixture library 缺失**：无法系统验证五态的稳定性
3. **危险漏检检测缺失**：无法验证 critical second mode 不被漏掉

**需要补充**
4. Fast vs Dense 对照实验
5. 状态硬路由的确定性测试
6. Escalation trigger 实现

**建议行动**
- 实现 two-level profile (coarse → dense refinement)
- 构造 F-POINT/F-AXIS/F-RANGE/F-MULTI/F-REJECT fixtures
- 添加 escalation trigger logic
- 运行危险漏检专测

---

## 阶段 4：Independent Audit 增量反证价值

### 文档要求（第8章，841-960行）

#### 核心问题（8.2节）
> 在 hypothesis、state、representatives 和 sentinels 全部锁定后，E_audit 是否能在 null 下不过度拒绝，在 minimum-relevant/large errors 下产生有方向性的拒绝，并比 fit-only 或 absolute-only 提供独立价值？

#### 核心约束（文档 1.3节）

3. 样本内证据必须遵守 `E_fit | frozen buffer | E_audit`
4. **E_audit 不参与搜索、候选选择、profile、topology、representative 或 threshold 调整**
7. audit underpowered 不等于通过；holdout 通过也不等于真实髓心正确

#### 必须验证的内容

1. **实验 4A：Audit firewall**
   - [ ] 固定 E_fit，生成多个 E_audit variants
   - [ ] E_audit 改变不影响 fit/profile/components/representatives
   - [ ] Audit fail 不触发 re-fit 或 candidate switching

2. **实验 4B：Null 红旗筛查**
   - [ ] H0 fixtures 上的 rejection rate
   - [ ] 验证不超过目标 α 的两倍
   - [ ] Duplicate/resampling 不增加 rejection

3. **实验 4C：Sentinel 响应曲线**
   - [ ] `0/0.5/1/2/large × Δ_min` 错误
   - [ ] A0 (absolute only) vs A1 (relative only) vs A2 (joint)
   - [ ] Large effect 明显区别于 null

4. **实验 4D：Buffer、相关性与 non-point coverage**
   - [ ] Buffer candidate 扫描
   - [ ] Independence–power tradeoff
   - [ ] Non-point boundary representatives

### 代码实现状态

#### ✅ 已实现的功能

1. **Bias audit** (`experiments/01_bias_audit.py`)
   - ✅ GT 周围的 profile
   - ✅ Bias 量化

2. **CV 机制** (`oc_arcpith_rg/cv.py`)
   - ✅ Tree-grouped folds
   - ✅ `test_tree_folds_grouped()` 验证分组正确性

3. **Bias audit 边界** (`test_core.py:37-39`)
   ```python
   test_bias_audit_cannot_escape_preregistered_radius()
   # ✅ 验证 audit 不能逃出预注册半径
   ```

#### ⚠️ 概念对应问题

**文档中的"Audit"与代码中的"Bias Audit"可能不是同一概念：**

| 维度 | 文档 Independent Audit | 代码 Bias Audit |
|---|---|---|
| 目的 | 用 E_audit 反证 E_fit 的候选 | 检测 GT 附近的 local bias |
| 证据分区 | E_fit \| buffer \| E_audit | 不明确 |
| 输出 | HOLDOUT_FAILED / AUDIT_SUPPORTED / UNDERPOWERED | bias_norm, repair decision |
| Firewall | 严格禁止 E_audit 影响候选 | 不明确 |

#### ❌ 缺失的关键内容

1. **证据分区**
   - ❌ 无明确的 `E_fit | frozen buffer | E_audit` split 实现
   - ❌ 无 ring-level holdout 机制

2. **Firewall 测试**
   - ❌ 无 E_audit variants 不影响 upstream 的测试
   - ❌ 无"修改 audit threshold 不改变 fit result"的测试

3. **Null 红旗测试**
   - ❌ 无 H0 fixtures 的系统测试
   - ❌ 无 rejection rate 统计

4. **Sentinel 响应**
   - ❌ 无 applicable sentinels (translation/direction/range)
   - ❌ 无 A0/A1/A2 对照

5. **三分输出**
   - ❌ 无 HOLDOUT_FAILED / AUDIT_SUPPORTED / HOLDOUT_UNDERPOWERED 的明确输出

### 验证结论

#### 严重的概念差异 🚨

**文档的"Independent Audit"（阶段4）与代码的"Bias Audit"（实验01）是不同的验证机制：**

- **Bias Audit** 检测的是 S0 objective 自身的 local bias（在 GT 附近是否还有更低 loss）
- **Independent Audit** 验证的是用 holdout rings 反证 fitted candidate 的能力

**这意味着：**
1. 文档定义的阶段4（Independent Audit）在当前代码中**可能尚未实现**
2. 或者被实现为其他名称，但缺少文档要求的完整结构

#### 关键缺失 ❌

1. **证据分区机制未实现**：这是 Independent Audit 的基础
2. **Firewall 未建立**：无法保证 E_audit 不影响 upstream
3. **三分输出未实现**：无法区分 FAILED / SUPPORTED / UNDERPOWERED
4. **Sentinel 机制未实现**：无法测试 applicable errors 的响应

**建议行动（高优先级）**
- 明确 Independent Audit 是否已实现及实现位置
- 如果未实现，这是阶段4的阻塞性缺失
- 实现 `E_fit | buffer | E_audit` split
- 建立 Firewall 测试
- 实现 Sentinel 响应测试

---

## 阶段 5：Common Family Bias 与 Risk Route 价值

### 文档要求（第9章，963-1106行）

#### 核心问题（9.2节）
> 在 audit 已冻结的前提下，cheap scatter、residual diagnostic、可选 S1 或 domain/risk strata 是否能增量捕获"audit supported 但 oracle error 大"的 catastrophic cases，并控制正常弱几何样本的误 veto？

#### 核心概念

**Family Bias**：多个 parent rings 可能共同偏离同心圆模型（椭圆、中心漂移、低频形变），导致 S0 与 ring holdout 同时支持同一个错误 pseudocenter。

#### 必须验证的内容

1. **实验 5A：Blindness surface**
   - [ ] Deformation × coverage × distance × noise 交叉
   - [ ] 标记 `audit supported + oracle error large` cases
   - [ ] 运行 cheap scatter/residual features
   - [ ] 预注册高风险 strata 运行 S1 diagnostic

2. **实验 5B：Cheap scatter/residual 的增量**
   - [ ] Finite: projective location scatter
   - [ ] Far/axis: modulo-π direction scatter
   - [ ] 对 blind cases 有非零增量捕获
   - [ ] 不对正确 weak/far case 误 veto

3. **实验 5C：S1 diagnostic 的增量**
   - [ ] Inner-train 拟合 S0
   - [ ] Inner-train 拟合 S1 diagnostic
   - [ ] Inner-validation 比较 data loss
   - [ ] 不读取 E_audit

4. **实验 5D：Risk route 与负控制**
   - [ ] Risk 只能升不能降
   - [ ] 负控制（mixed centers, wrong normalization）

### 代码实现状态

#### ✅ 已实现的部分功能

1. **M1-RPC (S1 variant)** (`objectives.py:53-76`)
   ```python
   def m1rpc_components(h,prep,cfg,...):
       # Radial pair comparison - 检测 ring order 和 shape consistency
       for ra,rb in zip(prep.ring_order[:-1],prep.ring_order[1:]):
           sh,od=radial_pair_residuals(h,pa,pb,cfg)
           shape_loss += student(sh/m1_shape_sigma)
           order_loss += student(od/m1_order_sigma)
   ```

2. **RPC 测试** (`test_core.py:29-30`)
   ```python
   test_rpc_correct_center_small_shape_residual()
   # ✅ 验证正确中心下 shape residual 小
   ```

3. **Blind inversion CV** (`experiments/02_blind_inversion.py`)
   - ✅ Tree-grouped CV
   - ✅ Lambda tuning on inner folds

#### ⚠️ 概念对应问题

**文档的 S1 与代码的 M1-RPC 关系不明确：**

| 维度 | 文档 S1 diagnostic | 代码 M1-RPC |
|---|---|---|
| 目的 | 检测 common family bias | 改进定位（加入 radial profile consistency） |
| 训练方式 | Inner-train only | 直接用于 optimization |
| 输出 | Diagnostic score, NOT_RUN/UNDERPOWERED | 改进的 pith estimate |
| 使用 | Risk route 中可选，仅作盲区捕获 | 主定位模型 M1 |

**M1-RPC 更像是"改进的 S0"，而非文档定义的"S1 diagnostic"**

#### ❌ 缺失的关键内容

1. **Blindness surface**
   - ❌ 无 deformation fixtures (ellipse, center drift, low-freq deformation)
   - ❌ 无 blind catastrophic cases 的系统标记

2. **Cheap scatter**
   - ❌ 无 projective location scatter 计算
   - ❌ 无 modulo-π direction scatter
   - ❌ 无 per-state 的 scatter 定义

3. **S1 diagnostic as diagnostic**
   - 当前 M1-RPC 是主模型，不是 diagnostic
   - ❌ 无 inner-fold only 的 diagnostic 路径
   - ❌ 无 NOT_RUN/UNDERPOWERED 输出

4. **Risk route**
   - ❌ 无 LOW_CANDIDATE / POINT_BLOCKING / HARD_REJECT / UNKNOWN 路由
   - ❌ 无"diagnostic 只能升风险"的约束测试

5. **负控制**
   - ❌ 无 mixed centers 测试
   - ❌ 无 invalid association 测试
   - ❌ 无 common tangent rotation 测试

### 验证结论

#### 概念理解需要澄清 ⚠️

**关键问题：当前的 M1-RPC 是什么角色？**

从代码看，M1-RPC 是：
- 改进的定位模型（M0 → M1-RPC）
- 用于 blind inversion CV
- 目标是降低 bias 和改善 point estimation

从文档阶段5看，S1 应该是：
- Diagnostic only（不输出替代髓心）
- 用于检测 audit 的共同盲区
- 可选模块，无增量时 REMOVE_RUNTIME

**可能的解释：**
1. M1-RPC 对应文档的"改进的 S0"（应在阶段1验证）
2. 文档的 S1 diagnostic（阶段5）尚未实现
3. 或者 M1-RPC 同时承担两个角色，但未分离

#### 关键缺失 ❌

1. **Deformation fixtures 缺失**：无法测试 family bias
2. **Scatter features 未实现**
3. **Risk route 未实现**
4. **负控制测试缺失**

**建议行动**
- 澄清 M1-RPC 的设计角色
- 如果 M1-RPC 是改进的 S0，应回到阶段1验证
- 实现 cheap scatter features
- 构造 deformation fixtures
- 实现 risk route 逻辑

---

## 阶段 6：Full-section 与 Real-tree Shadow Run

### 文档要求（第10章，1108-1230行）

#### 核心问题（10.2节）
> 冻结后的最小管线能否在完整预注册真实队列上重放，主要失败是否可定位并与前述机制对应，是否存在足够多可审计样本支持下一阶段研究？

#### 必须验证的内容

1. **Pilot tree 选择** (10.3节)
   - [ ] Geometry easy/hard
   - [ ] Reference pith 在 FOV 内/邻近/FOV 外
   - [ ] Narrow/wide arcs
   - [ ] Low/high ring count
   - [ ] Good/poor surface quality
   - [ ] Concentric/eccentric/deformed section
   - [ ] 有独立 full-section/reference pith

2. **执行步骤** (10.5节)
   - [ ] Blind 运行全部预注册 crops
   - [ ] 保存 early REJECT 和各阶段 NOT_RUN
   - [ ] 保存完整 provenance
   - [ ] 完成全队列后才连接 reference
   - [ ] 按 tree→section→crop 分析

3. **必报结果** (10.6节)
   - [ ] State distribution
   - [ ] Runtime/memory
   - [ ] Catastrophic POINT cases
   - [ ] Tree-level summary
   - [ ] Failure review

### 代码实现状态

#### ✅ 已实现的功能

1. **真实数据运行** (`config.yaml:2-4`)
   ```yaml
   paths:
     manifest: data/manifest.jsonl
     results_dir: results_urudendro_no_background
   ```

2. **UruDendro 数据集** (EXPERIMENT_GUIDE.md 提到)
   - ✅ 4 棵 biological trees
   - ✅ 96 crops (24 per tree)

3. **Run all 脚本** (`experiments/run_all.py`)
   - ✅ 自动运行全部实验

4. **Final report** (`results_urudendro_no_background/FINAL_REPORT.md`)
   - ✅ 生成最终报告
   - ✅ `AUDIT_OUTPUT.json` 显示 smoke test 通过

5. **Manifest 验证** (`tools/validate_manifest.py`)
   - ✅ 验证 tree_id 是 biological tree

#### ⚠️ 部分对应

1. **预注册队列**
   - ⚠️ 未见 crops 的预注册设计文档
   - ⚠️ 不清楚 distance/orientation/coverage 的分层策略

2. **Blind 运行**
   - ⚠️ 不确定是否在连接 reference 前完整运行
   - ⚠️ 未见 provenance 的完整保存

3. **Failure review**
   - ⚠️ 未见按文档要求的 failure taxonomy:
     - OBSERVATION / LINEAGE / GEOMETRY / SEARCH
     - PREFLIGHT / PROFILE_TOPOLOGY
     - AUDIT_POWER / AUDIT_BLINDNESS
     - FAMILY_BIAS / DOMAIN_SHIFT
     - REFERENCE_UNCERTAINTY / IMPLEMENTATION / UNKNOWN

#### ✅ 已有初步结果

从 `AUDIT_OUTPUT.json` 的 smoke test 输出：
```json
{
  "global_pass": true,
  "direction_pass": false,
  "range_pass": true,
  "local_stationarity_pass": false,
  "decision": "NO_GO_GLOBAL_GEOMETRY"
}
```

以及 bias_audit:
```json
{
  "pass": false,
  "decision": "M1RPC_NOT_PROVEN"
}
```

以及 blind_inversion:
```json
{
  "pass": false,
  "decision": "USE_M0_BASELINE"
}
```

**这些结果表明当前系统在真实数据上遇到了挑战**

### 验证结论

#### 基础设施已就绪 ✅
- 真实数据集（UruDendro 4 trees）
- 运行脚本和报告生成
- Tree-level 分组

#### 需要明确的问题 ⚠️

1. **当前运行是否符合阶段6要求？**
   - 是否冻结了阶段0-5的配置？
   - 是否是 blind run（连接 reference 前运行）？
   - 是否保存了完整 provenance？

2. **Failure localization**
   - 当前的 `NO_GO_GLOBAL_GEOMETRY` 和 `M1RPC_NOT_PROVEN` 对应文档的哪个 failure bucket？
   - 是否能定位到具体的 first failing stage？

3. **决策路径**
   - 当前结果对应文档的哪个决策：GO_DEEPER / REVISE_AND_REPEAT / STOP_OR_DESCOPED？

**建议行动**
- 明确当前运行的阶段归属
- 补充 failure taxonomy 映射
- 按文档要求重新组织结果报告
- 明确下一步行动（修复哪个模块？重跑哪些阶段？）

---

## 跨阶段验证：文档第16章再审计清单

### 16.1 正确性检查

从代码验证情况：

- [✅] finite/axis 使用同一 `RP²` 语义
- [✅] `h↔-h` 规范化实现
- [❌] `t↔-t` 测试缺失
- [❌] frame round-trip 测试缺失
- [⚠️] near-point barrier 实现待确认
- [⚠️] equal-parent 预算实现待确认
- [❌] duplication/interpolation/fragment split 不变性测试缺失
- [❌] E_audit firewall 测试缺失
- [⚠️] provisional FINITE_COMPACT vs 最终 POINT 的区分待确认

### 16.2 模块独立性检查

- [❌] 每阶段只有一个主要操控因素 - **实验设计缺失**
- [❌] 上游配置在下游实验中冻结 - **未见明确的冻结机制**
- [❌] Complex vs cheap baseline 对比 - **B0/B1/B3 缺失**
- [⚠️] 无增量模块可删除 - **模块间依赖关系不明确**

### 16.3 参数纪律检查

- [✅] config.yaml 有明确参数
- [❌] 符号水平未映射到 manifest - **EASY/PARTIAL/HARD 等分层缺失**
- [❌] 快速参数 vs final threshold 区分 - **未见说明**
- [❌] 版本管理和依赖重跑矩阵 - **未见实现**

### 16.4 结果分析检查

- [❌] Hard red flags 优先报告 - **报告结构不完全符合**
- [⚠️] Paired comparison - **部分实验有，但不系统**
- [✅] Tree-level 汇总 - **CV 已实现**
- [❌] Underpowered/not applicable 区分 - **输出格式待确认**
- [❌] 小样本 p<0.05 替代机制判断 - **统计方法待审查**

### 16.5 目标一致性检查

- [✅] 系统允许 AXIS/RANGE/MULTIMODAL/REJECT
- [⚠️] Holdout 作为反证 - **Independent Audit 实现待确认**
- [⚠️] S1 不输出替代髓心 - **M1-RPC 角色待澄清**
- [❌] Multi-FOV 与 attribution 不阻塞 Core - **未见实现**

---

## 总体验证结论

### 实现完成度评估

| 阶段 | 核心功能 | 完整实验 | 决策路径 | 评估 |
|---|---|---|---|---|
| 阶段0 | 70% | 30% | 0% | 🟡 部分实现 |
| 阶段1 | 60% | 10% | 0% | 🔴 关键缺失 |
| 阶段2 | 30% | 0% | 0% | 🔴 概念不符 |
| 阶段3 | 50% | 20% | 0% | 🟡 部分实现 |
| 阶段4 | 20% | 0% | 0% | 🔴 关键缺失 |
| 阶段5 | 40% | 10% | 0% | 🟡 角色不清 |
| 阶段6 | 60% | 40% | 30% | 🟡 已初步运行 |

### 关键发现

#### 🔴 阻塞性问题

1. **阶段1：Baseline 缺失**
   - 无 B0/B1/B3 对照，无法证明 B2 (S0) 有独立价值
   - **影响**：无法声称通过阶段1

2. **阶段2：Preflight 概念不符**
   - 代码的 `candidate_screen` 与文档的 Preflight 是不同的概念
   - **影响**：阶段2可能尚未实现

3. **阶段4：Independent Audit 缺失**
   - 无 E_fit | buffer | E_audit split
   - 无 Firewall 保证
   - **影响**：无法声称 reliable POINT

#### 🟡 需要澄清的概念

1. **M1-RPC 的角色**
   - 是改进的 S0（应在阶段1验证）？
   - 还是 S1 diagnostic（应在阶段5验证）？
   - 还是两者兼有？

2. **Current implementation 的阶段归属**
   - 当前的 4 个实验 (00-03) 对应文档的哪些阶段？
   - Bias audit 是阶段4还是阶段5？

3. **决策路径的映射**
   - 当前的 `NO_GO_GLOBAL_GEOMETRY` / `M1RPC_NOT_PROVEN` 对应文档的哪个 STOP/REVISE 决策？

#### ✅ 已有的良好基础

1. **几何基础扎实**
   - RP² 投影几何正确
   - Finite/infinity 连续表达

2. **Tree-level 分组**
   - CV 正确实现 biological tree grouping

3. **真实数据运行**
   - 已在 UruDendro 数据集上运行
   - 有初步结果和报告

### 建议的行动路径

#### 阶段0：立即补充（1-2周）

1. 补充 lineage tracking (tree→ring→fragment→observation)
2. 添加等价变换测试套件
3. 添加负控制测试

#### 阶段1：关键补充（2-3周）

1. **明确 equal-parent 实现**（验证 omega 计算）
2. **实现 B0/B1/B3 baseline**
3. **运行配对对照实验**（同一 crop，不同模型）
4. **添加复制/插值不变性测试**

#### 阶段2&4：概念澄清（1周）

1. **澄清 Preflight 实现位置**
   - 如果未实现，标记为 TODO
   - 如果已实现但命名不同，建立映射

2. **澄清 Independent Audit 实现**
   - 如果未实现，标记为 TODO
   - 明确 Bias Audit 与 Independent Audit 的关系

#### 阶段5：角色澄清（1周）

1. **明确 M1-RPC 的设计角色**
2. **决定是否需要独立的 S1 diagnostic**
3. **如果需要，实现 cheap scatter features**

#### 阶段6：结果整理（1周）

1. **按文档要求重新组织报告**
2. **建立 failure taxonomy 映射**
3. **明确当前状态的决策路径**

### 最终评估

**当前代码实现了一个可运行的树髓定位系统**，但：

1. **与文档定义的验证协议存在较大差距**
2. **部分关键验证实验缺失**（特别是阶段1的对照实验和阶段4的 Independent Audit）
3. **概念理解需要澄清**（Preflight, Audit, M1-RPC 的角色）
4. **实验设计的系统性和完整性不足**

**要声称完成文档定义的快速验证方案，估计还需要 6-10 周的工作**，重点是：
- 补充阶段1的对照实验
- 实现或澄清阶段2的 Preflight
- 实现或澄清阶段4的 Independent Audit
- 系统化实验设计和结果报告

---

## 附录：代码与文档术语映射表

| 文档术语 | 代码实现 | 对应关系 | 备注 |
|---|---|---|---|
| S0 (Shared objective) | m0_components | ✅ 直接对应 | 需验证 equal-parent |
| B0 (one-ring pseudocenter) | ? | ❌ 未找到 | Baseline 缺失 |
| B1 (pooled micro-arcs) | ? | ❌ 未找到 | Baseline 缺失 |
| B2 (equal-parent S0) | m0_components | ⚠️ 疑似对应 | 需确认 |
| B3 (dense reference) | ? | ❌ 未找到 | Reference 缺失 |
| M1 | m1rpc_components | ⚠️ 角色不清 | RPC variant |
| Preflight | candidate_screen? | ⚠️ 概念不符 | 需澄清 |
| Independent Audit | bias_audit? | ⚠️ 概念不符 | 需澄清 |
| S1 diagnostic | m1rpc_components? | ⚠️ 角色不清 | 需澄清 |
| E_fit \| buffer \| E_audit | ? | ❌ 未找到 | 证据分区缺失 |
| POINT/AXIS/RANGE/MULTI/REJECT | classify output | ⚠️ 部分对应 | 需确认完整性 |
| Provisional FINITE_COMPACT | ? | ❌ 未明确 | 与最终 POINT 区分 |
| AUDIT_SUPPORTED/FAILED/UNDERPOWERED | ? | ❌ 未找到 | 三分输出缺失 |
| Risk route | ? | ❌ 未找到 | 风险路由缺失 |

---

**报告结束**
