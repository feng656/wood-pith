# ArcPith v5-slim 验证总结（中文）

## 一、验证工作概述

本次验证对比了：
- **文档**：`ARCPITH_v5-slim_sequential_rapid_validation_protocol-v2.md`（1495行，6个验证阶段）
- **代码**：`OC_ArcPith_RG_72h_algorithm_package_adjusted 2`

验证方法：逐阶段检查文档要求的实验、测试和决策路径是否在代码中实现。

## 二、核心发现

### 🔴 三个阻塞性问题

#### 1. 阶段1：缺少对照基线（Baseline）

**文档要求**：
- B0: 单环伪中心 + 投影中位数
- B1: 合并微弧 S0（无等预算）
- B2: 等预算共享 S0（核心候选）
- B3: 密集搜索参考解

**代码现状**：
- ✅ 有 M0（疑似对应 B2）
- ❌ 缺少 B0、B1、B3

**影响**：
- **无法证明 S0 的独立增量价值**（文档阶段1的核心问题）
- 无法判断是否应该 `KEEP S0` 或 `STOP CORE`

**需要做什么**：
```python
# 需要实现三个对照系统
def b0_one_ring_pseudocenter(rings):
    """每个环单独估计中心，然后投影中位数"""
    pass

def b1_pooled_microarcs(observations):
    """直接合并所有微弧，不做环间等权"""
    pass

def b3_dense_search(observations, high_budget=True):
    """密集网格 + 多次精化，作为参考解"""
    pass

# 然后在同一 crop 上运行 B0, B1, B2(M0), B3
# 比较 d_RP2, state error, stability
```

#### 2. 阶段2：Preflight 概念不对应

**文档定义的 Preflight**：
- 在 S0 搜索**之前**判断可行性
- 只读 candidate-free 的 lineage/coverage
- 输出：POINT_CANDIDATE / AXIS_ONLY / INSUFFICIENT

**代码的 candidate_screen**：
- 在 M0/M1 运行**之后**筛选候选
- 需要已计算的候选 loss
- 输出：NO_GO_GLOBAL_GEOMETRY / BIAS_REPAIR_REQUIRED

**这是两个不同的模块！**

**需要澄清**：
1. 文档的 Preflight（阶段2）是否已实现？在哪里？
2. 如果未实现，这是严重缺失
3. 如果已实现但命名不同，需要建立映射关系

#### 3. 阶段4：Independent Audit 缺失

**文档要求**：
- 证据分区：`E_fit | frozen buffer | E_audit`
- E_audit **严格不参与**候选选择、profile、topology
- 防火墙测试：修改 E_audit 不影响 upstream
- 三分输出：HOLDOUT_FAILED / AUDIT_SUPPORTED / UNDERPOWERED

**代码现状**：
- 有 `bias_audit`（检测 GT 附近的 local bias）
- ❌ 没有证据分区机制
- ❌ 没有防火墙测试
- ❌ 没有三分输出

**影响**：
- **无法声称 "reliable POINT"**
- 这是文档 1.3 节的绝对约束第4条：E_audit 不参与搜索和候选选择

**需要做什么**：
```python
# 1. 实现证据分区
def split_evidence(rings, buffer_radius):
    """将 parent rings 分为 E_fit 和 E_audit"""
    e_fit = []
    e_audit = []
    # ... 按距离或 ring order 分离
    return e_fit, e_audit

# 2. 锁定候选后才打开 E_audit
candidates = optimize_on_E_fit_only(e_fit)  # 第一阶段
lock_candidates(candidates)
audit_results = audit_with_E_audit(candidates, e_audit)  # 第二阶段

# 3. 防火墙测试
def test_audit_firewall():
    e_fit = load_e_fit()
    candidates_before = optimize(e_fit)
    
    # 改变 E_audit
    e_audit_v1 = create_audit_variant_1()
    e_audit_v2 = create_audit_variant_2()
    
    # 验证候选不变
    assert candidates_before == optimize(e_fit)
```

### 🟡 三个需要澄清的概念问题

#### 4. M1-RPC 的角色

**代码中 M1-RPC**：
```python
def m1rpc_components(h, prep, cfg, ...):
    # 加入 radial profile consistency
    # 加入 ring order soft constraint
    # 用于改进定位
```

**问题**：M1-RPC 是什么？
- A. 改进的 S0（应在阶段1验证）？
- B. S1 diagnostic（应在阶段5作为可选盲区检测）？
- C. 两者兼有？

**从代码使用看**：
- `experiments/02_blind_inversion.py` 把它当作主定位模型
- 有 lambda tuning 和 CV
- 输出改进的 pith estimate

**→ 更像是"改进的 S0"（选项A），不是 diagnostic**

**但是**：
- 文档阶段5的 S1 要求"不输出替代髓心，只作 diagnostic"
- M1-RPC 明显输出髓心位置

**需要澄清**：
1. M1-RPC 应该归入哪个阶段验证？
2. 如果是改进的 S0，需要回到阶段1，与 B0/B1/B2/B3 一起对比
3. 文档阶段5的 S1 diagnostic 是否还需要单独实现？

#### 5. Bias Audit 与 Independent Audit 的关系

**代码的 Bias Audit** (`01_bias_audit.py`):
- 在 GT 周围做二维 profile
- 检测是否存在 loss 更低的局部最优点
- 量化 bias_norm, bias_px

**文档的 Independent Audit**（阶段4）:
- 用 holdout rings 反证 fitted candidate
- 检测 overfitting 和偶然一致
- 输出 HOLDOUT_FAILED / SUPPORTED / UNDERPOWERED

**这是两个不同的审计机制！**

| 对比维度 | Bias Audit | Independent Audit |
|---|---|---|
| 检测对象 | S0 objective 自身的局部偏差 | Fitted candidate 的过拟合 |
| 证据来源 | 同一证据集 | Holdout 证据 |
| 参考标准 | GT（仅实验期可用） | 无需 GT |
| 输出 | bias 量化 | 三分决策 |

**需要澄清**：
- Bias Audit 检测的问题属于文档的哪个阶段？
- Independent Audit 是否已实现？在哪里？

#### 6. 当前实验的阶段归属

**代码有 4 个实验**：
- `00_screen.py`
- `01_bias_audit.py`
- `02_blind_inversion.py`
- `03_observability.py`

**它们对应文档的哪些阶段？**

| 实验 | 可能对应的文档阶段 | 匹配度 |
|---|---|---|
| 00_screen | 阶段2 Preflight? | ⚠️ 概念不符 |
| 01_bias_audit | 阶段4 Audit? 或阶段5? | ⚠️ 概念不符 |
| 02_blind_inversion | 阶段1 S0价值? | ⚠️ 但缺 B0/B1/B3 |
| 03_observability | 阶段3 Topology | ✅ 较符合 |

**需要做什么**：
- 明确建立实验到阶段的映射
- 如果某阶段未实现，明确标记为 TODO
- 如果某实验不属于任何阶段，说明其独立目的

### ✅ 做得好的地方

#### 1. 几何基础扎实
```python
# geometry.py 正确实现了：
- RP² 投影几何
- finite/infinity 连续表达
- h/-h 规范化
- φ,κ 参数化（κ=0 表示 infinity）
```

#### 2. Tree-level 分组正确
```python
# cv.py 和 test_core.py
test_tree_folds_grouped()  # ✅ 验证同 tree 的 crops 分在同一 fold
```

#### 3. 已在真实数据上运行
- UruDendro 数据集（4 棵树，96 crops）
- 有初步结果和报告
- Smoke test 通过

## 三、实现完成度评估表

| 阶段 | 文档要求 | 代码实现 | 完成度 | 阻塞性 |
|---|---|---|---|---|
| 阶段0 数值正确性 | 4个实验子类 | 部分测试 | 40% | ⚠️ 中 |
| 阶段1 S0核心价值 | B0/B1/B2/B3对照 | 仅有M0/M1 | 30% | 🔴 高 |
| 阶段2 Preflight | P0/P1/P2/P3层级 | 概念不符 | 20% | 🔴 高 |
| 阶段3 五态Topology | Profile + fixtures | 部分实现 | 50% | 🟡 中 |
| 阶段4 Independent Audit | E_fit\|buffer\|E_audit | 缺失 | 10% | 🔴 高 |
| 阶段5 Bias Route | Scatter + S1 + Risk | 角色不清 | 30% | 🟡 中 |
| 阶段6 真实数据 | Shadow run | 已初步运行 | 60% | 🟢 低 |

**总体完成度：约 35%**

## 四、优先级行动建议

### P0 - 概念澄清（1周内完成）

**目标**：明确当前代码与文档的对应关系

1. **召开技术讨论会**，澄清：
   - M1-RPC 的设计角色（S0变体 vs S1 diagnostic）
   - Preflight 是否已实现（在哪里？）
   - Independent Audit 是否已实现（在哪里？）

2. **建立映射文档**：
   ```markdown
   # 代码-文档映射表
   | 文档术语 | 代码实现 | 位置 | 状态 |
   |---|---|---|---|
   | B2 (equal-parent S0) | m0_components | objectives.py | ✅ 已实现 |
   | M1-RPC | ? | ? | ⚠️ 角色待定 |
   | Preflight | ? | ? | ❌ 待确认 |
   | Independent Audit | ? | ? | ❌ 待确认 |
   ```

3. **决策**：如果某关键阶段确认未实现，评估：
   - 是否必须实现才能发表？
   - 还是可以在文档中明确"未验证"？

### P1 - 补充阻塞性缺失（2-4周）

#### 如果要完整遵循文档验证协议：

**任务1：实现阶段1对照**（1周）
```python
# 新文件：experiments/stage1_s0_value.py
- 实现 B0, B1, B3
- 在同一 synthetic crops 上运行 B0/B1/B2(M0)/B3
- 比较 d_RP2, state error, stability
- 验证复制/插值不变性
- 输出：KEEP / STOP CORE 决策
```

**任务2：实现或澄清阶段2 Preflight**（1周）
```python
# 新文件：experiments/stage2_preflight.py（如果确认未实现）
- 实现 P0 (ring count only)
- 实现 P1 (+ arc span)
- 实现 P2 (+ F_geom)
- 实现 P3 (+ correlation)
- 比较 unsafe pass, unnecessary reject
- 输出：KEEP / SIMPLIFY / REMOVE 决策
```

**任务3：实现阶段4 Independent Audit**（1-2周）
```python
# 新文件：experiments/stage4_independent_audit.py
- 实现证据分区 E_fit | buffer | E_audit
- 实现防火墙测试
- 实现 Sentinel 响应测试
- 实现三分输出
- 输出：KEEP / STOP RELIABLE POINT 决策
```

### P2 - 系统化实验设计（2-3周）

**任务4：构造系统化 fixtures**
```python
# 新文件：data/fixtures/
fixtures = {
    'stage1_s0': {
        'concentric': [...],  # distance × arc × rings × noise
        'partial_arcs': [...],
        'noise_variants': [...]
    },
    'stage3_topology': {
        'F-POINT': [...],
        'F-AXIS': [...],
        'F-RANGE': [...],
        'F-MULTI': [...],
        'F-REJECT': [...]
    },
    'stage5_bias': {
        'ellipse': [...],
        'center_drift': [...],
        'deformation': [...],
        'mixed_centers': [...]
    }
}
```

**任务5：实现版本管理和依赖矩阵**
```python
# 新文件：version_control.py
class ExperimentVersion:
    def __init__(self, stage, config_hash, code_hash):
        self.stage = stage
        self.config_hash = config_hash
        self.code_hash = code_hash
    
    def check_upstream_changed(self):
        """检查上游配置是否改变，决定是否需要重跑"""
        pass
```

### P3 - 文档和报告对齐（1周）

**任务6：按文档格式重新组织报告**
```markdown
# 每个阶段一个结果文件
results/
├── stage0_numerical/
│   ├── invariance_tests.json
│   ├── firewall_tests.json
│   └── decision.txt  # PASS / STOP
├── stage1_s0_value/
│   ├── b0_vs_b1_vs_b2_vs_b3.json
│   ├── duplication_invariance.json
│   └── decision.txt  # KEEP / STOP CORE
├── stage2_preflight/
│   └── decision.txt  # KEEP / SIMPLIFY / REMOVE
...
```

**任务7：实现 Failure Taxonomy**
```python
FAILURE_BUCKETS = [
    'OBSERVATION', 'LINEAGE', 'GEOMETRY', 'SEARCH',
    'PREFLIGHT', 'PROFILE_TOPOLOGY',
    'AUDIT_POWER', 'AUDIT_BLINDNESS',
    'FAMILY_BIAS', 'DOMAIN_SHIFT',
    'REFERENCE_UNCERTAINTY', 'IMPLEMENTATION', 'UNKNOWN'
]

def localize_failure(result):
    """定位首个失败阶段和失败类型"""
    pass
```

## 五、两种可选路径

### 路径A：完整遵循文档（6-10周）

**目标**：实现文档定义的全部 6 个阶段

**优点**：
- 验证系统完整、严谨
- 可以声称"通过快速验证方案"
- 为正式实验（E3）打好基础

**缺点**：
- 工作量大（估计6-10周）
- 需要大量 synthetic fixtures
- 可能延迟论文发表

**适用场景**：
- 这是博士论文的核心方法
- 需要声称"已建立完整验证体系"
- 有足够时间和资源

### 路径B：最小可行验证（2-3周）

**目标**：补充最关键的缺失，其余在文档中明确说明

**必须做**：
1. 阶段1：至少实现 B0 和 B3，证明 M0 有增量价值
2. 阶段0：补充基本的等价变换测试
3. 明确说明哪些验证"已实现"，哪些"计划中"

**在论文/报告中诚实说明**：
```markdown
## 验证状态

✅ 已验证：
- 投影几何正确性（阶段0部分）
- S0 相对简单基线的增量（阶段1部分）
- 五态分类框架（阶段3部分）
- Tree-level CV（阶段6部分）

⚠️ 部分验证：
- 等价变换完整性（阶段0）
- 密集参考解对照（阶段1）

❌ 计划中：
- Preflight 层级对照（阶段2）
- Independent Audit 完整框架（阶段4）
- Bias blindness 系统测试（阶段5）
```

**优点**：
- 工作量可控
- 诚实透明
- 可以先发表，后续继续完善

**缺点**：
- 无法声称"完整验证"
- 审稿人可能质疑验证不足

**适用场景**：
- 需要尽快发表初步结果
- 资源有限
- 可以接受"部分验证"的定位

## 六、立即可以做的事情（本周）

### 1. 运行现有测试并记录结果

```bash
cd "D:\教务处实习\wood_preproject\树髓定位\代码\OC_ArcPith_RG_72h_algorithm_package_adjusted 2\OC_ArcPith_RG_72h_algorithm_package_adjusted"

# 运行现有测试
python -m pytest tests/ -v > test_results_current.txt

# 运行现有实验
python experiments/run_all.py --config config.yaml

# 查看最终报告
cat results_urudendro_no_background/FINAL_REPORT.md
```

### 2. 读取并分析最终报告

```python
# 新脚本：analyze_current_results.py
import json

# 读取当前结果
with open('results_urudendro_no_background/FINAL_REPORT.md') as f:
    report = f.read()

# 提取关键指标
# - State distribution
# - Error statistics
# - Failure modes

# 映射到文档的决策框架
# - 对应哪个阶段？
# - 应该 GO_DEEPER / REVISE / STOP？
```

### 3. 验证 equal-parent 实现

```python
# 新脚本：verify_equal_parent.py
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.objectives import m0_components

# 加载一个样本
sample = load_sample()
prep = prepare(sample, config)

# 检查 omega 的计算
print("Number of rings:", len(prep.points_by_ring))
print("Arc weights (omega):")
for arc in prep.arcs:
    print(f"  Ring {arc.ring_id}, Arc {arc.arc_id}: omega = {arc.omega}")

# 验证：同一 ring 的不同 arcs 的 omega 之和是否相同
# 如果是 equal-parent，应该每个 ring 贡献 1/n_rings
```

### 4. 快速实现 B0 baseline

```python
# 新文件：oc_arcpith_rg/baselines.py
import numpy as np
from .geometry import h_from_point

def b0_one_ring_pseudocenters(prep):
    """B0: 每个环单独估计中心，然后投影中位数"""
    centers = []
    
    for ring_id, fragments in prep.points_by_ring.items():
        for points in fragments:
            # 简单方法：用点集的几何中心作为伪中心
            center = np.mean(points, axis=0)
            centers.append(center)
    
    # 投影中位数（在 RP² 中）
    if len(centers) == 0:
        return None
    
    # 简化版：用欧氏中位数
    median_center = np.median(centers, axis=0)
    return h_from_point(median_center)

# 然后对比
h_b0 = b0_one_ring_pseudocenters(prep)
h_m0 = optimize_m0(prep, config)
print(f"B0 vs M0 distance: {projective_angle(h_b0, h_m0)}")
```

### 5. 创建问题清单文档

```markdown
# 新文件：QUESTIONS_FOR_DISCUSSION.md

## 需要澄清的概念问题

### Q1: M1-RPC 的设计角色
- 是改进的 S0（应在阶段1验证）？
- 是 S1 diagnostic（应在阶段5验证）？
- 应该与哪些 baseline 对比？

### Q2: Preflight 的实现位置
- candidate_screen 是 Preflight 吗？
- 如果不是，Preflight 在哪里？
- 如果未实现，是否必须实现？

### Q3: Independent Audit 的实现位置
- bias_audit 是 Independent Audit 吗？
- 如果不是，是否需要实现证据分区？
- 如果未实现，是否可以暂时跳过？

### Q4: 发表策略
- 是否需要完整实现 6 个阶段？
- 可以接受"部分验证"吗？
- 论文的主要创新点是什么？

## 待验证的技术问题

### T1: Equal-parent 预算
- 当前的 omega 是否实现了 equal-parent？
- 如何验证？

### T2: 五态完整性
- RANGE_UNCERTAIN 的实现路径是什么？
- Provisional vs Final 状态的区分在哪里？

### T3: 当前结果的解读
- NO_GO_GLOBAL_GEOMETRY 意味着什么？
- M1RPC_NOT_PROVEN 意味着什么？
- 下一步应该做什么？
```

## 七、总结

### 当前状态
- ✅ **有一个可运行的系统**
- ⚠️ **但与文档定义的验证协议存在较大差距**
- 🔴 **关键验证实验缺失**（阶段1对照、阶段4 Audit）

### 估计工作量
- **最小可行验证**：2-3周
- **完整遵循文档**：6-10周

### 建议
1. **本周**：概念澄清 + 读取当前结果
2. **下周**：决定路径（路径A 或 路径B）
3. **后续**：按优先级执行补充任务

### 关键决策点
**你需要决定**：
- 这个验证协议的目的是什么？（博士论文？期刊论文？内部验证？）
- 可以接受的完成度是多少？（100%？70%？50%？）
- 有多少时间和资源？（10周？5周？2周？）

**根据你的回答，我可以帮你制定更具体的行动计划。**
