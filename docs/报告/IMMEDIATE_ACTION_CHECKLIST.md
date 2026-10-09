# 立即行动检查清单

> 本清单提供可立即执行的验证任务，每个任务都有明确的脚本和预期输出

---

## 📋 今天就可以完成的验证（2-4小时）

### ✅ 任务1：验证测试套件状态（15分钟）

```bash
cd "D:\教务处实习\wood_preproject\树髓定位\代码\OC_ArcPith_RG_72h_algorithm_package_adjusted 2\OC_ArcPith_RG_72h_algorithm_package_adjusted"

# 运行所有测试
python -m pytest tests/ -v --tb=short > test_results_$(date +%Y%m%d).txt

# 检查结果
echo "=== 测试结果统计 ===" >> test_results_$(date +%Y%m%d).txt
python -m pytest tests/ --collect-only -q | wc -l >> test_results_$(date +%Y%m%d).txt
```

**预期输出**：
- 所有测试应该通过（基于 AUDIT_OUTPUT.json 显示 7/7 通过）
- 记录通过的测试数量

**验证文档要求**：
- 阶段0的部分测试（h_roundtrip, axis 等）

---

### ✅ 任务2：检查 equal-parent 实现（30分钟）

创建验证脚本：

```python
# 新文件：verify_equal_parent.py
"""验证 omega 是否实现了 equal-parent 预算"""

import numpy as np
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.io import Sample, Ring
import yaml

def circle_points(center, radius, n=100, span=(-np.pi, np.pi)):
    """生成圆弧点"""
    th = np.linspace(span[0], span[1], n)
    return np.column_stack([
        center[0] + radius * np.cos(th),
        center[1] + radius * np.sin(th)
    ])

def test_equal_parent():
    """测试 equal-parent 预算"""
    # 加载配置
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    # 创建测试样本：3个同心圆环
    center = np.array([256.0, 256.0])
    rings = [
        Ring('0', circle_points(center, 50, 100), 0),
        Ring('1', circle_points(center, 100, 200), 1),  # 2倍采样密度
        Ring('2', circle_points(center, 150, 100), 2),
    ]
    sample = Sample('test', 'T0', (512, 512), rings, center)
    
    # 预处理
    prep = prepare(sample, cfg)
    
    # 检查每个 ring 的总权重
    print("\n=== Equal-Parent 预算验证 ===\n")
    print(f"总环数: {len(prep.points_by_ring)}")
    print(f"总弧段数: {len(prep.arcs)}")
    
    ring_weights = {}
    for arc in prep.arcs:
        if arc.ring_id not in ring_weights:
            ring_weights[arc.ring_id] = 0.0
        ring_weights[arc.ring_id] += arc.omega
    
    print("\n每个环的总权重 (omega):")
    for ring_id, total_weight in sorted(ring_weights.items()):
        print(f"  Ring {ring_id}: {total_weight:.6f}")
    
    # 检查是否接近 equal-parent (每个环权重应该相同)
    weights = list(ring_weights.values())
    if len(weights) > 1:
        weight_std = np.std(weights)
        weight_mean = np.mean(weights)
        print(f"\n权重统计:")
        print(f"  均值: {weight_mean:.6f}")
        print(f"  标准差: {weight_std:.6f}")
        print(f"  变异系数: {weight_std/weight_mean:.4f}")
        
        # 判断
        if weight_std / weight_mean < 0.01:
            print("\n✅ PASS: 权重接近相等，实现了 equal-parent")
            return True
        else:
            print("\n❌ FAIL: 权重差异较大，未实现 equal-parent")
            return False
    
    return True

if __name__ == '__main__':
    test_equal_parent()
```

**运行**：
```bash
python verify_equal_parent.py
```

**预期结果**：
- 如果实现了 equal-parent：每个环的总 omega 应该相同（变异系数 < 0.01）
- 如果未实现：权重会随采样密度变化

**对应文档**：
- 阶段1，5.1节：equal-parent normalization

---

### ✅ 任务3：快速实现 B0 baseline（1小时）

```python
# 新文件：oc_arcpith_rg/baselines.py
"""简单的 baseline 定位器"""

import numpy as np
from .geometry import h_from_point, projective_angle

def b0_one_ring_pseudocenters(prep):
    """
    B0: One-ring pseudocenters + projective medoid
    
    每个环独立估计中心，然后计算投影中位数
    """
    centers_norm = []
    
    for ring_id, fragments in prep.points_by_ring.items():
        for points_px in fragments:
            # 转换到规范坐标
            points_norm = (points_px - prep.center_px) / prep.scale_px
            
            # 简单方法：用点集的质心作为该环的伪中心
            # （忽略切向信息）
            center_norm = np.mean(points_norm, axis=0)
            centers_norm.append(center_norm)
    
    if len(centers_norm) == 0:
        return None
    
    # 投影中位数（简化版：欧氏中位数）
    # 更精确的版本应该在 RP² 中计算 Fréchet mean
    median_center = np.median(centers_norm, axis=0)
    
    return h_from_point(median_center)


def b3_dense_search(prep, cfg, grid_density=5):
    """
    B3: Dense grid search (simplified reference)
    
    在高密度网格上评估 M0 objective，作为参考解
    """
    from .objectives import objective
    from .geometry import h_from_point
    
    # 在规范坐标下建立网格
    # 覆盖 [-2, 2] x [-2, 2] 的区域
    x = np.linspace(-2, 2, grid_density)
    y = np.linspace(-2, 2, grid_density)
    
    best_h = None
    best_loss = float('inf')
    
    for xi in x:
        for yi in y:
            h = h_from_point(np.array([xi, yi]))
            loss = objective(h, prep, cfg, model='m0')
            
            if loss < best_loss:
                best_loss = loss
                best_h = h
    
    return best_h, best_loss


def compare_baselines(prep, cfg, h_m0, verbose=True):
    """
    比较不同 baseline 的性能
    
    Args:
        prep: 预处理后的数据
        cfg: 配置
        h_m0: M0 优化得到的解
        verbose: 是否打印详细信息
    
    Returns:
        dict: 比较结果
    """
    from .geometry import projective_angle, point_from_h
    from .objectives import objective
    
    results = {}
    
    # B0
    h_b0 = b0_one_ring_pseudocenters(prep)
    if h_b0 is not None:
        loss_b0 = objective(h_b0, prep, cfg, model='m0')
        d_b0_m0 = projective_angle(h_b0, h_m0)
        results['B0'] = {
            'h': h_b0,
            'loss': loss_b0,
            'd_to_m0': d_b0_m0,
            'point': point_from_h(h_b0)
        }
    
    # M0 (B2)
    loss_m0 = objective(h_m0, prep, cfg, model='m0')
    results['M0'] = {
        'h': h_m0,
        'loss': loss_m0,
        'd_to_m0': 0.0,
        'point': point_from_h(h_m0)
    }
    
    # B3 (simplified)
    h_b3, loss_b3 = b3_dense_search(prep, cfg, grid_density=20)
    d_b3_m0 = projective_angle(h_b3, h_m0)
    results['B3'] = {
        'h': h_b3,
        'loss': loss_b3,
        'd_to_m0': d_b3_m0,
        'point': point_from_h(h_b3)
    }
    
    if verbose:
        print("\n=== Baseline 比较 ===\n")
        for name, res in results.items():
            print(f"{name}:")
            print(f"  Loss: {res['loss']:.6f}")
            if res['point'] is not None:
                print(f"  Point: ({res['point'][0]:.3f}, {res['point'][1]:.3f})")
            else:
                print(f"  Point: AXIS (infinity)")
            print(f"  d_RP2 to M0: {res['d_to_m0']:.6f} rad")
            print()
        
        # 判断 M0 是否优于 B0
        if 'B0' in results:
            improvement = results['B0']['loss'] - results['M0']['loss']
            print(f"M0 vs B0 改善: {improvement:.6f}")
            if improvement > 0:
                print("✅ M0 优于 B0")
            else:
                print("❌ M0 未优于 B0")
        
        # 判断 M0 是否接近 B3
        if 'B3' in results:
            gap = results['M0']['loss'] - results['B3']['loss']
            print(f"\nM0 vs B3 gap: {gap:.6f}")
            if gap < 0.1:
                print("✅ M0 接近参考解 B3")
            else:
                print("⚠️  M0 与 B3 有较大差距，可能是搜索不充分")
    
    return results
```

**测试脚本**：

```python
# 新文件：test_baselines.py
"""测试 baseline 比较"""

import yaml
import numpy as np
from oc_arcpith_rg.preprocess import prepare
from oc_arcpith_rg.io import Sample, Ring
from oc_arcpith_rg.optimizer import optimize
from oc_arcpith_rg.baselines import compare_baselines

def circle_points(center, radius, n=100):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(th),
        center[1] + radius * np.sin(th)
    ])

def test_on_synthetic():
    """在合成数据上测试"""
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)
    
    # 创建同心圆样本
    center = np.array([256.0, 256.0])
    rings = [
        Ring('0', circle_points(center, 50), 0),
        Ring('1', circle_points(center, 100), 1),
        Ring('2', circle_points(center, 150), 2),
    ]
    sample = Sample('test', 'T0', (512, 512), rings, center)
    
    # 预处理
    prep = prepare(sample, cfg)
    
    # M0 优化
    result_m0 = optimize(prep, cfg, model='m0')
    h_m0 = result_m0['h']
    
    # 比较 baselines
    results = compare_baselines(prep, cfg, h_m0, verbose=True)
    
    return results

if __name__ == '__main__':
    test_on_synthetic()
```

**运行**：
```bash
python test_baselines.py
```

**预期结果**：
- M0 的 loss 应该低于 B0（证明 equal-parent S0 有价值）
- M0 与 B3 的 loss 应该接近（证明搜索充分）

**对应文档**：
- 阶段1，实验1A：核心定位器配对比较

---

### ✅ 任务4：分析当前真实数据结果（30分钟）

```python
# 新文件：analyze_current_results.py
"""分析当前在 UruDendro 上的运行结果"""

import json
import os
from pathlib import Path

def analyze_screen_results():
    """分析 candidate screening 结果"""
    results_dir = Path('results_urudendro_no_background')
    
    screen_file = results_dir / 'step3_v2' / 'm0_screen.json'
    
    if screen_file.exists():
        with open(screen_file) as f:
            data = json.load(f)
        
        print("\n=== Candidate Screening 结果 (M0) ===\n")
        print(f"决策: {data.get('decision', 'N/A')}")
        print(f"Global pass: {data.get('global_pass', 'N/A')}")
        print(f"Direction pass: {data.get('direction_pass', 'N/A')}")
        print(f"Range pass: {data.get('range_pass', 'N/A')}")
        
        if 'per_tree' in data:
            print(f"\n各树统计:")
            for tree in data['per_tree']:
                print(f"  {tree['tree_id']}:")
                print(f"    GT beats fraction: {tree.get('global_truth_beats_fraction', 'N/A')}")
                print(f"    Local bias flag: {tree.get('local_bias_flag', 'N/A')}")
        
        # 解读
        print("\n=== 解读 ===")
        if data.get('decision') == 'NO_GO_GLOBAL_GEOMETRY':
            print("❌ 当前判定: 全局几何不可辨识")
            print("   含义: 候选筛选无法排除错误候选")
            print("   对应文档: 可能是阶段1失败 (S0 无独立价值)")
            print("   建议: 检查 B0 vs M0 对比，确认 M0 是否真的有改善")
        
        return data
    else:
        print(f"未找到文件: {screen_file}")
        return None


def analyze_bias_audit():
    """分析 bias audit 结果"""
    results_dir = Path('results_urudendro_no_background')
    
    bias_file = results_dir / 'bias_audit' / 'repair_decision.json'
    
    if bias_file.exists():
        with open(bias_file) as f:
            data = json.load(f)
        
        print("\n=== Bias Audit 结果 ===\n")
        print(f"决策: {data.get('decision', 'N/A')}")
        print(f"Pass: {data.get('pass', 'N/A')}")
        print(f"配对样本数: {data.get('paired_samples', 'N/A')}")
        print(f"树数量: {data.get('tree_count', 'N/A')}")
        print(f"中位 bias 降低: {data.get('median_tree_bias_reduction', 'N/A')}")
        print(f"通过的树比例: {data.get('tree_pass_fraction', 'N/A')}")
        
        # 解读
        print("\n=== 解读 ===")
        if data.get('decision') == 'M1RPC_NOT_PROVEN':
            print("❌ 当前判定: M1-RPC 未被证明")
            print("   含义: M1-RPC 未能稳定降低 bias")
            print("   对应文档: 可能是阶段5问题 (family bias 检测失败)")
            print("   建议: 使用 M0 baseline，暂不使用 M1-RPC")
        
        return data
    else:
        print(f"未找到文件: {bias_file}")
        return None


def analyze_cv_results():
    """分析 CV 结果"""
    results_dir = Path('results_urudendro_no_background')
    
    cv_file = results_dir / 'blind' / 'g2.json'
    
    if cv_file.exists():
        with open(cv_file) as f:
            data = json.load(f)
        
        print("\n=== Cross-Validation 结果 ===\n")
        print(f"决策: {data.get('decision', 'N/A')}")
        print(f"Pass: {data.get('pass', 'N/A')}")
        print(f"Folds: {data.get('nfold', 'N/A')}")
        print(f"改善的 folds: {data.get('improved_folds', 'N/A')} / {data.get('required_improved_folds', 'N/A')}")
        print(f"中位点改善: {data.get('median_point_improvement_norm', 'N/A')}")
        
        # 解读
        print("\n=== 解读 ===")
        if data.get('decision') == 'USE_M0_BASELINE':
            print("⚠️  当前判定: 使用 M0 baseline")
            print("   含义: M1-RPC 的改善不足或不稳定")
            print("   对应文档: 阶段1或5的问题")
            print("   建议: 先验证 M0 本身的价值 (B0 vs M0)")
        
        return data
    else:
        print(f"未找到文件: {cv_file}")
        return None


def main():
    """主分析函数"""
    print("=" * 60)
    print("当前结果分析")
    print("=" * 60)
    
    screen_data = analyze_screen_results()
    bias_data = analyze_bias_audit()
    cv_data = analyze_cv_results()
    
    print("\n" + "=" * 60)
    print("总体判断")
    print("=" * 60)
    
    print("\n基于当前结果，系统状态:")
    print("1. Candidate screening: NO_GO")
    print("2. Bias audit: M1RPC NOT PROVEN")
    print("3. CV: USE M0 BASELINE")
    
    print("\n对应文档验证阶段的解读:")
    print("- 可能卡在阶段1: M0 本身可能没有独立价值")
    print("- 需要验证: M0 vs B0 (one-ring pseudocenter)")
    print("- 需要验证: M0 vs B3 (dense search)")
    
    print("\n建议的下一步行动:")
    print("1. 运行 test_baselines.py，验证 M0 vs B0 vs B3")
    print("2. 如果 M0 优于 B0，问题可能在后续阶段")
    print("3. 如果 M0 未优于 B0，需要修正 S0 实现或重新设计")


if __name__ == '__main__':
    main()
```

**运行**：
```bash
python analyze_current_results.py
```

**预期输出**：
- 当前系统在 UruDendro 上的判定状态
- 对应文档的哪个阶段
- 建议的下一步行动

---

### ✅ 任务5：创建概念澄清问题清单（15分钟）

直接编辑并填写：

```markdown
# 新文件：CLARIFICATION_NEEDED.md

## 需要项目组讨论和澄清的概念问题

### 紧急度：🔴 高

#### Q1: M1-RPC 的设计角色

**问题描述**：
- 代码中 M1-RPC 被用作主定位模型（experiments/02_blind_inversion.py）
- 但文档阶段5的 S1 要求"不输出替代髓心，只作 diagnostic"
- 这两者矛盾

**需要澄清**：
- [ ] M1-RPC 是改进的 S0（应在阶段1验证）吗？
- [ ] M1-RPC 是 S1 diagnostic（应在阶段5验证）吗？
- [ ] 还是两个概念被混在一起了？

**如果是改进的 S0**：
- 应该回到阶段1，与 B0/B1/M0(B2)/B3 一起对比
- M1-RPC 应该叫 B2'，M0 叫 B2
- 阶段5的 S1 diagnostic 需要另外实现

**如果是 S1 diagnostic**：
- 不应该输出替代髓心
- 应该只输出风险评分
- 02_blind_inversion.py 的使用方式不对

**决策者**：___________
**决策日期**：___________
**决策结果**：___________

---

#### Q2: Preflight 是否已实现

**问题描述**：
- 文档阶段2定义的 Preflight：在 S0 搜索前判断可行性
- 代码的 candidate_screen：在 M0/M1 运行后筛选候选
- 这是两个不同时机的模块

**需要澄清**：
- [ ] candidate_screen 就是 Preflight 吗？
- [ ] 如果不是，Preflight 在哪里实现了？
- [ ] 如果未实现，是否必须实现？

**如果 candidate_screen 不是 Preflight**：
- 阶段2（Preflight）属于缺失状态
- 需要评估是否阻塞发表

**如果 candidate_screen 就是 Preflight**：
- 需要建立文档术语到代码的映射
- 需要验证是否满足文档的 P0/P1/P2/P3 要求

**决策者**：___________
**决策日期**：___________
**决策结果**：___________

---

#### Q3: Independent Audit 是否已实现

**问题描述**：
- 文档阶段4要求：E_fit | buffer | E_audit 证据分区
- 代码的 bias_audit：检测 GT 附近的 local bias
- 这是两个不同的审计机制

**需要澄清**：
- [ ] bias_audit 就是 Independent Audit 吗？
- [ ] 如果不是，Independent Audit 在哪里？
- [ ] 如果未实现，是否阻塞"reliable POINT"主张？

**如果未实现 Independent Audit**：
- **这是严重缺失**，直接影响可靠性主张
- 根据文档1.3节第4条：E_audit 不参与搜索是绝对约束
- 根据文档1.3节第7条：无 audit 不能声称 reliable POINT

**如果已实现但位置不明**：
- 需要指出代码位置
- 需要验证防火墙机制

**决策者**：___________
**决策日期**：___________
**决策结果**：___________

---

### 紧急度：🟡 中

#### Q4: Equal-parent 的实现验证

**问题**：arc.omega 是否真的实现了 equal-parent？

**验证方法**：
- [ ] 运行 verify_equal_parent.py
- [ ] 检查不同采样密度的环是否权重相同

**结果**：___________

---

#### Q5: 五态的完整性

**问题**：RANGE_UNCERTAIN 的实现路径是什么？

**验证方法**：
- [ ] 阅读 observability.py 的 classify 函数
- [ ] 检查是否有 RANGE_UNCERTAIN 输出

**结果**：___________

---

### 紧急度：🟢 低

#### Q6: 发表策略

**问题**：论文需要多完整的验证？

**选项**：
- A. 完整实现文档的6个阶段（6-10周）
- B. 最小可行验证，其余在文档中说明（2-3周）

**决策者**：___________
**决策日期**：___________
**决策结果**：___________
```

---

## 📊 本周可完成的验证进度表

| 任务 | 时间 | 状态 | 输出 |
|---|---|---|---|
| 运行测试套件 | 15分钟 | ⬜ | test_results_YYYYMMDD.txt |
| 验证 equal-parent | 30分钟 | ⬜ | equal_parent_check.txt |
| 实现 B0 baseline | 1小时 | ⬜ | baselines.py |
| 测试 B0 vs M0 | 30分钟 | ⬜ | baseline_comparison.txt |
| 分析当前结果 | 30分钟 | ⬜ | result_analysis.txt |
| 创建问题清单 | 15分钟 | ⬜ | CLARIFICATION_NEEDED.md |

**总计：约 3.5 小时**

---

## 🎯 完成后你将获得

1. **明确的测试状态**：知道哪些测试通过了
2. **Equal-parent 验证**：确认核心实现是否正确
3. **Baseline 对比**：初步验证 M0 的价值
4. **当前结果解读**：理解系统在真实数据上的表现
5. **概念问题清单**：知道需要讨论什么

---

## 📝 下一步计划（下周）

根据今天的验证结果：

### 如果 M0 优于 B0：
- M0 有独立价值，阶段1部分通过
- 继续检查后续阶段（Preflight, Audit）

### 如果 M0 未优于 B0：
- **阻塞性问题**：S0 设计可能有问题
- 需要修正 S0 实现或重新设计
- 暂停后续阶段，先解决阶段1

### 如果 Equal-parent 未正确实现：
- **阻塞性问题**：核心假设未满足
- 需要修正 omega 计算
- 重新运行所有实验

---

## 💡 提示

- 每个任务都是独立的，可以单独运行
- 建议按顺序执行，因为后面的任务依赖前面的结果
- 遇到问题随时停下来，不要强行继续
- 记录所有输出，便于后续讨论

---

**准备好开始了吗？从任务1开始！** 🚀
