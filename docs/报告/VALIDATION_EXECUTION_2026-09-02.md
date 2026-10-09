# ArcPith v5-slim 参数验证执行记录（2026-09-02）

## 1. 任务边界

本记录把三个 Markdown 文件中的内容分为两类：

- **本次用户请求**：按协议顺序验证参数是否必要，并为每个判断提供证据。
- **附件中的操作指令**：作为验证协议和检查清单使用；其中示例脚本、发表策略、未来实现建议不是已授权的自动实现任务。

本次只读取和运行现有代码、测试及已有结果，没有修改算法参数，也没有把已有实验结果重新解释成正式准确率、显著性、type-I/power、CALIBRATED 或生产结论。

代码目录：`D:/教务处实习/wood_preproject/树髓定位/代码/OC_ArcPith_RG_72h_algorithm_package_adjusted 2/OC_ArcPith_RG_72h_algorithm_package_adjusted`

## 2. 执行过的检查与原始证据

### 2.1 测试套件

命令：`python -m pytest tests/ -v --tb=short`

- 13/13 通过。
- 覆盖：`h_roundtrip`、AXIS、候选库确定性、缺失 parent ring 不伪造相邻关系、fragment 保留、profile 连通分量、tree-level fold 分组、M0 向量化/标量一致性等。
- 这只能证明局部单元测试通过，**不能证明协议阶段 0–6 通过**。

### 2.2 Manifest、lineage 和坐标来源

命令：`python tools/validate_manifest.py data/manifest.jsonl --require-order --require-images --require-coordinate-provenance`

结果：通过。

- 5,182 条 crop；4 棵 biological tree：T0、T2、T4、T6；102 个 section。
- 72,975 条 curve；全部有显式 parent-ring order。
- 2,670 条记录包含同一 parent ring 的多个 fragment，代码保持分离。
- 图片存在，`pith_full_px - crop_origin_px == pith_px`。
- 但协议要求的 `input_hash / observation_hash / lineage_hash` 在 manifest metadata 中均未发现（0/5,182）。因此数据可追溯性通过了当前 manifest 检查，但没有达到协议规定的完整 hash schema。

### 2.3 数值等价变换（补充确定性检查）

对 finite、infinity、far finite 三个齐次解执行 `λ∈{-10,-2,-0.5,0.5,2,10}`、frame round-trip、`t→-t`、seam 邻域检查：

- `h` 同尺度/符号的 projective angle 最大差为 0。
- frame round-trip 最大坐标差为 0。
- `t` 与 `-t` 的 residual 最大差为 0。
- `φ≈±π` seam 的 projective angle 差约 `1.49e-8 rad` 或 0。

这些结果支持几何基础实现正确；但协议阶段 0 还要求 near-point barrier、E_audit firewall、B3 搜索充分性等检查，尚未完成。

### 2.4 Equal-parent 预算

代码路径：`oc_arcpith_rg/preprocess.py:52-57`。每个 ring 的总预算为：

`Omega = ring_budget_max * (1 - exp(-mass / ring_budget_tau_norm))`。

因此预算随可见弧长 `mass` 改变，而不是严格的每个 eligible parent ring 相等。

确定性 fixture 结果：

| fixture | ring 0 | ring 1 | ring 2 | 权重 CV |
|---|---:|---:|---:|---:|
| 三个完整环 | 0.997990 | 0.999998 | 1.000000 | 0.0009 |
| 短弧 + 两个完整环 | 0.304813 | 0.999998 | 1.000000 | 0.4266 |
| 极短弧 + 两个完整环 | 0.070140 | 0.999998 | 1.000000 | 0.6352 |
| 一个环拆成两个 fragment | 0.962364 | 0.999998 | 1.000000 | 0.0180 |

短弧场景改变 `ring_budget_tau_norm`：

| tau | 短环总权重 | 完整环示例权重 |
|---:|---:|---:|
| 0.001 | 1.00000 | 1.00000 |
| 0.05 | 0.66403 | 1.00000 |
| 0.15（当前） | 0.30481 | 1.00000 |
| 1 | 0.05308 | 0.85943 |
| 10 | 0.00544 | 0.17816 |

结论：

- **“parent ring 作为预算单位”在设计上是必要的。**
- **当前 `ring_budget_tau_norm`/饱和 `Omega` 不是已验证的 equal-parent 实现**；在不等覆盖时会系统性降权短环。
- `ring_budget_max` 与 `ring_budget_tau_norm` 目前只能标为 **待修正/待重新验证**，不能作为已确定的科学参数。

### 2.5 现有真实数据结果

已有结果文件显示：

- M0/M1-RPC screen：4 棵树 global/direction/range 指标通过，但 local stationarity 为 false，决策为 `BIAS_REPAIR_REQUIRED`。
- bias audit：5,182 个 paired crops，4 棵树，median bias reduction `7.25e-12`，tree pass fraction `0.0`，决策 `M1RPC_NOT_PROVEN`。
- observability：5,182 crops 中 `POINT=5082`、`MULTIMODAL=59`、`REJECT=41`、`AXIS=0`；GT 在 profile support 中仅 `3262/5182=62.95%`。
- 因此当前 `POINT` 是尖锐的 M0 有限解，不等于正确髓心，也不等于可靠 POINT。

## 3. 按协议阶段的验证判定

| 阶段 | 必要条件/参数 | 证据 | 当前判定 |
|---|---|---|---|
| 0 数值与证据链硬门 | h/t/frame/seam 基础不变量、near barrier、搜索充分性、hash/replay、firewall | 基础不变量通过；缺 barrier、B3、完整 hash、E_audit firewall | **STOP（部分通过，不能进入完整 R0 PASS）** |
| 1 Shared S0 / equal-parent | B0、B1、B2、B3 同 crop 配对；equal-parent 不受复制/fragment 影响 | 代码只有 M0/B2；B0/B1/B3 未作为正式实验实现；短弧权重明显不等 | **STOP/REVISE** |
| 2 Preflight | S0 搜索前 candidate-free P0/P1/P2/P3 | 未找到该接口；`candidate_screen` 在搜索后运行且依赖 candidate loss | **未实现，不能判 KEEP** |
| 3 Profile/Topology 五态 | POINT/AXIS/RANGE/MULTI/REJECT fixture 和 refinement | 有 profile grid 与 MULTIMODAL；`classify` 只有 MULTIMODAL、AXIS、POINT、REJECT，无 RANGE_UNCERTAIN | **REVISE；RANGE 未验证** |
| 4 Independent Audit | `E_fit | buffer | E_audit`、firewall、null/alternative sentinel、三分输出 | 只有使用 GT 的 `bias.audit`；无 E_fit/E_audit split、firewall、HOLDOUT_FAILED/AUDIT_SUPPORTED/UNDERPOWERED | **STOP RELIABLE POINT** |
| 5 Bias/Risk route | cheap scatter、S1 diagnostic、risk 只能升、负控制 | M1-RPC 作为主定位模型；无独立 risk route/S1 语义验证；M1-RPC reduction≈0 | **REMOVE RUNTIME / RESTRICT，待重新设计** |
| 6 Real-tree shadow | 阶段 0–5 冻结后盲跑、完整分母、tree-level 分析 | 有真实数据运行，但上游硬门未闭合，且只有 4 棵树 | **不能作 GO_DEEPER；最多是 exploratory shadow** |

## 4. 参数去留结论

### 当前有证据支持“必要/保留”的

1. **显式 parent-ring order 与 fragment 分离**：manifest 检查和单元测试支持其必要性；缺失 parent association 时不能运行完整 equal-parent/audit 结论。
2. **RP² 齐次表示、h 规范化、t 符号不变、frame transform**：确定性检查和 13 个单测支持保留。
3. **M0 作为描述性 baseline**：真实数据中可运行，但它仍是未校准的探索性基线，不是可靠 POINT 管线。
4. **按 biological tree 分组**：代码测试和 manifest 支持；4 棵树规模不足以做正式泛化结论。

### 当前不能证明必要，或应暂时移出主线的

1. **`ring_budget_tau_norm` 的当前饱和形式**：直接破坏严格 equal-parent，必须改定义或限制适用域后重跑阶段 1。
2. **M1-RPC shape/order 惩罚与 lambda 网格**：现有 5,182 个 paired crops 的 reduction 约为 `7.25e-12`，未达到预注册 20% tree-level 目标；当前不应继续作为 runtime 主模型。
3. **`candidate_screen` 参数**：它不是协议定义的 Preflight，不能用其阈值证明 P0–P3 已验证。
4. **profile 的 `support_delta`、phi/kappa 网格和 POINT/AXIS 宽度**：只能作为 provisional topology 候选；没有 RANGE、refinement 漏检和 E3 校准证据，不能冻结为最终阈值。
5. **bias audit 的 radius/coarse grid**：这是 GT 附近 local-bias 工具，不是 Independent Audit；不能支持 reliable POINT。
6. **Q8 contribution 与 Q9 Multi-FOV**：协议明确是 Core 通过后的可选项，当前不阻塞，也没有必要优先验证。

## 5. 依赖顺序和下一步

依据协议，当前应停在阶段 0/1 的修正，而不是继续扩大真实数据或调 M1-RPC：

1. 先实现并验证严格 equal-parent（每个 eligible parent ring 固定总预算；复制、插值、fragment split 不改变 ring 贡献）。
2. 补齐 B0、B1、B3，在同一 synthetic crop、同一 observation/split/seed/search budget 下与 B2 配对比较。
3. 补齐阶段 0 的 near barrier、search adequacy、evidence hash/replay 和 firewall 负控。
4. 只有阶段 0 PASS 且阶段 1 KEEP 后，才验证 candidate-free Preflight。
5. 然后再补 RANGE_UNCERTAIN 和 E_fit/E_audit Independent Audit；在此之前不得声称 reliable POINT。

当前最稳妥的总体结论是：**几何基础和工程数据链有局部通过证据，但协议要求的“必要参数”尚未被逐阶段闭合验证；equal-parent 实现、Preflight、RANGE 状态和 Independent Audit 是阻塞项。**

## 6. 本轮修复后的复验（2026-09-02）

针对上述根因已做以下最小改动：

- `preprocess.py`：改为每个 eligible parent ring 固定总预算 `ring_budget_max`；弧长仅用于环内分配。旧的 `ring_budget_tau_norm` 饱和公式不再改变 parent-ring 总权重。
- `evidence.py`：新增确定性的 `E_fit | buffer | E_audit` parent-ring 分区、fit-only subset、固定 hypothesis audit 和保守 `HOLDOUT_UNDERPOWERED` 接口。
- `preflight.py`：新增不读取 candidate loss 的 P0–P3 candidate-free 特征与路由。
- `observability.py`：新增 provisional `RANGE_UNCERTAIN` 分类；最终 RANGE 阈值仍留给 E3。
- `io.py`：在 manifest ingestion 边界补齐稳定的 input/observation/lineage hash。
- 新增 `experiments/00_preflight.py` 和 `experiments/stage4_independent_audit.py`，分别用于顺序验证和小规模结构审计。
- `baselines.py`：补齐 B0（one-ring pseudocenter + RP² medoid）、B1（pooled micro-arcs）和 B3（dense/offline reference search），并新增 `stage1_s0_value.py` 配对运行入口。

复验结果：

- `python -m pytest tests/ -q`：18/18 通过。
- `python -m compileall -q oc_arcpith_rg experiments`：通过。
- 全部 5,182 条 manifest 运行 P0：5,182 条得到 `POINT_CANDIDATE`；5,180 条具有可用 buffered audit，2 条因 parent-ring 数量不足而不可审计。
- Independent Audit 结构试跑 4 条：firewall failures=0；4 条均完成锁定候选后的审计流程。
- 阶段 1 基线 smoke（1 条 crop，B3=24×12 网格）：B0/B1/B2/B3 均返回有限结果；该结果仅验证接口可运行，未作为 KEEP 判据。

### 7.1 正式快速批次的修正结果

阶段 1 现在使用 B1/B2 相同的 72×40 rapid grid；B3 使用 24×12 低成本 reference grid。32 条 sentinel 的 tree-level 结果为：

| tree | B2 vs B0 `d_RP²` 中位变化 | B2 vs B0 胜出 | B2 vs B1 `d_RP²` 中位变化 | B2 vs B3 `d_RP²` 中位变化 |
|---|---:|---:|---:|---:|
| T0 | -1.29058 | 8/8 | 0 | 0.00926 |
| T2 | -0.66419 | 8/8 | 0 | 0.02793 |
| T4 | -1.07439 | 8/8 | 0 | 0.03158 |
| T6 | -0.97507 | 8/8 | 0 | 0.00973 |

该结果支持 B2 相对 B0 的共享结构方向性收益，但不支持 B2 相对 pooled B1 已有稳定独立增量；B3 仍显示搜索/网格差距。因此阶段 1 保留 B2 候选，但不发布无条件 `KEEP S0`。

Independent Audit 32 条结构试跑：firewall failures=0，`HOLDOUT_UNDERPOWERED=32`。没有 sentinel response curve 时，不能输出 `AUDIT_SUPPORTED`。

这些复验只证明接口、分区和预算规则已按设计工作；它们**不**等价于阶段 0–6 完整通过，也不提供正式 audit power/type-I 或 calibrated POINT 结论。阶段 1 的 B0/B1/B3 正式 paired 实验、阶段 2–5 的系统化 synthetic strata 和阶段 6 的冻结 shadow 仍需单独运行。

## 7. 四棵树 exploratory 队列复验

用户确认继续使用 T0/T2/T4/T6；本轮将其明确为 exploratory/shadow，不作正式泛化或校准样本。

- `tools/select_stage1_sentinel.py` 按 tree、crop size、髓心内外和距离做不读取算法输出的 round-robin 选择。
- 生成 32 条 sentinel（每棵树 8 条），覆盖 800/1200/1600 crop 和 inside/near/mid/far outside strata。
- B0/B1/B2/B3 smoke 已在该队列上运行；由于 B3 dense reference 计算成本较高，完整 32 条运行需继续作为批处理，不在本轮冒充正式结论。
- 4 条低网格 smoke 的中位结果为：`B2 loss - B0 loss = -65.4273`，`B2 d_RP²-to-GT - B0 d_RP²-to-GT = -1.4533`。这是单树、低网格、未完成分层汇总的探索性方向信号，**不能据此判定 KEEP**。

后续新增 biological tree 或独立 full-section/FOV reference 时，只需追加 manifest 行并重新生成 tree-balanced sentinel；不得把新增 tree 与旧四树结果拼成同一版本的已校准结论。
