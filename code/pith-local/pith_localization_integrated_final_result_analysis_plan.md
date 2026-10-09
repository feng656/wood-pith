# RAC-Pith v2 分阶段结果分析、评估与结论规范

> 对应主方案：[pith_localization_integrated_final.md](./pith_localization_integrated_final.md)  
> 适用数据：UruDendro4 完整横切面图像、年轮标注和髓心坐标派生的局部重叠 crop  
> 实验数据根目录：/root/2026/dataset/UruDendro4/UruDendro4/  
> 默认输出根目录：项目根目录下的 `./outputs/`；路径统一来自 `configs/racpith_v1.json` 的 `paths`  
> 实现环境：Python 3.11；所有可执行入口仅由用户在生产 conda `py311` 中手动运行；当前几何主线固定为 NumPy/SciPy float64 CPU，GPU 未启用  
> 文档性质：结果分析与验收规范，不包含任何尚未运行的实验结果

---

## 0. 文档目标与最终边界

本规范回答的不是“程序是否跑完”，而是以下问题：

1. 数据划分、坐标、年轮 lineage 和证据预算是否正确；
2. 连续弧、凸共同中心初值和二维鲁棒 VarPro 是否按主方案实现；
3. 远场低谷、距离退化和多模态是否被充分搜索；
4. 输出 POINT、RANGE、RAY、AXIS、MULTIMODAL 或 REJECT 是否与实际证据相称；
5. POINT 的误差、尾部风险和覆盖率是否优于公平基线；
6. 不确定区是否校准，模型风险是否能识别“内部稳定但生物学上错误”的伪 POINT；
7. ring、visible arc 和 subarc 的正负贡献是否来自冻结管线的完整删除重求解；
8. 无 GT 的 cross-fitted contribution 和符号概率是否在独立树上有效；
9. 当前结果最多支持多强的结论，哪些结论仍不得宣称。

本规范固定以下边界：

- 最终坐标始终来自 **All-Arc RAC-Pith v2**；
- 贡献模块只用于解释、质检和研究分析；
- **不实现、不评估、不启用 Safe-Prune 作为最终坐标更新模块**；
- GT-oracle 单删只表示理论上限，不是可部署方法；
- 不把边界上的有限点当作图外或无穷远髓心；
- 不把低损失、低残差、高杠杆或高冲突单独解释为正负贡献；
- 不把同一树的大量高度截面和重叠 crop 当作独立样本。

### 0.1 结果分析的三层有效性

| 层级 | 核心问题 | 允许形成的结论 |
|---|---|---|
| 数值有效性 | 输入输出、坐标、预算、求积、优化和 replay 是否正确 | “实现与数值过程有效/无效” |
| 科学有效性 | 每个算法模块是否完成其唯一职责 | “模块能/不能解决相应问题” |
| 使用有效性 | 当前结果能作为有限点、范围、方向、候选集合，还是必须拒绝 | “输出可用于何种任务” |

三层不得互相替代：

- 优化器收敛不等于搜索充分；
- 搜索到有限低谷不等于有限距离可辨识；
- POINT 支持区紧致不等于径向汇聚中心就是生物学髓心；
- POINT 样本误差低不等于方法对所有 crop 有效；
- 大量重叠 crop 不等于存在大量独立树；
- 删除弧后点误差下降不等于删除后仍保留可辨识性。

### 0.2 阶段结论标签

每个阶段对每个 crop 和数据集汇总使用：

| 标签 | 含义 |
|---|---|
| PASS | 产物完整且核心门通过，可进入下游 |
| PASS_WITH_RISK | 可进入下游，但风险必须传播 |
| EXPLORATORY | 可作趋势或方法研究，尚不足以冻结或宣称校准 |
| FAIL | 本阶段核心职责未完成，依赖它的下游结果无效 |
| NOT_APPLICABLE | 当前状态不适用该阶段或指标 |

任何上游 FAIL 都必须记录在完整分母中。下游即使偶然产生数值，也只能标记为 UPSTREAM_INVALID，不能继续解释。

### 0.3 必须分开的结果维度

最终结果至少同时保存：

| 维度 | 合法取值示例 |
|---|---|
| stage_verdict | PASS / PASS_WITH_RISK / EXPLORATORY / FAIL / NOT_APPLICABLE |
| geometry_state | POINT / RANGE / RAY / AXIS / MULTIMODAL / REJECT |
| target_domain | TARGET_ALIGNED / ECCENTRIC_GT / TARGET_UNKNOWN |
| model_risk | LOW_RISK / MEDIUM_RISK / HIGH_RISK / UNKNOWN |
| production_usable | true / false |
| reason_codes | 一个或多个明确原因码 |

不再引入 X0–X5 一类额外执行等级。geometry_state 描述几何事实，model_risk 描述模型适用性，production_usable 表示在冻结操作点下是否允许使用；三者不得互相改写。

### 0.4 不编造结果和阈值

本文只规定指标、计算顺序和决策逻辑，不给出未运行的数值。

以下阈值必须作为配置项，在 train 内部校准后冻结：

- 应用误差容限 tau_app；
- POINT 支持宽度和有限—远场目标间隔阈值；
- 条件数或特征值比触发阈值；
- profile 支持集阈值 Delta_support；
- 数值、求积、坐标和 replay 容差；
- 模型风险门；
- 贡献死区 epsilon_C；
- 非劣效界 delta_noninferiority；
- 概率校准和拒答操作点。

若数据没有可靠像素—毫米标定，不得人为指定毫米阈值；应改用像素和 D_FOV 归一化量。

### 0.5 总体阶段流

~~~text
数据 manifest、树级 split 和完整分母
        ↓
S0  数据/坐标/crop/lineage 硬合同
        ↓
S1  连续弧、图像质量、sigma 与证据守恒
        ↓
S2  凸共同中心初值、切向第二初值、二维鲁棒 VarPro
        ↓
S3  J_infinity、条件距离 profile、二维兜底、六态
        ↓
S4  不确定性、target/model risk 与定位评测
        ↓
S5  exact ring/arc/subarc contribution 与 CF 校准
        ↓
S6  基线、消融、压力测试、工程性能和 sealed test
~~~

---

## 1. UruDendro4 数据合同与树级划分

### 1.1 只读数据源

生产数据目录与输出目录默认由 `configs/racpith_v1.json::paths` 一次性给出；显式命令行覆盖只允许用于临时诊断，并必须进入审计记录。预期结构为：

~~~text
/root/2026/dataset/UruDendro4/UruDendro4/
├── pith_location.txt
├── annotations/
│   └── annual_rings/
└── images_no_background/

PROJECT_ROOT/outputs/
├── prepared/
├── development/
└── sealed/
~~~

脚本不得改写、移动或重命名原始数据。所有标准化索引、split、crop、缓存和结果写入配置指定的 `./outputs/`；相对路径按项目根目录解析。

本规范不硬编码图像数量、年轮数量或坐标格式，必须以实际 manifest 审计结果为准。UruDendro4 官方数据来源见：

- [UruDendro4 数据集（Zenodo）](https://doi.org/10.5281/zenodo.15653340)

### 1.2 树 ID 是最小划分单位

UruDendro4 的独立树标识固定定义为：

\[
\boxed{
\texttt{tree\_id}
=
\texttt{T\{treatment\}\_B\{block\}\_N\{tree\}}
}
\]

文件名中的高度、圆盘、截面或其他后缀属于 section_id，不属于 tree_id。

解析规则必须满足：

1. 将空格、连字符和下划线等允许分隔符按预注册规则标准化；
2. 从样本主名解析 treatment、block、tree 三个字段；
3. 生成规范形式 T{treatment}_B{block}_N{tree}；
4. 完整原始样本名保留为 section_id；
5. 解析失败、字段冲突或同一 section 对应多个 tree_id 时立即 FAIL；
6. 不得退化为按图片随机划分。

同一 tree_id 下的以下数据必须位于同一顶层 split：

- 不同高度的所有完整截面；
- 同一截面的所有图像版本；
- 同一截面的所有年轮标注；
- 所有规则或随机生成的 crop；
- 所有重叠 crop；
- 所有增强、扰动和复现实验副本。

### 1.3 顶层 train/sealed_test 与 train 内部校准

数据核对得到 24 个独立 tree_id。默认正式划分固定为：

~~~text
18 trees -> train
 6 trees -> sealed_test
~~~

职责如下：

| 集合 | 树数 | 允许用途 |
|---|---:|---|
| train | 18 | 开发、超参数冻结、tree-grouped 内部交叉验证、支持阈值校准、贡献概率模型训练 |
| sealed_test | 6 | 冻结后一次性端到端评价 |

train 内部使用 tree_id 分组的交叉验证完成开发和校准，不再从 6 棵 sealed_test 树中抽取验证样本。所有 fold 的同树约束与顶层 split 相同。

若研究协议必须显式区分 development、calibration 和 sealed_test，可选：

~~~text
16 trees -> development/train
 4 trees -> calibration
 4 trees -> sealed_test
~~~

但 4 棵 calibration 树和 4 棵 sealed_test 树的独立样本量都很小，尾部分位数、coverage、概率校准、False-POINT 上置信界和贡献分类性能的区间会很宽。该三分法只能形成低功效或探索性结论，不应因 crop 数量巨大而宣称校准充分。除非研究协议必须使用显式三分法，否则默认采用 18/6，并在 18 棵 train 树内做 grouped CV。

默认方案中的 6 棵 sealed_test 树仍然是有限的独立样本。P90/P95、coverage 和稀有失败率必须同时列出逐树结果和宽区间；不能因为每棵树产生大量 crop 就缩窄树级置信区间，或声称已经获得高精度总体尾部估计。

具体 tree 名单、随机种子和是否按 treatment/block 分层必须来自冻结配置。不得为了让某次结果更好而重复尝试 split 后选择最优划分。划分脚本至少输出：

~~~text
tree_split.csv
section_split.csv
split_summary.json
split_leakage_report.json
~~~

其中 split_leakage_report 必须验证：

~~~text
train_tree_ids ∩ sealed_test_tree_ids = ∅
同一 section 不跨 split
同一 crop lineage 不跨 split
每个输出样本均可回溯到唯一原始图像
~~~

### 1.4 配对原图、标注和髓心

数据索引必须建立一一对应关系：

~~~text
section_id
↔ images_no_background 中的 RGB 图像
↔ annotations/annual_rings 中的年轮标注
↔ pith_location.txt 中的髓心坐标
~~~

不得仅按目录顺序配对。必须按标准化 section_id join，并报告：

- 缺图；
- 缺标注；
- 缺髓心；
- 重复 key；
- 多对一或一对多冲突；
- 图像尺寸与标注坐标范围冲突；
- LabelMe 的空白 `imageWidth/imageHeight` 可按“未提供元数据”处理，但非空尺寸声明必须与实际解码图像严格一致；
- LabelMe 的空白 `imagePath` 可由严格的文件 stem join 补足，但任何非空声明必须匹配 section ID；
- 无法解析的 annotation schema。

任何冲突都保留在原始完整分母中，并标记 INVALID_SOURCE_RECORD。

### 1.5 crop 生成合同

从完整截面生成大量重叠 crop 时，每个 crop 至少保存：

| 字段 | 含义 |
|---|---|
| tree_id / section_id / crop_id | 分组、来源和唯一 ID |
| split / inner_fold | 顶层集合和 train 内折 |
| source_image_path / annotation_path | 只读来源 |
| crop_box_xyxy | 原图坐标中的 crop 矩形 |
| source_to_crop / crop_to_source | 双向坐标变换 |
| source_shape / crop_shape | 图像尺寸 |
| pith_source_xy / pith_crop_xy | 不裁剪的 GT 坐标 |
| pith_inside_crop | GT 是否在 crop 内 |
| parent_ring_id | 完整截面中的年轮身份 |
| ring_order | 若原始标注可可靠恢复 |
| visible_arc_id / fragment_id | crop 后的可见连通段 |
| source_arc_parameter | 回到原始年轮的弧长参数 |
| image_scale | 像素—毫米标定及来源；未知时为空 |
| generator_config_hash / seed | crop 生成可复现性 |
| crop_image_sha256 / crop_annotation_sha256 | 已物化 crop 图像与 GT-free 标注的逐文件内容哈希 |

裁切必须使用年轮边界曲线与矩形求交。若原始标注是闭合 polygon，不得把 polygon 与 crop 相交后生成的 ROI 矩形边界误当成年轮。

`crop_manifest.jsonl` 的每条已选 crop 必须登记实际 crop PNG 与 `racpith.crop_annotation.v1` 的 SHA-256；`full_section_reference_manifest.jsonl` 用同名字段分别绑定只读源图和新生成的 GT-free reference annotation，并将 `analysis_role` 固定为 `TARGET_REFERENCE_ONLY`。evidence 构建在读取前重新计算两者，路径存在但内容哈希不符时立即失败。`crop_candidate_manifest.jsonl` 包含未物化候选，因此不能要求每条候选都具有产物内容哈希，也不能把 manifest 文件自身的 SHA-256 当作逐 crop 内容校验的替代。

髓心在 crop 外时，pith_crop_xy 可以为负数或大于图像宽高，必须原样保留。

### 1.6 crop 难度描述

每个 crop 记录原始、非结果选择式难度量：

\[
d_{\mathrm{out}}
=
\frac{d(c^*,\Omega)}{D_{\mathrm{FOV}}},
\]

其中髓心在 crop 内时 \(d(c^*,\Omega)=0\)。另保存：

- pith_inside_crop；
- 从 crop 锚点到 GT 的距离除以 D_FOV；
- parent-ring 数；
- visible arc 和 fragment 数；
- 总有效弧长；
- 总转角；
- 单侧性和空间覆盖；
- crop 尺寸与分辨率；
- 自然裂纹、节疤、污渍、模糊等质量描述；
- S1/S2 后才可获得的 H0 特征值比和切向方向张开度。

分层边界应由配置给出或用 train 分位数冻结。sealed_test 不得重新定义“容易/困难”边界。

### 1.7 完整分母

结果分析从唯一 manifest 出发：

\[
N_{\mathrm{denominator}}=N_{\mathrm{crop,manifest}}.
\]

必须同时报告：

~~~text
N_tree
N_section
N_crop_manifest
N_started
N_finished
N_missing_result
N_invalid_source
N_stage0_pass ... N_stage6_pass
N_POINT / N_RANGE / N_RAY / N_AXIS / N_MULTIMODAL / N_REJECT
N_solver_failure
N_search_inadequate
N_model_risk_high
N_production_usable
~~~

贡献分析另有组级完整分母：

~~~text
N_ring_groups_total / evaluable / state_only / unresolved
N_arc_groups_total / evaluable / state_only / unresolved
N_subarc_groups_total / evaluable / state_only / unresolved
N_CF_evaluable / N_CF_abstain
~~~

缺失产物、程序异常、NaN、反序列化失败、上游无效和不适用必须分类计数，不能只分析成功样本。

---

## 2. 统一结果索引、产物目录与运行记录

### 2.1 result_index

每个 crop 一条主索引：

| 字段 | 说明 |
|---|---|
| tree_id / section_id / crop_id | 唯一 lineage |
| run_id | 一次冻结运行 |
| split / inner_fold | 数据职责 |
| data_manifest_hash | 输入 manifest 版本 |
| crop_config_hash | crop 生成版本 |
| estimator_config_hash | 定位器冻结配置 |
| contribution_config_hash | 贡献冻结配置 |
| code_commit | 代码版本 |
| python_version / dependency_lock_hash | 环境版本 |
| device | cpu 或 cuda:0 |
| stage0_status ... stage6_status | 阶段结论 |
| first_failure_stage | 第一个失败阶段 |
| geometry_state | 六态之一 |
| target_domain / model_risk | 目标域和模型风险 |
| production_usable | 冻结规则下是否允许使用 |
| reason_codes | 多值原因 |
| artifact_paths | 各阶段产物路径 |

### 2.2 单 crop 主记录

~~~text
CropAnalysisRecord
├── provenance_and_split
├── source_crop_transform
├── evidence_and_lineage
├── image_annotation_quality
├── c0_H0_and_ct
├── finite_varpro_solution
├── far_scan_and_profiles
├── search_adequacy
├── geometry_state_and_support
├── uncertainty
├── target_and_model_risk
├── localization_metrics
├── contribution_summary
├── runtime_and_memory
├── stage_verdicts
├── first_failure_stage
└── conclusion_text
~~~

### 2.3 贡献记录

每个删除组一条记录：

| 字段 | 说明 |
|---|---|
| level | ring / visible_arc / subarc |
| group_id / parent_ring_id / arc_id / subarc_id | 层级身份 |
| source_arc_interval | 原弧物理或归一化位置 |
| scale / phase | 子弧尺度与切分相位 |
| removed_budget_mass | 被删除的冻结证据质量 |
| full_state / deleted_state | 删除前后状态 |
| full_search_status / deleted_search_status | 删除前后搜索充分性 |
| contrib_gt_px/mm/norm | POINT–POINT 时的 GT 贡献 |
| contrib_gt_sq / contrib_gt_rel | 平方和相对贡献 |
| contrib_label / contribution_interval | 符号与区间 |
| directional_info_gain | 非负方向信息 |
| loo_shift_vector / loo_shift | 中心位移 |
| conflict_cost | 与其余证据冲突 |
| direction_change / range_change | 支持变化 |
| state_change / new_modes | 可辨识性变化 |
| contrib_cf | 无 GT 交叉拟合贡献 |
| p_positive / p_neutral / p_negative | 冻结校准器输出 |
| roles / abstain_reason | 多角色或拒答原因 |

### 2.4 推荐产物层次

~~~text
outputs/<run_id>/
├── manifests/
├── audits/
│   ├── s0_data_contract/
│   ├── s1_evidence/
│   ├── s2_finite_estimator/
│   ├── s3_far_state/
│   ├── s4_uncertainty_risk/
│   ├── s5_contribution/
│   └── s6_benchmark/
├── records/
├── tables/
├── figures/
├── case_cards/
├── logs/
└── run_summary.json
~~~

分析脚本必须从 records 和 manifest 派生表图，不能解析 PNG 或自由文本日志来恢复关键数值。

本实现进一步采用“索引为权威、目录为审计对象”的规则：定位、evidence、不确定性、贡献和基线分别由对应 index 注册；正式评估必须核对 schema、配置哈希、tree/section/split lineage、内容 SHA-256 和清单分母。目录中存在未注册或哈希已变化的旧产物时应 fail closed，不能按文件名扫描后混入统计。

派生分析与报告同样必须有内容索引：

- `denominators.json` 登记定位统计输入与输出哈希；
- `runtime_denominator.json` 绑定实际 stage indexes，并登记 `stage_crop_runtime.csv`、`stage_tree_runtime.csv`、`runtime_summary.csv`、`localization_search_diagnostics.csv`、`search_trigger_summary.csv` 的 SHA-256；
- `contribution_evaluation_index.json` 绑定 config、crop manifest、正式 contribution index、可选 partition-audit index、唯一 source result-index SHA，并登记所有贡献统计表或空结果状态文件的哈希；
- `contribution_visualization_index.json` 绑定同一贡献 lineage，并登记 cohort 图与 overlay 图的哈希；
- `RAC_Pith_*_report.md.index.json` 登记 Markdown 报告自身 SHA-256，以及报告实际使用的 localization/runtime/contribution/visualization/baseline index 或 denominator 哈希。

报告 Markdown 与其 `.md.index.json` sidecar 是同一个封存单元；sidecar 缺失、输入索引缺失或任何登记哈希漂移时，不得据此形成正式结论。

封存报告生成还设三重前置门：主定位/正式贡献、独立分区敏感性贡献、完整截面目标域必须分别得到 `racpith.combined_audit.v1` 的 `PASS`。报告脚本拒绝重复审计、外层或内层 artifact audit 非 PASS、畸形 finding，以及名义 PASS 中仍含 ERROR 的情况；并要求三份审计中恰有一份含非空 `target_domain_records`，防止用普通 PASS 审计替代完整截面目标域门。三份审计文件的 SHA-256 和报告自身 SHA-256 一并写入 sidecar。审计与可视化阶段同时拒绝 crop 内容漂移、未注册或哈希漂移的不确定性 sidecar、minus JSON 和 PNG。

### 2.5 环境与可复现记录

正式运行记录：

- Python 3.11 完整补丁版本；
- py311 环境的包版本；
- NumPy/SciPy 版本；若未来经批准启用 GPU，再记录 PyTorch 版本；
- CPU、GPU、CUDA 和驱动信息；
- 随机种子；
- 线程数；
- 浮点类型；
- 配置文件和 Git commit；
- 开始、结束时间；
- 当前设备必须记录为 CPU；若未来显式启用 cuda:0，再记录该选择；
- 若未来启用 GPU，记录 CPU/GPU 数值一致性 replay。

当前实现不启用 GPU，不能因检测到 CUDA 或安装了 PyTorch 而自动切换。核心二维几何固定使用 NumPy/SciPy float64 CPU，crop 级并行由 worker 多进程承担。只有后续确认批量图像特征、profile 或 bootstrap 是瓶颈，并在生产 `py311` 中证明 CPU/cuda:0 的中心、目标/profile、状态、支持区和贡献标签在预注册容差内一致，才允许增加显式 opt-in 的 GPU0 加速；GPU 不得形成不同的目标函数、精度默认值或状态分支。

---

## 3. 通用统计原则

### 3.1 独立统计单位

首选独立单位是 tree_id，即 T{treatment}_B{block}_N{tree}。同树不同高度、同截面的所有重叠 crop 都是簇内相关观测。

如果某项分析只覆盖少数 tree，必须报告 tree 数和 crop 数，不能只写 crop 数。

### 3.2 树级聚合

建议顺序：

1. 每个 crop 计算原始指标；
2. 每个 section 内按预注册规则聚合；
3. 每个 tree 内聚合 section 结果；
4. 方法比较形成每棵树的 paired difference；
5. 在 tree 层做 cluster bootstrap、配对 permutation 或非参数区间；
6. 同时报告效果量、置信区间和完整分母。

对于失败率、状态比例等二元/多项结果，bootstrap 时抽 tree 并携带该树全部 section/crop。

默认的 tree-balanced 描述统计使用分层等权：

1. 每棵树总权重相同；
2. 同一树内每个 section 总权重相同；
3. 同一 section 内每个预注册 crop 总权重相同；
4. nested、增强或扰动副本不得额外增加原 crop 的主分析权重。

主表中的 tree-balanced quantile 应使用上述层级权重计算，并另附逐树 median/P90/状态率。方法的配对推断以“每树一个预注册汇总值”为输入；不能先将所有 crop 混合后再做普通 bootstrap。

### 3.3 配对比较

所有基线和消融必须使用同一 crop manifest。比较时禁止：

- 每种方法删除不同失败样本；
- 只在两者都成功的子集上报告主结果而不报告失败；
- 把一个方法的 REJECT 从分母移除；
- 使用 sealed_test GT 为某方法单独调阈值；
- 将重叠 crop 当作独立观测计算极小 p 值。

推荐同时报告：

- 完整分母结果；
- common-valid 子集的配对诊断；
- 失败原因差异；
- tree-level 配对效果量和区间。

common-valid 子集只能帮助解释，不能替代完整分母主结论。

### 3.4 GT 的合法用途

GT 可以用于：

- train 内部阈值和校准；
- sealed_test 的一次性误差与支持覆盖评价；
- exact GT contribution；
- GT-oracle 单删上限；
- target-domain 的事后分层；
- synthetic/controlled fixture 的状态真值。

GT 不得用于：

- 在 sealed_test crop 上选择 c0、ct、某个有限 basin 或远场方向；
- 在多个模态中挑选最接近 GT 的模态并改报 POINT；
- 决定是否触发 profile 或二维审计；
- 调整当前样本的 sigma、Huber 参数或搜索范围；
- 选择要删除的弧并改变最终 All-Arc 坐标；
- 从 sealed_test 结果反向调整 epsilon_C 或状态阈值。

### 3.5 误差单位

有限 POINT 的基本误差：

\[
e_{\mathrm{px}}=\|\hat c-c^*\|_{\mathrm{px}},
\]

\[
e_{\mathrm{norm}}=
\frac{\|\hat c-c^*\|}{D_{\mathrm{FOV}}},
\]

有可靠标定时：

\[
e_{\mathrm{mm}}=\|\hat c-c^*\|_{\mathrm{mm}}.
\]

三者必须标明单位。无物理标定时，e_mm 为缺失值，而不是用默认 DPI 或图像尺寸推断。

### 3.6 联合主终点

选择性输出方法不能只用 POINT 子集平均误差评价。建议把以下三项作为联合主终点：

1. 冻结操作点下 POINT 的 tree-balanced P90 误差；
2. False-POINT 风险及其 tree-cluster 95% 上置信界；
3. POINT coverage 与 operational usable coverage。

定义：

\[
\mathrm{FalsePOINT}
=
\mathbf 1[
\mathrm{state}=\mathrm{POINT}
\land e>\tau_{\mathrm{app}}
].
\]

必须同时报告：

- emitted-POINT risk：False-POINT / N_POINT；
- population incidence：False-POINT / N_manifest；
- POINT coverage：N_POINT / N_manifest；
- structured-output coverage：N_non-REJECT / N_manifest；
- operational usable coverage：按预注册应用范围计入 POINT、RANGE、RAY、AXIS；MULTIMODAL 单列。

若应用只接受有限坐标，则 production coverage 只计 production_usable POINT。

### 3.7 次要定位终点

- POINT median、IQR、P95、max 和 catastrophic rate；
- 原始 pith_raw 误差，仅作诊断；
- RANGE 的方向误差、GT 区间覆盖和 log-range 宽度；
- RAY 的有向角误差和距离下界违反率；
- AXIS 的模 \(\pi\) 角误差；
- MULTIMODAL 的 GT-consistent component 和 mode persistence；
- 支持区 coverage、面积或宽度；
- solver failure、search inadequate、boundary-pinned 和多初值分歧率；
- aligned-domain 与 full-domain 的所有主终点；
- risk–coverage 曲线及 AURC。

### 3.8 方向指标

以 crop 锚点 o 为基准：

\[
u^*=\frac{c^*-o}{\|c^*-o\|}.
\]

RAY 的有向误差：

\[
e_{\mathrm{ray}}
=
\arccos\!\left(
\operatorname{clip}(\hat u^T u^*,-1,1)
\right).
\]

AXIS 的无向误差：

\[
e_{\mathrm{axis}}
=
\arccos\!\left(
|\hat u^T u^*|
\right).
\]

GT 距离过近使方向不稳定时，必须标记 direction_not_applicable，不能强算角度。

### 3.9 结构化支持区的 GT 覆盖定义

POINT、RANGE、RAY、AXIS 的覆盖必须以定位器同一个 profiled objective 定义。设结果保存的全局参考值为

\[
J_{\mathrm{ref}}=\min(J_{\mathrm{finite}},J_{\infty}),
\]

冻结的支持阈值为 \(\Delta_{\mathrm{support}}\)，则对 GT 只做后验固定点评分：

\[
J_{\mathrm{GT}}
=
\min_{\{R_r\}}
J(c=c^*,\{R_r\}),
\qquad
\mathrm{contains}_{\mathrm{support}}
=
\mathbf 1[J_{\mathrm{GT}}\le J_{\mathrm{ref}}+\Delta_{\mathrm{support}}].
\]

这里允许重新 profile nuisance radii，但严禁优化中心、用 GT 选择模态、改变状态或回写估计器。正式评分前还必须在保存的 `raw_center_norm` 处重算目标，并与结果中的 objective 在冻结数值容差内一致，否则该 crop 的评估链路 FAIL。

`radial_interval_contains_gt` 只检查 \(\|c^*-o\|\) 是否落在报告的径向区间，是便于解释的附加诊断。它会丢失方向、模态和支持拓扑信息，特别是 RAY/AXIS 下不能作为正式 coverage。兼容旧表名时，`range_contains_gt` 必须明确指向上述完整 objective-support 命中，而不是径向命中。

---

## 4. S0：数据、坐标、crop 和 lineage 硬合同

### 4.1 最小产物

~~~text
source_manifest
tree_split / section_split
crop_manifest
pairing_report
coordinate_transform_records
lineage_table
ring_order_table
source_crop_overlays
hard_invariant_report
result_index skeleton
~~~

### 4.2 检查顺序

#### A. 数据配对

检查图像、年轮 JSON 和 pith_location.txt 是否按 section_id 唯一 join。目录顺序匹配一律视为实现错误。

#### B. tree_id 与 split

验证 T{treatment}_B{block}_N{tree} 解析、同树所有高度归组，以及 train/sealed_test 与 train 内 fold 无 tree 泄漏。

#### C. 坐标语义

必须通过可视 overlay 和数值 round-trip 同时确认：

- 髓心文件是 x,y 还是 row,column；
- 坐标原点；
- 是否为零基；
- 图像旋转或翻转；
- JSON 点与 RGB 图像尺寸；
- source→crop→source 往返；
- 图外 GT 未 clip。

往返误差：

\[
e_{\mathrm{rt}}
=
\|T^{-1}(T(x))-x\|.
\]

#### D. 变换等变性 fixture

对平移、旋转、统一缩放和坐标归一化重放，映回原坐标后检查：

\[
e_{\mathrm{equiv}}
=
\|\hat c-T^{-1}(\hat c_T)\|.
\]

方向、范围、状态也必须一致。

#### E. 年轮 lineage

检查：

- parent_ring_id 唯一；
- 每个 fragment 只属于一个 parent ring；
- source arc parameter 可逆追踪；
- ring order 来自标注拓扑，而非当前候选；
- crop 边界不进入年轮；
- 真缺口没有被样条连接；
- 插值点未改变 parent-ring 数。

### 4.3 主要指标

| 指标 | 层级 | 含义 |
|---|---|---|
| pairing failure count | source | 数据配对完整性 |
| tree leakage count | split | 独立测试是否合法 |
| transform round-trip max/median | crop | 坐标实现正确性 |
| overlay failure count | source/crop | 坐标语义错误 |
| clipped outside-GT count | crop | 是否篡改图外真值 |
| lineage conflict count | crop | 贡献可追溯性 |
| crop-edge false ring count | crop | polygon 裁切错误 |
| equivariance failure count | fixture | 实现是否破坏几何等变性 |

### 4.4 S0 门

- 任一 split 泄漏：整个正式实验 FAIL，必须重新划分和重跑；
- 坐标顺序、翻转或索引基错误：相关数据全部 FAIL；
- lineage 或 crop-edge 错误：定位和贡献均无效；
- 个别源文件缺失：样本标记 INVALID_SOURCE_RECORD，仍计入完整分母；
- 所有硬合同通过：S0 PASS。

### 4.5 阶段结论模板

> S0 对原始图像、年轮标注和髓心记录完成唯一配对，并按 T{treatment}_B{block}_N{tree} 实现树级划分。同一树的不同高度与所有重叠 crop 未跨 split。坐标往返、overlay、图外 GT 保持及 lineage 硬合同为“[通过/失败]”；因此后续结果“[可/不可]”解释为算法误差。

---

## 5. S1：连续弧、图像质量、测量尺度与证据守恒

### 5.1 最小产物

~~~text
continuous_curve_records
fragment_validity
endpoint_guard
quadrature_nodes_and_weights
smoothing_replay
quadrature_replay
image_annotation_quality
sigma_x_records
sigma_alg_records
evidence_budget_records
resampling_split_invariance
evidence_overlays
~~~

### 5.2 连续弧视觉审计

代表性样本和全部失败样本至少检查：

1. 原始标注点与 spline 是否一致；
2. 是否跨越真实缺口；
3. 是否自交、回折或错接；
4. 端点是否出现导数摆动；
5. 高转角部分是否被过度平滑；
6. 低转角短弧是否仅因“像直线”被错误删除；
7. valid、guarded、invalid 及原因是否正确着色。

### 5.3 平滑尺度评价

平滑尺度只能在 train 内冻结。比较相邻预注册尺度时报告：

- 曲线位置重放差异；
- 切向模 \(\pi\) 差异；
- 总转角偏差；
- c0、最终中心和 geometry_state 漂移；
- 端点保护长度；
- 真实缺口保持率。

不得按单 crop GT 选择最有利平滑尺度。

### 5.4 测量尺度

必须区分：

- sigma_x：几何距离残差尺度，单位为归一化长度；
- sigma_alg：凸代数残差尺度，单位为归一化长度平方；
- sigma_perp：切向第二初值的投影尺度。

检查：

- 单位传播；
- 物理或经验下限；
- clipping 比例；
- 图像质量映射是否仅依赖候选无关特征；
- 是否根据当前候选 residual 反向放大 sigma；
- 同一质量信号是否被重复计入多个权重通道。

若没有重复标注，sigma 只能标为开发尺度或稳定尺度，不能称为概率校准不确定度。

### 5.5 图像—标注质量

候选无关质量至少包括：

- boundary_support；
- orientation_agreement；
- texture_anomaly；
- polarity_consistency；
- 可选 ray_support。

评价方式：

- 与人工抽检或重复标注差异的关系；
- 按裂纹、节疤、锯痕、模糊分层；
- geometry-only 与加入质量尺度后的配对消融；
- 是否改善 P90/P95、False-POINT 或模型风险；
- 是否仅降低 coverage 而没有风险收益。

若独立验证无收益，图像质量保留为诊断字段，不进入 sigma 或拟合权重。

### 5.6 parent-ring 证据守恒

构造以下 deterministic replay：

~~~text
同一折线顶点复制
插值加密
等弧长采样密度改变
同一 visible arc 人工拆成多个 fragment
fragment 顺序改变
子弧边界轻微平移
~~~

必须比较：

- parent-ring 总预算；
- c0 和 H0；
- J_geo 与最终中心；
- J_infinity；
- profile、support components 和 state；
- ring/arc/subarc deletion 的 removed_budget_mass。

删除某组 G 后，原始 L_r^0 和证据密度冻结，被删质量不能重新分给相邻段。

### 5.7 quadrature 收敛

对节点数 K 和 2K：

\[
e_{\mathrm{quad}}
=
\frac{|A^{2K}-A^K|}{1+|A^{2K}|}.
\]

至少对以下量检查：

- ring 均值与 mean squared norm；
- c0；
- J_geo；
- VarPro 中心；
- J_infinity 最低值和方向；
- profile 支持端点；
- state。

### 5.8 S1 门

- 样条跨缺口、自交、错误回折或 crop 边进入年轮：FAIL；
- 点密度或 fragment 数改变证据权重：FAIL；
- sigma 候选依赖或单位混用：FAIL；
- quadrature 未收敛且改变状态：FAIL；
- 主要几何正确但 sigma 缺少重复标注校准：PASS_WITH_RISK；
- 连续弧、预算和求积均稳定：PASS。

---

## 6. S2：凸共同中心初值与二维鲁棒 VarPro

### 6.1 最小产物

~~~text
c0_solution_and_irls_log
H0_eigenvalues_and_vectors
ct_solution_and_tangent_log
varpro_inner_radius_log
varpro_outer_optimizer_log
candidate_table
distinct_finite_minima
per_ring_radius_and_loss
solver_reason_codes
finite_solution_overlay
initializer_ablation
~~~

### 6.2 凸恒等式 fixture

对理想同心圆、不同半径、多 fragment 和任意刚性变换 fixture 验证：

\[
2(x-\bar x_r)^Tc
=
\|x\|^2-\overline{\|x\|^2}_r.
\]

检查：

- 精确圆上的代数残差；
- 平移、旋转、缩放后的等价；
- ring 半径变化不改变公共中心；
- 重采样不改变解；
- H0 弱特征方向与预期退化方向一致。

### 6.3 c0 数值评价

每个 crop 记录：

- IRLS 迭代数和终止原因；
- 目标单调性；
- 最终梯度/正规方程残差；
- H0 的 lambda_weak/lambda_strong；
- 弱特征向量；
- c0 是否有限；
- c0 到 crop 的归一化距离；
- solver failure。

H0 条件比用于触发后续审计和分层，不能单独决定 POINT/AXIS。

### 6.4 ct 的职责

ct 只作为：

- 第二初值；
- 切向实现检查；
- c0 与局部方向证据的一致性诊断；
- profile/二维审计触发信号。

报告 c0–ct 距离、方向差和各自经相同 J_geo 重评分后的目标。不得按 GT 在二者间选择。

### 6.5 VarPro 内外层

给定中心时每 ring 半径求解和二维中心优化分别检查：

- 内层是否存在足够剩余证据；
- 内层凸求解是否收敛；
- 外层目标是否下降；
- 终止原因；
- 梯度和步长；
- 半径是否为有限合法值；
- 不同启动是否到达同一谷或明确分离谷；
- 重复运行漂移；
- 数值边界是否影响结果。

合法失败码至少包括：

~~~text
INSUFFICIENT_RING_EVIDENCE
INNER_RADIUS_UNSTABLE
OUTER_OPTIMIZER_UNSTABLE
NONFINITE_OBJECTIVE
SOLVER_UNSTABLE
~~~

失败后不得沿用上一次半径或中心。

### 6.6 S2 科学评价

核心比较：

| 对照 | 回答的问题 |
|---|---|
| 切向初值 vs c0 | 凸初值是否更稳、更少依赖导数 |
| c0 alone vs c0 + VarPro | 几何精化是否真正改善 |
| L2 VarPro vs pseudo-Huber VarPro | 稳健核是否改善尾部与异常弧 |
| 单初值 vs c0+ct | 第二初值是否发现分离有限谷 |
| 粗求积 vs收敛求积 | 结果是否被离散化控制 |

比较先看数值成功率、目标和搜索覆盖，再看 GT 误差。

### 6.7 S2 门

- 凸恒等式或等变性 fixture 失败：FAIL；
- 内外层未收敛却输出正常坐标：FAIL；
- 不同启动出现分离谷但只保留一个：FAIL；
- c0/ct 分歧已被送入 S3 审计：允许 PASS_WITH_RISK；
- 数值稳定并完整保留相关有限谷：PASS。

---

## 7. S3：远场扫描、条件距离 profile、搜索充分性和六态

### 7.1 最小产物

~~~text
j_infinity_scan
j_infinity_refined_valleys
finite_far_gap
far_scan_resolution_replay
profile_trigger_trace
weak_direction_profiles
competitive_direction_profiles
compact_2d_audit
support_components
state_decision_trace
search_adequacy_report
profile_and_state_figures
~~~

### 7.2 每个 crop 的强制 J_infinity

每个 crop 均须计算：

\[
J_\infty(u)
=
\sum_r\int
\rho_H\!\left(
\frac{u^Tx-m_r(u)}{\sigma_{x,r}(x)}
\right)d\mu_r(x),
\]

其中 \(u=(\cos\phi,\sin\phi)\)，\(\phi\in[0,\pi)\)。

检查：

- phi 与 phi+pi 对称 fixture；
- 角度网格和加密网格的最低方向；
- 低谷数量和持久性；
- 最低 J_infinity；
- 最佳有限解与远场最低值的目标间隔；
- 有限 J_geo(Ru) 随 R 增大是否趋近解析极限；
- 是否存在与有限谷等价但方向分离的远场谷。

任何拟输出 POINT 的 crop 都不能跳过本步骤。

### 7.3 profile 触发

以下任一出现时触发：

- H0 病态；
- c0 与 ct 明显分歧；
- VarPro 解远离 crop；
- 多初值结果分歧；
- 有限—远场间隔不足；
- 解接近宽松物理边界；
- J_infinity 出现竞争方向；
- bootstrap 或删除重求解发现新长尾/新模态。

分析报告必须列出每个触发原因，禁止只保存 triggered=true。

### 7.4 距离 profile

沿弱方向或竞争方向使用紧致坐标：

\[
z=\frac{s}{1+|s|}\in(-1,1).
\]

检查：

- profile 的有限局部谷；
- z=±1 解析远场端点；
- 支持区连通分量；
- 支持区是否触及一个或两个端点；
- 强方向局部优化区间是否有限；
- 网格加密后端点、宽度和分量是否稳定；
- 竞争方向是否全部被覆盖。

一维 profile 无法覆盖所有竞争方向时，必须进入小型二维紧致方向—逆距离审计。

### 7.5 高预算 reference-search 子集

在 train/audit 子集建立高预算、同一 J_geo 的 reference：

- 覆盖图内、近图外、中远、极远；
- 覆盖低/高 H0 条件比；
- 覆盖 c0–ct 一致与分歧；
- 覆盖单侧和多方位年轮；
- 覆盖自然异常和人工扰动；
- 覆盖疑似多模态。

主要指标：

- 最佳有限/远场低谷 recall；
- relevant component recall；
- state agreement；
- boundary false finite；
- 角度与距离分辨率加倍后的稳定性；
- Core/Profile 与二维高预算审计的差异；
- 运行时间。

该结果只能称搜索充分性评价，不能称数学全局最优证书。

### 7.6 六态决策优先级

~~~text
1. 数据、数值或搜索不充分
   -> REJECT

2. 存在两个以上分离且持久的支持分量
   -> MULTIMODAL

3. 单一分量同时触及两个远场端点
   -> 轴稳定则 AXIS，否则 REJECT

4. 单一分量触及一个远场端点
   -> 有可靠 ring-order/极性定向则 RAY，否则 AXIS

5. 不触及远场，有限距离区间闭合但过宽
   -> RANGE

6. 单一、紧致、有限且通过搜索与 replay 门
   -> POINT
~~~

有限分量与分离远场分量并存时必须是 MULTIMODAL，不能优先选择有限点。

### 7.7 S3 指标

| 指标 | 含义 |
|---|---|
| far-scan resolution agreement | J_infinity 数值稳定性 |
| finite–far gap stability | POINT 与远场竞争关系 |
| profile trigger rate | Core 中退化样本比例 |
| compact 2-D audit rate | 一维覆盖不足比例 |
| reference best/component recall | 搜索覆盖能力 |
| boundary-pinned false finite rate | 是否伪造有限点 |
| state replay agreement | 状态是否依赖网格 |
| search inadequate rate | 诚实拒绝比例 |

### 7.8 S3 门

- 解析远场 fixture 不收敛：FAIL；
- 角度/距离加密产生新相关谷且原结果未降级：FAIL；
- 边界点被当作 POINT：FAIL；
- 搜索不足被写成 RANGE/AXIS：FAIL；
- profile/二维审计覆盖相关分量且状态稳定：PASS；
- reference 子集规模不足但逻辑正确：EXPLORATORY 或 PASS_WITH_RISK。

---

## 8. S4：不确定性、target/model risk 与定位评测

### 8.1 最小产物

~~~text
measurement_replay
moving_block_bootstrap
ring_bootstrap_or_ring_loo
support_regions
uncertainty_calibration
full_section_radial_center
target_alignment_registry
structured_residual_diagnostics
model_risk_records
risk_coverage_records
localization_metrics
case_cards
~~~

### 8.2 三类不确定性分开

#### 测量不确定性

- 重复标注差异；
- 连续移动块曲线扰动；
- spline 和 endpoint replay；
- 图像质量导致的测量尺度变化。

不得做独立点 bootstrap。

#### 几何可辨识性

- H0 条件比；
- 有限—远场间隔；
- profile 宽度和端点；
- support components；
- 多初值和搜索 replay；
- ring LOO 状态散布。

#### 模型风险

- 完整截面径向汇聚中心与 GT 的差；
- ring 残差的共同低频方向趋势；
- ring LOO 的中心和状态散布；
- ring-order 系统违反；
- 偏心生长、节疤、裂纹、反应木等图像证据。

三类结果不可压成单一 confidence。

### 8.3 状态化不确定性

| 状态 | 必须输出 |
|---|---|
| POINT | bootstrap 样本、二维支持区/椭圆、长短轴、coverage 级别 |
| RANGE | 方向区间、有限距离区间、log-range 宽度 |
| RAY | 有向方向区间、距离下界、无有限上界标记 |
| AXIS | 无向轴区间、距离下界或远场接触 |
| MULTIMODAL | 每个持久模态及其支持区 |
| REJECT | first failure 和修复建议；不输出伪不确定区 |

bootstrap 若发现 profile 未覆盖的新长尾或新模态，必须回到 S3 扩大审计并重新判定最终状态。

### 8.4 target domain

在 `analysis_role=TARGET_REFERENCE_ONLY` 的完整截面上用冻结的全截面参考过程估计径向汇聚中心：

\[
B_{\mathrm{target}}
=
\frac{\|c_{\mathrm{rc}}^{\mathrm{full}}-c^*\|}
{D_{\mathrm{section}}}.
\]

target-domain 边界只在 train 内依据应用容限和 GT/参考误差冻结：

- TARGET_ALIGNED；
- ECCENTRIC_GT；
- TARGET_UNKNOWN。

sealed_test 上 target domain 只用于事后分层，不能改变候选、状态或最终坐标。

target-domain index 同时保存完整截面 reference 的归一化尺度，并满足
`target_bias_px = target_bias_norm × reference_normalization_scale_px`；审计阶段逐条复核该恒等式及 reference 角色，防止把局部 crop 尺度误当作完整截面尺度。

主结果必须同时报告：

- full-domain；
- TARGET_ALIGNED；
- ECCENTRIC_GT；
- TARGET_UNKNOWN；
- 各域完整分母和 risk capture。

### 8.5 模型风险

模型风险与 geometry_state 并列：

| model_risk | 含义 |
|---|---|
| LOW_RISK | 当前诊断未发现显著 target mismatch 或共同偏差 |
| MEDIUM_RISK | 存在可解释的轻中度模型偏离，禁止直接签发 production_usable |
| HIGH_RISK | 几何候选存在，但生物学髓心代理风险高 |
| UNKNOWN | 数据或校准功效不足 |

MEDIUM_RISK/HIGH_RISK 不得改写成 RANGE 或 REJECT；它们可以阻止 POINT 的 production_usable。错误 association、非法拓扑或数据合同破坏属于数据/结构无效，应由 S0/S1/S3 将 geometry_state 判为 REJECT，而不是借 model_risk 改写几何状态。

### 8.6 风险—覆盖

对冻结置信/风险排序计算：

- POINT risk–coverage；
- catastrophic risk–coverage；
- structured-output coverage；
- production-usable POINT coverage；
- REJECT 及原因；
- aligned/full-domain 分层曲线。

风险很低但几乎全部 REJECT 不能视为优秀；覆盖高但 False-POINT 高也不能视为成功。

### 8.7 S4 门

- 点 bootstrap 被当作独立样本：FAIL；
- 模型风险改变几何状态以规避错误：FAIL；
- sealed_test GT 被用于风险门调节：FAIL；
- 未校准却宣称 nominal coverage：FAIL；
- 测量、几何、模型风险分开且状态支持与 replay 稳定：PASS；
- 校准树不足：PASS_WITH_RISK 或 EXPLORATORY，禁止强 coverage 结论。

---

## 9. 各几何状态的专用评价

### 9.1 POINT

报告：

- e_px、e_norm 和可用时 e_mm；
- tree-balanced median、P90、P95、max；
- catastrophic rate；
- support containment；
- 支持区长短轴；
- False-POINT；
- finite–far gap；
- search/replay 稳定性；
- target domain 和 model risk；
- production_usable。

典型错误解释：

- 点误差小、支持不覆盖 GT：点估计偶然好，不确定性失准；
- 点误差大、支持很紧：危险伪 POINT，检查 target/model risk；
- 点误差大、支持很宽：状态门过激；
- 远场存在等价分量：不应为 POINT。

### 9.2 RANGE

报告：

- GT 方向误差；
- GT 距离是否位于有限区间；
- 区间宽度和 log-range 宽度；
- 上下界对 replay 的稳定性；
- 是否被错误转成 POINT；
- pith_raw 误差仅作诊断。

RANGE 的主要成功标准是诚实覆盖和有用方向/距离区间，不是 raw candidate 点误差。

### 9.3 RAY

报告：

- 有向角误差；
- 方向区间覆盖；
- GT 是否满足距离下界；
- 单端远场接触稳定性；
- ring-order/极性定向来源；
- RAY↔AXIS 重放一致性；
- false finite conversion。

### 9.4 AXIS

报告：

- 模 \(\pi\) 方向误差；
- 轴区间宽度；
- 双端远场接触；
- 是否存在可靠定向信息但未使用；
- AXIS→POINT 严重误转率；
- 远场 fixture 一致性。

### 9.5 MULTIMODAL

报告：

- 持久 support component 数；
- 分量间距；
- 有限/远场组合；
- GT 是否落入任一稳定分量；
- 相对高预算 reference 的 relevant-mode recall；
- threshold、网格和 replay 下的 mode persistence。

不得用 GT 选中一个分量后改报 POINT。

### 9.6 REJECT

至少区分：

~~~text
INVALID_SOURCE_RECORD
COORDINATE_INVALID
LINEAGE_BROKEN
CURVE_TOPOLOGY_INVALID
INSUFFICIENT_RING_EVIDENCE
RING_ASSOCIATION_INVALID
SOLVER_UNSTABLE
SEARCH_INADEQUATE
DIRECTION_UNIDENTIFIABLE
UPSTREAM_INVALID
~~~

报告 first_failure_stage、可修复性、难度分层和完整分母占比。

### 9.7 状态正确性

#### synthetic/controlled fixture

建立六态混淆矩阵，重点错误：

~~~text
RAY/AXIS -> POINT
MULTIMODAL -> POINT
SEARCH_INADEQUATE -> RANGE/AXIS
有限 + 远场双分量 -> POINT
模型高风险 -> 未标记 production POINT
~~~

#### 真实 UruDendro4 nested crops

真实 crop 不宜用固定 GT 距离直接定义“真状态”。评价：

- 随总转角下降或图外距离增大，距离支持总体是否变宽；
- 状态是否总体从 POINT 向 RANGE/RAY/AXIS 退化；
- 方向误差是否比距离误差退化更慢；
- False-POINT 是否受控；
- support containment 是否稳定；
- 同一 section 的嵌套 crop 状态转移是否合理。

---

## 10. S5：层级 exact contribution 与无 GT CF 校准

### 10.1 最小产物

~~~text
contribution_registry
ring_delete_refit
visible_arc_delete_refit
subarc_delete_refit
full_vs_deleted_search
full_vs_deleted_profiles
contribution_gt_records
contribution_intervals
direction_shift_conflict_roles
scale_phase_stability
cross_fitted_contribution
sign_calibrator_oof
sign_calibrator_test
contribution_heatmaps
~~~

### 10.2 冻结估计器

生成正式贡献前必须冻结：

- 曲线平滑与端点保护；
- sigma 和图像质量映射；
- pseudo-Huber 参数；
- parent-ring budget；
- c0、ct、VarPro；
- J_infinity、profile 和二维审计策略；
- 支持集和状态规则；
- 模型风险规则；
- 删除组尺度和相位。

### 10.3 exact delete-refit 合同

对每个 G：

\[
M_{\mathrm{all}}=\mathcal M(\mathcal E),
\qquad
M_{-G}=\mathcal M(\mathcal E\setminus G).
\]

删除后必须：

1. 删除完整连续组及其冻结预算质量；
2. 不重新分配删除预算；
3. 重新计算剩余 ring 均值和 squared norm；
4. 重新计算 c0、H0 和 ct；
5. 重新求所有剩余 ring 半径；
6. 重新运行二维 VarPro；
7. 重新运行 J_infinity；
8. 按固定触发器重跑 profile/二维审计；
9. 重建支持分量、状态和模型风险；
10. 保存删除前后搜索充分性。

完整解可作额外 warm start，但不能取代冻结的其他搜索路径。

不满足时标记：

~~~text
INVALID_DELETION
UNRESOLVED_SEARCH
INSUFFICIENT_REMAINING_EVIDENCE
STATE_ONLY
NOT_APPLICABLE
~~~

### 10.4 三层不可相加

正式层级：

1. leave-one-ring-out；
2. leave-one-visible-arc-out；
3. leave-one-subarc-out。

ring、arc 和 subarc 分别汇总，不得跨层相加。相邻 subarc 高度相关；同一弧不同 scale/phase 只能作稳定性 replay，不能当作更多独立贡献样本。

V1 发布顺序：

- ring 必做；
- 一个预注册非重叠主尺度 subarc 必做；
- visible arc、多尺度和双相位可作为 V2；
- 若细尺度不稳，回退到更高层级结论。

### 10.5 GT signed contribution

仅当 full 和 deleted 结果均通过搜索门且均为 POINT 时计算：

\[
C_{G,\mathrm{px}}^{GT}
=
\|\hat c_{-G}-c^*\|_{\mathrm{px}}
-
\|\hat c-c^*\|_{\mathrm{px}},
\]

\[
C_{G,\mathrm{norm}}^{GT}
=
\frac{
\|\hat c_{-G}-c^*\|-\|\hat c-c^*\|
}{D_{\mathrm{FOV}}},
\]

有物理标定时：

\[
C_{G,\mathrm{mm}}^{GT}
=
\|\hat c_{-G}-c^*\|_{\mathrm{mm}}
-
\|\hat c-c^*\|_{\mathrm{mm}}.
\]

平方误差贡献：

\[
C_{G,\mathrm{sq}}^{GT}
=
\|\hat c_{-G}-c^*\|^2
-
\|\hat c-c^*\|^2.
\]

相对贡献：

\[
C_{G,\mathrm{rel}}^{GT}
=
\frac{
\|\hat c_{-G}-c^*\|^2-\|\hat c-c^*\|^2
}{
\|\hat c_{-G}-c^*\|^2+\|\hat c-c^*\|^2+\tau_C^2
}.
\]

正值表示保留 G 使结果更接近 GT；负值表示保留 G 使结果偏离 GT。

### 10.6 贡献区间和死区

full 与 deleted 必须使用配对扰动 seed。区间来源可包括：

- GT 重复标注或明确 GT 不确定性；
- 连续移动块曲线扰动；
- spline/quadrature replay；
- solver replay。

标签：

~~~text
lowerCI(C) > +epsilon_C  -> BENEFICIAL_GT
upperCI(C) < -epsilon_C  -> HARMFUL_GT
区间完整位于死区内     -> NEUTRAL
其余                     -> UNCERTAIN
~~~

若无 GT 不确定度资料，必须说明区间未覆盖 GT 标注误差，不能把 solver 稳定区间称为完整统计置信区间。

### 10.7 非 POINT 删除结果

以下情况不计算或不解释 signed mm/px contribution：

- full 不是 POINT；
- deleted 不是 POINT；
- 任一搜索不充分；
- 删除后证据不足；
- 出现新稳定模态。

优先报告：

| 删除后变化 | 角色 |
|---|---|
| POINT→RANGE/RAY/AXIS | IDENTIFIABILITY_CRITICAL |
| 方向区间显著变宽 | DIRECTION_CRITICAL |
| 距离区间显著变宽 | RANGE_CRITICAL |
| 新增稳定分量 | MODE_EXCLUSION |
| 中心大幅移动 | HIGH_LEVERAGE |
| 其余证据拟合显著改善 | CONFLICTING |
| 所有变化均在死区内 | REDUNDANT |

同一 G 允许多角色。

### 10.8 信息、位移、冲突与符号分离

固定单尺度非重叠分割中：

\[
I_G^{\mathrm{dir}}
=
\log\det(A+\varepsilon I)
-
\log\det(A-A_G+\varepsilon I)
\ge 0.
\]

位移：

\[
C_G^{\mathrm{shift}}
=
\|\hat c-\hat c_{-G}\|.
\]

冲突：

\[
K_G^{\mathrm{conflict}}
=
J_{-G}(\hat c)-J_{-G}(\hat c_{-G})
\ge0.
\]

解释限制：

- directional_info_gain 非负，不是正贡献；
- shift 大表示高杠杆，不决定方向；
- conflict 大表示证据冲突，不证明 G 错；
- residual 大不等于 HARMFUL_GT；
- 图像质量差不等于负贡献。

### 10.9 无 GT cross-fitted contribution

对目标组 G，把其余 parent rings 分成 witness folds \(V_f\)：

\[
T_f=\mathcal E\setminus(G\cup V_f),
\]

\[
\hat c_f^-=\mathcal M(T_f),
\qquad
\hat c_f^+=\mathcal M(T_f\cup G),
\]

\[
C_G^{CF}
=
\operatorname{median}_f
\left[
L(V_f;\hat c_f^-)-L(V_f;\hat c_f^+)
\right].
\]

witness ring 必须完全未参与两次拟合，并在给定中心下重新 profile 自己的半径。

ring 级 CF 中，目标 parent ring 整体从训练与 witness 中排除。subarc 级 CF 测量的是“在同 ring 兄弟段仍作为上下文时该 subarc 的条件作用”，兄弟段可按预注册规则保留在训练证据中，但目标 subarc 所属 parent ring 的任何部分都不得进入 witness。

以下情况必须 ABSTAIN：

- 独立 parent rings 不足；
- 两次训练任一为 AXIS、MULTIMODAL 或 REJECT；
- witness 几何方向单一；
- fold 间符号不稳；
- scale/phase 不稳；
- 搜索不充分。

C_CF 表示对未参与拟合年轮的支持/冲突，不等于生物学真值贡献。

### 10.10 符号概率校准器

在 train 内以 tree_id 外折方式生成：

- 标签：exact C_GT 的 BENEFICIAL / NEUTRAL / HARMFUL；
- 特征：只允许无 GT 特征；
- 模型：低容量正则化三分类或有序模型；
- 概率校准：tree-grouped OOF；
- 最终测试：冻结后在 sealed_test 一次性评价。

允许特征：

- C_CF；
- conflict；
- LOO shift；
- direction/range/state change；
- directional_info_gain；
- ring-order 违反；
- 图像—标注质量；
- scale/phase/replay 稳定性。

输出：

~~~text
p_positive
p_neutral
p_negative
expected_contribution
prediction_interval
abstain_reason
~~~

评价：

- macro-F1；
- balanced accuracy；
- negative-class PR-AUC；
- Brier score；
- ECE/可靠性图；
- expected contribution 与 C_GT 的 Spearman；
- precision@k；
- abstention–risk 和 coverage。

必须与类别先验、仅 residual、仅 conflict、仅 shift 等简单基线比较。

### 10.11 S5 门

- 删除后未完整重跑：FAIL；
- 预算被重新分配：FAIL；
- 非 POINT 用 raw point 生成巨大 signed contribution：FAIL；
- residual/conflict/图像质量直接决定正负：FAIL；
- scale/phase 不稳却发布细粒度热图：EXPLORATORY；
- exact GT contribution 合法但 CF 模型不稳：贡献解释 PASS，运行时符号 EXPLORATORY；
- 校准器未优于先验或概率失准：运行时输出 ABSTAIN；
- 无论贡献结果如何，最终坐标仍为 All-Arc。

---

## 11. S6：端到端基线、消融、压力测试与 sealed test

### 11.1 几何公平基线

所有方法使用同一人工年轮标注和 crop manifest：

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | 每 ring 独立圆拟合后聚合圆心 | 传统局部圆基线 |
| B1 | 切向零投影 WLS/Huber | 方向束基线 |
| B2 | 凸共同中心 c0 | 验证主初值 |
| B3 | L2 共同中心 VarPro | 验证稳健核 |
| B4 | c0 + pseudo-Huber VarPro | 核心有限中心 |
| B5 | B4 + J_infinity + 条件 profile | 有限—远场与状态 |
| RAC-Pith v2 | B5 + 质量、风险、不确定性和贡献输出 | 完整方法 |

B0–B4 若没有状态机制，其点误差可作定位诊断，但不得与 B5/RAC 的 selective POINT coverage 混成单一排名。

仅图像输入的 APD/LFSA/ACO 类方法如被实现，应单独成表，明确其输入信息少于人工年轮线方法。

### 11.2 必做消融

1. 点数加权 vs parent-ring evidence conservation；
2. 切向初值 vs 凸共同中心 c0；
3. c0 alone vs c0 + VarPro；
4. L2 vs pseudo-Huber；
5. 固定外部搜索框 vs J_infinity + 条件距离 profile；
6. 无/有 ring-order 轻量门；
7. geometry only vs image/annotation quality；
8. 独立点 bootstrap vs 移动块/ring 层级重采样；
9. in-sample residual heuristic vs C_CF；
10. influence approximation vs exact delete-refit；
11. 预注册 subarc 多尺度和双相位；
12. All-Arc vs GT-oracle single delete，仅作上限；
13. TARGET_ALIGNED vs full-domain，作为适用域分层；
14. 无/有低容量偏差修正，仅限研究版；
15. Core/Profile vs 高预算二维审计；
16. CPU vs cuda:0 数值 replay，若使用 GPU。

每项报告：

- 模块唯一职责；
- paired tree-level 主终点差；
- P90/P95/catastrophic/False-POINT；
- coverage；
- solver/search failure；
- runtime；
- KEEP / AUDIT_ONLY / OFF 结论。

### 11.3 synthetic 几何 fixture

至少覆盖：

- 理想同心圆；
- 不同半径、不同 ring 数；
- 图内、图外和趋于无穷远；
- 近直线极短弧；
- 单端/双端远场接触；
- 有限与远场竞争分量；
- 两个有限稳定分量；
- 只有一到两条 ring；
- 同 ring 多 fragment；
- 不同点密度；
- 坐标平移、旋转和统一缩放；
- 受控异方差和 outlier；
- 错误 ring association/order。

fixture 用于状态和数学实现真值，不用于模拟真实生物学全部复杂性。

### 11.4 UruDendro4 真实 crop 压力测试

从同一完整截面构造：

- 不同 crop 尺寸；
- 不同 overlap；
- 同一位置的 nested crops；
- 不同图外距离；
- 单侧与多方位证据；
- 不同可见 parent-ring 数；
- 自然裂纹、节疤、污渍、模糊区域；
- 不同图像分辨率。

所有派生 crop 继承原 tree split。

### 11.5 人工扰动

预注册严重度后施加：

- 连续子弧平移；
- 法向或切向低频偏转；
- 局部抖动；
- 子弧缺失；
- 伪弧；
- ring ID 错配；
- ring order 错误；
- 多 ring 共同偏移；
- 偏心低频形变；
- 图像模糊、对比度下降、裂纹样遮挡；
- 标注分辨率下降。

分别评价：

1. 异常是否被质量/风险诊断发现；
2. 定位与状态是否稳健；
3. exact contribution 符号和角色是否合理；
4. CF/概率模型能否识别；
5. 是否出现新的 False-POINT。

人工异常不必然产生 HARMFUL_GT；它可能偶然抵消原偏差。因此异常检测和贡献符号必须分别评分。

### 11.6 工程性能

按运行模式分开：

| 模式 | 内容 |
|---|---|
| CORE | 曲线统计、c0、ct、二维 VarPro、J_infinity |
| PROFILE | 条件方向/距离 profile |
| AUDIT | 二维紧致搜索、完整 bootstrap、高预算 replay |
| CONTRIBUTION | K 个删除组的完整重求解 |

报告：

- P50/P90/P95 wall time；
- CPU time；
- 峰值 RAM；
- GPU 显存，若启用；
- profile 和二维审计触发比例；
- 每 crop 求积节点数；
- 每删除组平均时间；
- 并行效率；
- solver failure；
- 输出磁盘量。

GPU 加速只在相同配置、相同浮点精度或明确记录精度差异下比较；必须验证 CPU/GPU 的中心、profile、状态和贡献标签在容差内一致。

### 11.7 sealed test

sealed_test 前必须冻结：

- split；
- crop generator；
- 所有算法配置；
- 状态和风险门；
- 支持集阈值；
- epsilon_C；
- CF 模型和概率校准器；
- 表图脚本；
- 主终点和非劣界。

sealed_test 运行后：

- GT 仅用于一次性评价；
- 不更改阈值；
- 不删失败样本；
- 不换 seed；
- 不针对失败图像改 crop；
- 不训练第二版后继续称同一 sealed_test 为 sealed。

若未达门，结论是“当前冻结版本未通过”，而不是回看 sealed_test 后修正并重报。

### 11.8 核心实验注册表

| 实验 | 唯一问题 | 主要产物 | 允许结论 |
|---|---|---|---|
| E0 | 数据配对、树级 split、坐标和 lineage 是否正确 | S0 硬合同报告 | 数据可否进入实验 |
| E1 | 连续弧、sigma、图像质量和证据预算是否可信 | S1 replay 与 overlay | 测量层是否有效 |
| E2 | c0 与鲁棒 VarPro 是否提供稳定有限估计 | S2 初值/精化消融 | 有限估计模块 KEEP/OFF |
| E3 | J_infinity/profile 是否避免伪有限点并保留多模态 | S3 reference 和状态结果 | 搜索及六态是否诚实 |
| E4 | 完整方法是否改善精度、尾部风险与 coverage | 主定位表和 risk–coverage | 核心定位性能 |
| E5 | 支持区、target domain 和模型风险是否有效 | S4 calibration/risk 表 | 可用性与适用域 |
| E6 | exact contribution、CF 和符号校准是否有效 | S5 贡献与概率表 | 解释分辨率和无 GT 能力 |
| E7 | 扰动、nested crop 和异常条件下是否稳健 | 压力测试表 | 失效边界 |
| E8 | Core/Profile/Audit/Contribution 成本是否可接受 | 工程性能表 | 运行模式和预算 |
| E9 | 冻结版本在 6 棵 sealed_test 树上是否复现结论 | sealed_test 全套表图 | 最终结论；区间受 6 树功效限制 |

每个实验只回答表中的唯一问题。上游实验失败时，下游数值即使存在也不得形成正式结论。

---

## 12. 模块级验收逻辑

### 12.1 硬正确性门

以下任一失败即不可进入科学比较：

- tree split 泄漏；
- 图像/标注/髓心错配；
- x/y、row/column 或坐标变换错误；
- 图外 GT 被裁剪；
- lineage 断裂；
- crop 边误作年轮；
- 点复制、插值或 fragment split 改变 parent budget；
- c0 恒等式 fixture 失败；
- 删除后重新分配证据预算；
- 非有限数值未被捕获。

### 12.2 搜索诚实性门

拟输出 POINT 必须同时满足：

- J_infinity 已运行；
- 有限—远场间隔通过冻结门；
- 角度加密稳定；
- 必要 profile 已运行；
- 不存在分离的等价有限或远场分量；
- 边界未截断；
- 多初值和 replay 稳定；
- 高预算审计子集达到预注册 relevant-mode recall。

不满足即 MULTIMODAL 或 REJECT，不能为了提高 POINT coverage 放宽。

### 12.3 核心定位 KEEP 门

RAC-Pith v2 相对最强公平基线的结论必须联合考虑：

- tree-level median/P90 是否改善或至少不劣；
- P95、catastrophic 和 False-POINT 是否不恶化；
- POINT/operational coverage 是否未异常下降；
- solver/search failure 是否受控；
- aligned-domain 和 full-domain 是否一致；
- 计算开销是否与收益相称。

不以单一平均误差或单个 p 值决定 KEEP。

### 12.4 可选模块 KEEP 门

图像质量映射、ring-order 定向、完整 bootstrap、二维高预算审计和可选低容量校正分别要求：

- 在独立 tree 上改善其预定职责；
- 不恶化尾部和 False-POINT；
- 不通过大量拒识换取表面收益；
- 不把 RANGE/RAY/AXIS 伪升级为 POINT；
- 不使用 sealed_test GT。

否则标记 AUDIT_ONLY 或 OFF。

### 12.5 贡献发布门

ring/arc/subarc 贡献必须：

- exact delete-refit；
- 搜索充分；
- 状态合法；
- 有完整分母；
- signed 结果仅限 POINT–POINT；
- CI 与 epsilon_C 规则冻结；
- scale/phase 稳定；
- 角色和符号分开；
- 不反馈最终坐标。

细粒度不稳时，只发布 ring 或 visible-arc 结论。

### 12.6 CF/概率模型门

无 GT 模型只有在独立 tree sealed_test 上：

- 优于类别先验和简单诊断基线；
- negative PR-AUC、precision@k 达到预注册用途要求；
- Brier/ECE 显示概率可用；
- abstention 后风险—覆盖合理；
- 各 treatment/block/难度层无明显崩溃；

才允许输出部署期符号概率。否则只输出 C_CF 支持/冲突与 ABSTAIN，不输出“负贡献弧”断言。

### 12.7 默认 All-Arc 门

无论 GT-oracle 单删表现如何：

~~~text
final coordinate = All-Arc RAC-Pith v2
contribution = explanation / quality control
Safe-Prune = OFF
~~~

本规范不提供任何根据贡献改变最终坐标的验收通道。

---

## 13. 端到端核心结果表

### 表 A：数据、split 与目标域

| split | trees | sections | crops | inside/outside strata | ring-count strata | aligned | eccentric | unknown |
|---|---:|---:|---:|---|---|---:|---:|---:|

### 表 B：阶段通过率和首失败

| stage | PASS | PASS_WITH_RISK | EXPLORATORY | FAIL | NOT_APPLICABLE | main reason |
|---|---:|---:|---:|---:|---:|---|

### 表 C：主定位比较

| method | POINT coverage | operational coverage | median | P90 | P95 | False-POINT | catastrophic | reject/fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

分别以 px、D_FOV 归一化量和可用时 mm 输出，不得混在同一列。

### 表 D：状态专用质量

| state | trees/crops | containment | direction error | range/support width | topology stability | false conversion |
|---|---:|---:|---:|---:|---:|---:|

### 表 E：搜索充分性

| variant | far-grid agreement | profile trigger | 2-D audit | best recall | component recall | false finite | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|

### 表 F：不确定性和模型风险

| domain/state | nominal level | empirical coverage | support size | False-POINT | risk capture | false veto |
|---|---:|---:|---:|---:|---:|---:|

### 表 G：消融

| ablation | unique duty | paired median delta | paired P90 delta | False-POINT delta | coverage delta | runtime delta | decision |
|---|---|---:|---:|---:|---:|---:|---|

### 表 H：exact contribution

| level/scale | total | signed evaluable | state-only | unresolved | beneficial | neutral | harmful | uncertain | critical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 表 I：CF 与概率校准

| model | evaluable coverage | abstain | macro-F1 | balanced accuracy | negative PR-AUC | Brier | ECE | precision@k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### 表 J：压力测试

| perturbation | severity | center drift | state agreement | False-POINT delta | risk capture | contribution stability |
|---|---|---:|---:|---:|---:|---:|

### 表 K：工程性能

| mode | P50 | P90 | P95 | peak RAM | peak VRAM | solver fail | output size |
|---|---:|---:|---:|---:|---:|---:|---:|

所有表必须同时显示独立 tree 数和 crop 数；统计区间以 tree cluster 为单位。

---

## 14. 端到端主图与单 crop 分析卡

### 14.1 主图

1. 数据/split/难度完整分母图；
2. 原图、crop、parent rings 和图外 GT overlay；
3. c0、ct、VarPro、GT 和支持区的扩展画布；
4. 方法误差 ECDF 与 tree-level paired difference；
5. False-POINT risk–POINT coverage；
6. catastrophic risk–operational coverage；
7. H0 条件比、总转角、图外距离与状态/支持宽度；
8. J_infinity 和有限—远场间隔；
9. 弱方向/竞争方向 P(z) profile；
10. nested crop 的状态转移矩阵；
11. nominal–empirical support coverage；
12. target-domain 分层误差与 False-POINT；
13. 模型风险 capture–false-veto；
14. 弧线上多通道贡献图；
15. full 与 deleted 的中心/profile/mode 对照；
16. CF/概率模型混淆矩阵、可靠性图和 abstention–risk；
17. runtime 随 ring、quadrature node 和 deletion group 数的曲线。

图外髓心必须用扩展坐标画布、方向箭头或 inset 表达，不得把点 clip 到 crop 边缘。

贡献颜色至少同时表达：

- 正/负/中性；
- UNCERTAIN；
- CI 或概率；
- ring/arc/subarc 层级；
- DIRECTION/RANGE/MODE 关键角色。

不能只用红绿二色。

### 14.2 单 crop 标准卡

每个代表性和失败 crop 生成：

~~~text
A. provenance
   tree / section / crop / split / hashes

B. data
   image / annotations / GT / transform / lineage

C. evidence
   fragments / quality / sigma / parent budget

D. finite estimator
   c0 / H0 / ct / VarPro / per-ring radius and loss

E. far and state
   J_infinity / finite-far gap / profiles / modes / state trace

F. uncertainty and risk
   support / replay / target domain / model risk / production_usable

G. localization metrics
   state-specific errors and containment

H. contribution
   ring/arc/subarc sign, CI, shift, conflict and roles

I. conclusion
   first failure, allowed use, forbidden interpretation
~~~

### 14.3 单 crop 结论必须回答

1. 当前证据支持有限点、范围、有向射线、无向轴还是多个模态？
2. 搜索是否充分？
3. GT 是否落入对应支持区？
4. 若为 POINT，误差、尾部风险和 target/model risk 如何？
5. 哪些弧提供方向、距离或排模态作用？
6. 哪些弧在 GT 下正、负、中性或不确定？
7. 无 GT CF 是否可计算，是否拒答？
8. 为什么最终坐标仍是 All-Arc？

---

## 15. 推荐的实际分析顺序

### Step 1：锁定 manifest 和环境

生成 source、tree、section、crop 和结果索引；记录配置、环境和版本。

### Step 2：只检查 S0

先完成数据配对、tree split、坐标、overlay、lineage 和图外 GT。任何硬错误先修复，不分析定位误差。

### Step 3：检查 S1

审计连续弧、缺口、端点、sigma、图像质量、parent budget 和 quadrature。

### Step 4：检查 S2

先看 c0 恒等式、IRLS、H0、ct 和 VarPro 收敛，再看有限 GT 误差。

### Step 5：检查 S3

每个 crop 检查 J_infinity；困难样本检查 profile 和二维审计；先确定搜索充分性和状态。

### Step 6：按状态评价 S4

POINT、RANGE、RAY、AXIS、MULTIMODAL、REJECT 分表分析；再做 uncertainty、target domain、model risk 和 risk–coverage。

### Step 7：先 ring 后 subarc 分析 S5

先验证 exact delete-refit 合同，再计算 GT sign；非 POINT 优先分析状态角色；最后训练和评价 CF/校准器。

### Step 8：做基线与消融

固定同一 manifest，使用 tree-level paired endpoint 判断各模块 KEEP、AUDIT_ONLY 或 OFF。

### Step 9：做压力与性能

synthetic fixture、真实 nested crop、人工扰动和 Core/Profile/Audit/Contribution 算力分开报告。

### Step 10：最后运行 sealed test

冻结后一次性输出所有表、图、失败和负结果，不回看 sealed_test 调参。

---

## 16. 论文或报告中的结论强度

### 16.1 强结论

只有同时满足：

- tree-level 独立 sealed_test；
- 全部硬合同和搜索门通过；
- POINT 的 P90、False-POINT 和 coverage 达到预注册门；
- RANGE/RAY/AXIS 的状态专用指标合理；
- target/full-domain 风险被完整报告；
- uncertainty 在独立 tree 上校准；
- exact contribution 合法且稳定；
- CF 概率仅在独立 sealed_test 上达标；

才可写：

> RAC-Pith v2 在声明的适用域内能够利用局部年轮弧恢复可辨识的有限髓心，并在距离证据不足或存在竞争解时输出 RANGE、RAY、AXIS 或 MULTIMODAL，而不是伪造有限点。其弧段 GT 正负贡献来自冻结 All-Arc 管线的完整删除重求解。

### 16.2 中等强度结论

几何主线和状态机制有效，但支持阈值、model risk 或贡献概率校准的独立树功效不足时：

> 实验支持 RAC-Pith v2 的有限—远场退化处理和 exact GT contribution 定义，但统计 coverage 或无 GT 符号概率仍属于研究级结果。

### 16.3 探索性结论

只有同树多 crop、校准样本不足或高预算 reference 覆盖不足时：

> 当前结果显示相应模块的趋势，但重叠 crop 不能替代独立树；状态阈值、coverage 和贡献预测不得视为已冻结。

### 16.4 必须报告的负结果

例如：

> exact GT contribution 表明部分弧在特定 crop 上具有负贡献，但无 GT cross-fitted/calibration 模型不能在独立树上稳定识别这些弧。因此贡献仅用于解释和标注质检，最终定位保持 All-Arc。

这属于有效科学结论，不应通过引入 Safe-Prune 掩盖。

### 16.5 禁止表述

在没有对应证据时禁止：

- “证明了全局最优”；
- “对所有局部 crop 都可输出精确髓心”；
- “高残差弧就是错误弧”；
- “贡献概率等于因果贡献”；
- “大量 crop 提供了大量独立树”；
- “RANGE/RAY/AXIS 是定位失败”；
- “只看平均误差即可证明全面提升”；
- “GT-oracle 单删是可部署改进”；
- “模型最终坐标经过贡献优化”。

---

## 17. 最终验收清单

### 17.1 数据和 split

~~~text
[ ] 数据根目录只读，输出写入独立目录
[ ] 图像、标注和 pith_location.txt 按 section_id 唯一 join
[ ] tree_id 严格为 T{treatment}_B{block}_N{tree}
[ ] 同树不同高度未跨 split
[ ] 同截面所有重叠 crop 未跨 split
[ ] 共解析出 24 个唯一 tree_id
[ ] 默认拆分为 18 棵 train 树和 6 棵 sealed_test 树
[ ] train/sealed_test tree_id 交集为空
[ ] train 内部 fold 也按 tree 分组
[ ] 所有输入和失败样本进入完整分母
[ ] manifest、配置、代码和环境版本可追溯
~~~

### 17.2 坐标和 lineage

~~~text
[ ] x/y 与 row/column 已由 overlay 确认
[ ] 坐标原点和索引基已确认
[ ] source↔crop round-trip 通过
[ ] 平移/旋转/缩放等变性通过
[ ] 图外 GT 未 clip
[ ] crop 边界未误作年轮
[ ] parent ring / visible arc / subarc lineage 完整
[ ] 真缺口没有被样条跨越
~~~

### 17.3 连续证据

~~~text
[ ] spline 拓扑和端点保护通过
[ ] sigma_x 与 sigma_alg 单位分开
[ ] sigma 不依赖当前候选 residual
[ ] 图像质量特征候选无关
[ ] 顶点复制不改变结果
[ ] 插值加密不改变结果
[ ] fragment split 不改变 parent budget
[ ] 删除预算不重新分配
[ ] quadrature 收敛不改变关键结论
~~~

### 17.4 定位器

~~~text
[ ] c0 半径消元恒等式 fixture 通过
[ ] c0 IRLS 收敛并记录 H0
[ ] ct 只作第二初值和诊断
[ ] VarPro 内层半径失败不沿用旧值
[ ] VarPro 外层收敛和 reason code 完整
[ ] 分离有限谷未被静默合并
[ ] 多初值全部以同一 J_geo 重评分
~~~

### 17.5 远场和状态

~~~text
[ ] 每个 crop 都运行 J_infinity
[ ] J_infinity 解析极限经 fixture 验证
[ ] 角度网格加密结论稳定
[ ] 所有触发条件都有 trace
[ ] 弱方向和竞争方向 profile 使用解析远场端点
[ ] 一维覆盖不足时进入二维紧致审计
[ ] 搜索不足输出 REJECT
[ ] 有限与远场双分量输出 MULTIMODAL
[ ] RAY 与 AXIS 定向语义正确
[ ] 边界点未被当作有限 POINT
~~~

### 17.6 不确定性和风险

~~~text
[ ] 未使用独立点 bootstrap
[ ] 测量、几何、模型风险分开
[ ] bootstrap 新模态会触发 S3 重审
[ ] POINT/RANGE/RAY/AXIS/MULTIMODAL 使用各自指标
[ ] 未校准支持区不宣称概率 coverage
[ ] TARGET_ALIGNED 和 full-domain 同时报告
[ ] sealed_test target domain 只作事后分层
[ ] model risk 未改写 geometry_state
[ ] False-POINT、coverage 和 catastrophic 联合报告
~~~

### 17.7 贡献

~~~text
[ ] 定位器和删除 registry 在贡献前冻结
[ ] ring/arc/subarc 分层单独计算
[ ] exact deletion 完整重跑 c0/ct/VarPro/J_infinity/profile/state
[ ] signed GT contribution 仅限 POINT–POINT
[ ] contribution CI 与 epsilon_C 已冻结
[ ] 非 POINT 删除结果优先报告状态角色
[ ] directional_info_gain、shift、conflict 与 sign 分开
[ ] residual 和图像质量未直接决定正负
[ ] scale/phase 不稳时回退更高层级
[ ] CF witness ring 未参与拟合
[ ] CF 不足时输出 ABSTAIN
[ ] 概率校准器按 tree OOF
[ ] 最终坐标未被贡献反馈改变
[ ] Safe-Prune 为 OFF
~~~

### 17.8 统计和 sealed test

~~~text
[ ] 主终点按 tree 聚合
[ ] 方法比较使用相同 crop manifest
[ ] cluster bootstrap/permutation 以 tree 为单位
[ ] 同时报告效果量和区间
[ ] 完整分母与 common-valid 诊断分开
[ ] px、normalized、mm 单位未混用
[ ] 无标定样本未伪造 mm
[ ] 所有阈值仅由 train 冻结
[ ] sealed_test 只运行一次冻结方案
[ ] sealed_test 后未删除失败或回调阈值
[ ] 所有主表、主图和负结果齐全
~~~

---

## 18. 手动运行与分析脚本的语义合同

本规范不限定代码内部结构，但手动脚本应覆盖以下职责：

| 职责 | 输入 | 输出 |
|---|---|---|
| 数据审计与 split | UruDendro4 根目录、split 配置 | source/tree/section split 和泄漏报告 |
| crop 生成 | section manifest、crop 配置 | crop manifest、图像/标注派生物、坐标变换 |
| 核心运行 | crop manifest、冻结定位配置 | 每 crop S0–S4 结构化记录 |
| 贡献运行 | 合格结果、冻结贡献 registry | ring/arc/subarc exact 与 CF 记录 |
| 结果分析 | manifest、所有结构化 records | tree-level 主表、统计区间和验收报告 |
| 可视化 | records、原图只读路径 | 主图和单 crop case card |
| 静态/fixture 审计 | 配置与 synthetic fixtures | 逻辑、公式和 I/O 审计；可执行 fixture 仍只在生产 `py311` 手动运行 |

本实现对应的入口已经固定如下；编号表达数据依赖顺序，不表示测试集可以在冻结前运行：

| 阶段 | Python 入口 | 手动入口 |
|---|---|---|
| 路径与环境合同 | `scripts/00_resolve_runtime_paths.py`、`00_check_environment.py` | `scripts/manual/00_check_environment.sh`；总控自动解析路径 |
| 索引、分树、重叠裁窗 | `scripts/00_index_dataset.py`、`01_split_by_tree.py`、`02_generate_crops.py` | `scripts/manual/00_prepare_data.sh` |
| 连续证据、核心、基线 | `scripts/03_build_evidence.py`、`04_run_localization.py`、`11_run_baselines.py` | `scripts/manual/10_run_development_core.sh` |
| 结构化不确定性 | `scripts/05b_run_uncertainty.py`（强制绑定定位 result index） | `scripts/manual/15_run_development_uncertainty.sh` |
| exact/CF 贡献 | `scripts/05_run_contributions.py`（强制绑定定位 result index） | `scripts/manual/20_run_development_contributions.sh` |
| 子弧尺度/相位敏感性 | 同一入口加 `--include-audit-partitions`，独立输出根 | `scripts/manual/22_run_development_partition_audit.sh` |
| 无 GT 贡献校准 | `scripts/06_calibrate_contributions.py`、`06b_apply_contribution_calibrator.py` | `scripts/manual/25_fit_contribution_calibrator.sh` |
| 完整截面目标域 | `scripts/06d_assess_target_domain.py` | `scripts/manual/30_analyze_development.sh` 与 sealed 总控 |
| 定位/贡献统计 | `scripts/07_evaluate.py`；`07b_evaluate_contributions.py` **必须传 `--crop-manifest`**，并与 contribution index 的完整分母/lineage 对齐 | `scripts/manual/30_analyze_development.sh` |
| 工程耗时与搜索触发 | `scripts/07c_evaluate_runtime.py` | 同上 |
| 定位/贡献可视化 | `scripts/08_visualize.py`、`08b_visualize_contributions.py` | 同上 |
| 代码和产物审计 | `scripts/09_audit.py` | 同上 |
| 结果 Markdown | `scripts/10_build_report.py`，输出 Markdown 与同名 `.md.index.json` | 同上 |
| 冻结与封存门 | `scripts/13_freeze_config.py`、`14_validate_frozen_config.py`、`15_open_sealed_test.py` | `scripts/manual/40_freeze_after_development.sh`、`50_run_sealed_once.sh` |

`scripts/manual/run_all_development.sh` 不接受位置路径参数，并从 `configs/racpith_v1.json::paths` 同时解析数据根、`./outputs/prepared` 和 `./outputs/development`；标准命令固定为 `bash scripts/manual/run_all_development.sh`。冻结与 sealed 入口同样从输入配置解析 prepared/output 根，命令行不再重复声明这些目录。总控只运行 development 链并明确不打开 sealed_test。所有这些 shell/Python 入口只能由用户在生产 conda `py311` / Python 3.11 环境中手动执行；静态审阅、本地文档更新或源码浏览不得隐式运行、导入或加载数据。几何主线当前固定为 NumPy/SciPy float64 CPU 实现，并以多进程在 crop 层并行；这避免 CPU 与 GPU 形成两套数值算法。只有后续存在经验证的批量张量瓶颈时，才允许在完全相同目标函数和 replay 容差下增加显式 opt-in 的 PyTorch `cuda:0` 加速。

手动总控脚本应：

1. 明确激活 conda py311；
2. 默认从同一配置解析数据根目录和输出目录，只允许显式、可审计的临时覆盖；
3. 默认并固定使用 float64 CPU 几何实现；
4. 当前不启用 GPU；只有另行完成 CPU/cuda:0 数值等价审计后，才允许增加 PyTorch GPU0 加速路径；
5. 任一阶段失败时保留状态并继续汇总其他样本；
6. 不自动修改原始数据；
7. 不在开发环境中隐式运行生产数据；
8. 允许分别运行 prepare、core、contribution、analyze、visualize、audit；
9. 所有正式表图从同一 run_id 和 config hash 派生。

静态审计可以只读源码与文档而不执行代码。任何 synthetic fixture、测试、导入、数据加载、调试或正式实验都只在用户指定的生产 conda `py311` 环境中手动执行；不得用当前静态审阅环境的试跑冒充生产验证。

正式 `07_evaluate.py` 同时要求 `--result-index` 与 `--evidence-index`：前者限定合法定位文件，后者只服务于固定 GT 支持评分和保存目标复核。正式 `07b_evaluate_contributions.py` 必须显式传入 `--crop-manifest ./outputs/prepared/crops/crop_manifest.jsonl`，并输出 `contribution_evaluation_index.json`；不能仅凭 contribution 目录决定评估分母。`12_evaluate_baselines.py` 同样要求 evidence index，并输出 `baseline_denominator.json`。`08_visualize.py` 必须验证 `denominators.json` 中登记的 crop manifest、evidence index、result index、uncertainty index 和 target-domain index 哈希，绘图过程不得重新拟合。`08b_visualize_contributions.py` 输出 `contribution_visualization_index.json`；`07c_evaluate_runtime.py` 输出带运行时 CSV 哈希的 `runtime_denominator.json`；`10_build_report.py` 只在三份 `racpith.combined_audit.v1` 均为 PASS 后输出封存报告和不可省略的 `.md.index.json` sidecar，正文写入审计摘要，sidecar 绑定三份审计文件及全部实际输入索引。

---

## 19. 最终结论框架

最终报告必须依次回答：

1. **数据有效性**：树级 split、坐标和 lineage 是否可信；
2. **有限估计**：c0 与 VarPro 是否数值稳定并优于公平基线；
3. **远场诚实性**：不可辨识时是否避免伪有限点；
4. **状态质量**：六态是否与支持区和退化趋势一致；
5. **精度与安全**：POINT 的 P90、False-POINT、catastrophic 和 coverage 如何；
6. **适用域**：径向汇聚中心何时能代表生物学髓心；
7. **不确定性**：支持区是否得到独立 tree 校准；
8. **贡献**：哪些弧在 GT 下有正、负或不确定影响，哪些弧承担方向、距离或排模态角色；
9. **无 GT 解释**：CF 和符号概率是否可靠，何时拒答；
10. **工程代价**：Core/Profile/Audit/Contribution 的耗时和资源；
11. **结论强度**：VALIDATED、RESEARCH_ONLY、EXPLORATORY 或 FAIL；
12. **最终输出**：始终保持 All-Arc，不启用 Safe-Prune。

完整评价链为：

\[
\boxed{
\text{树级数据合同}
\rightarrow
\text{连续证据守恒}
\rightarrow
\text{凸初值与鲁棒有限精化}
\rightarrow
\text{全方向远场与条件距离审计}
\rightarrow
\text{六态及状态化不确定性}
\rightarrow
\text{target/model risk}
\rightarrow
\text{层级 exact contribution}
\rightarrow
\text{无 GT CF 校准或拒答}
\rightarrow
\text{tree-level sealed 结论}
}
\]

---

## 参考来源

1. [RAC-Pith v2 主方案：pith_localization_integrated_final.md](./pith_localization_integrated_final.md)。
2. [UruDendro4 官方数据集（Zenodo DOI 10.5281/zenodo.15653340）](https://doi.org/10.5281/zenodo.15653340)。

旧文件 ArcPith_GT_v4_Stagewise_Result_Analysis_and_Conclusion_Plan.md 仅作为分析纪律参考；本规范已依据 RAC-Pith v2 的凸共同中心、二维 VarPro、解析远场扫描、条件 profile、六态和冻结 exact contribution 重新设计，不是对旧稿的改名。
