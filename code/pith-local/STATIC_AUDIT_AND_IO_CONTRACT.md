# RAC-Pith v2 代码静态审计与输入输出合同

> 审计性质：本文件记录数学、接口、维度、状态机和数据流的静态推理。遵照任务约束，本轮没有加载 UruDendro4、没有导入生产依赖、没有运行或调试任何脚本，因此不能把下述内容表述为运行通过。

## 1. 端到端数据流

| 阶段 | 主输入 | 主输出 | 不允许进入的量 |
|---|---|---|---|
| 数据索引 | 原始 JPG、LabelMe JSON、pith 文本 | `source_manifest.jsonl`、`dataset_audit.json` | 模型结果 |
| 分树 | source manifest | `tree_split.jsonl`、`section_manifest.jsonl` | crop、预测误差 |
| 裁窗 | section manifest、原始图和边界折线 | crop PNG、`racpith.crop_annotation.v1`、crop manifest | 任何预测结果 |
| 连续证据 | crop RGB、crop 年轮弧 | `racpith.evidence.v1` JSON+NPZ | GT、候选中心 |
| 定位 | evidence、冻结配置 | `racpith.locate.v1` | GT、测试误差、贡献标签 |
| exact contribution | evidence、定位 result index、不可变 All-Arc 结果、删除组 | minus 结果、`racpith.contribution.v1` | 贡献反馈后的新坐标 |
| partition sensitivity | 注册尺度/相位、结果盲 tree/crop 子样本 | 独立 contribution index、稳定性表 | 主贡献标签、定位配置、All-Arc 坐标 |
| 评估 | manifest、定位/贡献结果、GT | crop/tree/dataset 表和图 | 新的拟合或候选选择 |

估计器入口固定为：

```python
RacPithEstimator.fit(evidence, active_mask=None) -> LocateResult
```

入口没有 GT 参数。`active_mask=True` 表示保留；删除仅对原数组作布尔选择，保留节点的 `base_weight` 不变化。

## 2. 坐标合同

- 数据根目录、输出根目录及 prepared/development/sealed 子目录只由 `configs/racpith_v1.json::paths` 提供默认值；`./outputs` 按项目根目录解析。split 配置不重复保存数据路径，显式覆盖会写入数据与 crop audit。
- LabelMe 点保持 `(x, y) = (column, row)`；访问图像数组时才使用 `image[y, x]`。
- LabelMe 的 `imageWidth/imageHeight` 若为缺失、`null` 或空白字符串，视为无尺寸元数据并以实际解码图像为权威；任何非空声明必须是正整数且与实际图像一致，年轮点始终按实际图像尺寸做越界检查。
- LabelMe 的 `imagePath` 若为空、`null` 或缺失，使用已由文件 stem 严格 join 的 section 身份；非空 `imagePath` 必须与 section ID 匹配，不能覆盖文件级 join。
- crop 矩形使用半开区间 `[x0,x1) × [y0,y1)`。
- `crop_xy = source_xy - (x0,y0)`。
- `D_FOV = hypot(crop_width,crop_height)`。
- `norm_xy = (crop_xy - (width/2,height/2)) / D_FOV`。
- 图外 GT 和图外候选不做 clip；JSON 中无穷远用状态和 `null` 上界表达，不写 IEEE `NaN/Infinity`。
- 未提供可靠物理标定时，毫米字段为空，只报告 px 与 `D_FOV` 归一化量。

生产审计必须验证 source→crop→source 往返、图像尺寸、x/y 顺序、pith overlay，以及同一 tree/section 不跨 split。

## 3. 数据适配推理

官方样本 stem 为 `TX_BY_NZ_W`，因此：

```text
section_id = T{treatment}_B{block}_N{tree}_{height}
tree_id    = T{treatment}_B{block}_N{tree}
```

`N` 不能单独作为树 ID。索引器严格对 image/annotation/pith 三方 stem 做一一 join；重复、缺失、多余或解析歧义均失败。

公开资料没有给出 `pith_location.txt` 的具体分隔符和坐标顺序。因此解析器只接受有限白名单语法，并允许 `auto|xy|yx`。`auto` 只有在图像边界和最内环几何使唯一顺序成立时才接受，否则要求生产运行显式给定，不能猜测。

年轮 crop 使用折线段与矩形边界相交。若改为填充 polygon 与矩形相交，会把 ROI 边界误生成为年轮，属于硬错误。

## 4. 连续证据与质量尺度

每个 visible arc 独立拟合轻度平滑样条，真实缺口不跨越。求积区间的边界合并了预注册 subarc 尺度和相位边界，因此一个求积单元不会横跨两个删除块。

对 parent ring `r`：

```text
base_mass_i = (1 / R) * ds_i / sum_{j in ring r}(ds_j)
```

所以每条有效 parent ring 的总质量为 `1/R`。插值加密、顶点复制和 fragment 重切不应改变总质量。删除组 `G` 后：

```text
remaining_mass = original_mass - removed_mass
```

不得重新归一化为 1。

图像质量只由当前 crop、标注位置、梯度/结构张量和局部异常得到，与候选中心无关；它只映射到冻结的 `sigma_x`，不再重复乘入证据权重。`sigma_x` 的单位为归一化长度，`sigma_alg` 的单位为归一化长度平方。

## 5. 核心数学审计

### 5.1 凸初值

对每条 active parent ring 重新计算均值：

\[
a_i=2(x_i-\bar x_r),\qquad
b_i=\|x_i\|^2-\overline{\|x\|^2}_r.
\]

IRLS 法方程中的精度权为：

\[
\frac{\mu_i}{\sigma_{alg,i}^2}
\frac{1}{\sqrt{1+(z_i/\delta)^2}}.
\]

病态时使用特征分解伪逆并记录 rank/弱轴，不添加隐含中心先验。

### 5.2 切向初值

实现约束为：

\[
t_i^T(c-x_i)=0.
\]

切向初值只是第二 seed 和一致性诊断；不能误写为法向零投影。

### 5.3 VarPro

固定中心时，每条 ring 的半径由单调一维方程的 bracketed root 求解，而非均值或中位数替代：

\[
\sum_i\mu_i\frac{\psi((d_i-R)/\sigma_i)}{\sigma_i}=0.
\]

外层二维梯度依据 envelope theorem，不需要对已 profile 的半径求导。外层优化无物理搜索框；极大半径只作为数值逃逸诊断，不能作为 POINT。

### 5.4 解析远场

每个样本都计算：

\[
J_\infty(u)=\sum_r\min_m\sum_{i\in r}\mu_i
\rho\!\left(\frac{u^Tx_i-m}{\sigma_i}\right).
\]

它不是用任意大有限半径点代替。角度网格加倍 replay 不稳定、有限—远场间隔不足、profile 未覆盖竞争方向或 compact 审计触及有限 radius cap 时，都不能签发 POINT。

### 5.5 状态优先级

```text
invalid/search inadequate -> REJECT
multiple persistent components -> MULTIMODAL
one component touching infinity, oriented -> RAY
one component touching infinity, unoriented -> AXIS
closed finite but wide -> RANGE
closed finite, single, tight, replay-stable -> POINT
```

模型风险和目标域是并列字段，不得把 HIGH_RISK POINT 偷换为 RANGE。默认配置仍标记 provisional，所以即使几何为 POINT，`production_usable` 也保持 false，直至训练阶段冻结并通过独立验证。

## 6. 贡献数学与数据流审计

每个 ring/arc/subarc 组都从同一 EvidenceBundle 生成 mask。minus 结果调用同一个冻结估计器，重新运行凸初值、切向 seed、VarPro、远场扫描、条件 profile、compact fallback 和状态机。

只有 full 和 minus 都为 POINT 时计算：

\[
C_G^{GT}=\|c_{-G}-c^*\|-\|c-c^*\|.
\]

平方贡献用两条独立表达式计算并在产物审计中核对：

\[
\|c_{-G}-c^*\|^2-\|c-c^*\|^2
=2\delta_G^T(c^*-c_{-G})-\|\delta_G\|^2.
\]

`C>0` 表示保留该组使结果更接近 GT；`C<0` 表示保留该组使结果偏离 GT。若配对扰动区间未越过死区，只能输出 NEUTRAL、UNCERTAIN 或 abstain。

冲突量：

\[
K_G=J_{-G}(c)-J_{-G}(c_{-G})\ge 0.
\]

若低于数值容差，说明 minus 重求解或评分不一致，不能简单 clip 为 0。位移、冲突和方向信息均不直接决定正负号。

无 GT 的 `C_CF` 使用未参与两次拟合的完整 witness rings；witness 几何单一、训练状态无效、fold 不足或符号不稳时拒答。三分类校准器只使用无 GT 特征，并按 tree 做 out-of-fold 概率校准。

## 7. 预置但本轮未运行的测试

| 测试 | 预期不变量/行为 |
|---|---|
| 理想同心圆 | 凸恒等式恢复共同中心；VarPro 梯度接近零 |
| 平移/旋转/统一缩放 | 映回源坐标后等变 |
| 点复制/重采样 | 预算、目标和结果不变 |
| fragment 拆分/拼接 | parent 总质量和结果不变 |
| 图外 pith | 坐标不被裁到 ROI 边界 |
| 平行短弧/直线极限 | 不签发 POINT |
| 大半径序列 | `J_geo(Ru)` 收敛到解析 `J_infinity(u)` |
| 半径一阶条件 | 每条 ring 的 robust score 接近 0 |
| envelope gradient | 与中心有限差分一致 |
| polygon 裁窗 | 不产生矩形边界伪弧 |
| exact deletion | 保留权重逐元素不变；全搜索被重新调用 |
| 贡献恒等式 | direct `C_sq` 与 identity 一致 |
| 非 POINT minus | 不输出物理 signed contribution |
| JSON roundtrip | 无 object array、pickle、NaN 或 Infinity |
| split | train/test tree_id 与 section_id 交集为空 |

## 8. 当前能与不能声称的内容

基于源码结构和公式，可以声称已提供完整的生产实现路径、显式 I/O 合同、fail-closed 数据适配、核心算法和审计代码。

本轮不能声称：

- UruDendro4 的实际文本 dialect 已识别；
- 依赖能够成功导入；
- 任一数值优化已收敛；
- 阈值已经校准；
- CPU/GPU 一致性已验证；
- 定位误差、状态覆盖或贡献精度达到任何数值；
- 任何测试已经通过。

这些结论必须由生产端 `conda py311` 中的手动运行、审计报告和 tree-level sealed-test 结果给出。

## 9. 本轮静态链路复核新增项

在端到端复核中发现并已从源码结构上修正以下问题；这里只能称“已修改”，不能称“运行通过”：

| 原问题 | 修正后的合同 |
|---|---|
| crop 路径存在但内容可能在 manifest 生成后漂移 | `crop_manifest`/full-reference manifest 的已物化记录逐条保存 `crop_image_sha256` 与 `crop_annotation_sha256`；evidence 读取前重算并拒绝不一致。候选清单含未物化候选，不伪称每行都有产物哈希 |
| crop annotation 曾携带 pith GT | `racpith.crop_annotation.v1` 已删除所有 pith 字段；GT 只留在 crop/reference manifest |
| evidence 未传播可靠 ring order | evidence metadata 显式保存 `ring_order_reliable` 与 `ring_order_by_id`；只有可靠时才允许 AXIS 定向为 RAY |
| evidence 与定位 config 可能错配 | evidence index 和 metadata 均携带 config hash；定位器拒绝 hash 不同的 evidence |
| 定位结果未绑定 evidence 内容 | evidence JSON/NPZ 分别计算 SHA-256；每个定位结果记录两个 hash，resume 和 artifact audit 均核对 |
| contribution 根目录中的 index 会被误读成逐组记录 | loader/audit 只接受 `racpith.contribution.v1` schema |
| 默认零 replay 会使全部正式贡献标签为 UNCERTAIN | 正式配置提供配对 replay；同时把 `gt_point_label` 与区间稳定的 `gt_label` 分开，点符号不能冒充正式标签 |
| 单 crop 失败会中止完整分母 | 03/04/05/05b 默认保存失败并继续；仅显式 `--fail-on-any-error` 才提前失败；最终由 audit 统一判定 |
| 结果只给归一化坐标 | 定位 JSON 同时给 norm、crop px、source px 和 range px；artifact audit 检查 roundtrip |
| 不确定性未进入最终统计状态 | structured-replay sidecar 给出 `adjudicated_state`；S4 评估使用该状态并记录原始 `geometry_state` |
| `Code,cx,cy` 表头未覆盖 | pith parser 的 sample-key 白名单加入 `Code`，仍保留歧义即失败策略 |
| resume 可能混入另一 run 的预测 | 定位 resume 同时核对 run_id、config、evidence index、源码束和逐 crop evidence hash |
| 相关曲线扰动后坐标字段不一致 | replay 同步更新 norm/crop-px 点、切向与 algebraic sigma，并重新执行 EvidenceBundle 合同 |
| formal contribution 曾可扫描到旧 JSONL | 校准、统计、可视化只读取 contribution index 注册且内容 hash 匹配的文件，并核对 config hash |
| 单一 10%/phase-0 分区无法验证切分稳定性 | 独立结果盲 partition audit 覆盖 5%/10%/20% × 两相位；同一弧位置匹配后报告不稳定和正负翻转 |
| 基线均值没有配对不确定区间 | B0–B5 保留 manifest×method 全分母，并以 biological tree 配对 bootstrap B5−comparator |
| 工程代价没有统一分母 | evidence/core/uncertainty/contribution 统一记录单 worker elapsed time，运行时脚本报告失败率与 P50/P90/P95 |
| 冻结仅绑定配置而未绑定可执行实现 | freeze 同时绑定完整源码束、decision record、local/reference manifest 和 crop audit 哈希 |
| compact 无穷远的正反方向曾可能被数成两个模态 | 仅在无穷远边界识别对跖方向为同一 projective axis；有限的正反中心仍保持分离，之后仅由可靠 ring order 将 AXIS 定向为 RAY |
| 最低节点补点曾可能替换注册分区边界 | 只向既有 quadrature 边界并集增补均匀节点，绝不删除 5%/10%/20% 与 phase 边界 |
| evidence 两文件写入中断可留下半成品 | NPZ 与 JSON 各自使用同目录临时文件、flush/fsync 和原子 replace；index 仅在二者完成后记录内容哈希 |
| 相同 evidence 重建时 NPZ 的 ZIP 时间戳导致 hash 漂移 | NPZ 固定成员顺序、ZIP 时间戳、权限和压缩级别；相同数组产生逐字节一致归档，载入时在关闭文件前复制全部数组 |
| RANGE/RAY/AXIS 曾可能只按 GT 半径判断覆盖 | 评估在固定 GT 中心用同一 robust VarPro 重新 profile nuisance radii，以 `J(GT) <= global reference + support delta` 判定完整支持；径向命中仅为独立诊断 |
| 正式定位/可视化可能读取同配置的旧预测 | `result_index` 是唯一登记依据；评估核对预测 SHA、evidence lineage，并拒绝目录中任何未注册 locate JSON；可视化再核对评估分母登记的输入哈希 |
| replay/delete-refit 曾按目录寻找 All-Arc baseline | uncertainty 与 contribution 均强制使用定位 result index；sidecar/index/逐组记录绑定 result-index、baseline 和 evidence 内容哈希 |
| CSV 或 PNG 在生成后被替换但仍用于报告 | `denominators.json` 登记正式统计表哈希，`visualization_index.json` 登记 cohort/case-card 哈希；报告生成前复核登记内容 |
| 贡献统计与贡献图缺少独立封存索引 | `contribution_evaluation_index.json` 绑定 config、crop manifest、贡献 index、可选 partition-audit index、source result-index 及所有统计产物哈希；`contribution_visualization_index.json` 绑定相同 lineage 并登记 cohort/overlay PNG 哈希 |
| 工程运行时 CSV 可被替换 | `runtime_denominator.json` 绑定实际 stage indexes，并登记五个运行时/搜索 CSV 的 SHA-256；报告读取这些表时必须先验证登记哈希 |
| Markdown 报告本身未被登记 | `10_build_report.py` 在报告旁写 `*.md.index.json`，记录报告 SHA-256 和全部已使用输入 index/denominator 哈希；报告与 sidecar 必须作为一个封存单元 |
| 基线统计未绑定实际 evidence 内容 | B0–B5 评估要求 evidence index，逐条核对配置、lineage 和双文件哈希，并输出独立完整分母 |
| 一个基线方法异常会连带抹掉其他方法结果 | B0–B5 按方法独立捕获异常；仅失败方法登记 `CRASH`，其余方法照常形成同一 manifest×method 分母 |
| 冻结只核对 manifest 文件而不核对其指向内容 | 冻结、解封校验和 artifact audit 均要求绝对且存在的 crop/reference 文件，并重新计算图像与 GT-free annotation SHA-256 |
| 不确定性、minus 或 PNG 目录残留可混入当前运行 | index 为唯一权威；审计和可视化拒绝未注册、跨 split、路径越界或内容哈希漂移的 sidecar、minus JSON 与 PNG |
| 报告可在局部审计通过前生成 | 手动总控要求主定位/正式贡献、分区敏感性贡献、完整截面目标域三份 combined audit 全 PASS；报告正文列摘要，sidecar 逐份绑定哈希 |

## 10. 结构化不确定性的静态推理

`ring_bootstrap_bundle` 的数组变换为：

```text
source parent ring draw r_k
  -> 复制该 ring 的全部节点
  -> 新 ring_index = k
  -> arc_id 加 bootstrap draw 前缀
  -> w'_i = (w_i / sum_{j in r_k} w_j) / K
```

因此每个 bootstrap draw 的总质量严格为 `1/K`；重复抽到同一源 ring 时不会发生 ID 冲突，也不会让节点数本身改变该 draw 的质量。由于有放回重采样不保存严格生物年序，bootstrap bundle 主动关闭 `ring_order_reliable`，避免用人为抽样顺序签发 RAY。

相关曲线扰动按 visible arc 生成平滑法向位移，并从扰动后曲线重新计算单位切向；`base_weight`、ring/arc lineage 和尺度不变。两类 replay 都不读取 GT。

sidecar 的保守门为：

- 新长尾或新模态：`requires_search_reaudit=true`，adjudicated state 为 REJECT；
- POINT replay 比例不足或经验椭圆过宽：降为 RANGE；
- 点样本不足或协方差非法：REJECT；
- 非 POINT baseline 不通过 replay 升级为 POINT。

经验椭圆使用二维卡方参考尺度，但在独立树覆盖校准前明确标为 stability ellipse，不称统计置信域。

## 11. 完整截面 target-domain 隔离

crop 生成阶段另写一份 GT-free 的 `full_section_reference_manifest.jsonl` 配套曲线标注。正式流程先用相同冻结估计器拟合完整截面的径向汇聚中心；拟合结束后，`06d_assess_target_domain.py` 才在评估层读取 GT，且只接受 manifest 中明确标记为 `analysis_role=TARGET_REFERENCE_ONLY` 的完整截面 reference，计算：

```text
B_target = || full-section radial centre - anatomical pith GT || / D_FOV
```

并按冻结阈值标为 `TARGET_ALIGNED / ECCENTRIC_GT / TARGET_UNKNOWN`。该标签只用于适用域分层，不写回 evidence，不改变局部坐标，也不允许进入贡献无 GT 特征。

冻结记录同时绑定：

- section split manifest；
- local crop manifest；
- full-section reference manifest；
- crop-generation audit；
- crop config hash；
- 独立 `paths` 子配置 hash（数据/输出路径漂移必须重建 prepared；其他 development 阈值变化不误触发重裁剪）；
- freeze 元数据 hash，以及 freeze 元数据之外的配置正文 hash（校验时分别重算）；
- development decision record；
- code revision；
- executable source-tree SHA-256；
- development/sealed tree IDs。

sealed opening marker 只允许相同 `run_id + config_hash` 续跑。

## 12. 静态接口矩阵

| 生产入口 | 输入 schema/文件 | 输出 schema/文件 | 关键拒绝条件 |
|---|---|---|---|
| `00_resolve_runtime_paths.py` | `racpith.config.v1::paths`、项目根目录 | 单个绝对运行路径 | 数据根非绝对、输出落入数据源、子目录逃逸或重名 |
| `00_index_dataset.py` | 原始 UruDendro4 | source manifest、dataset audit | 数量/命名/三方 join/坐标顺序/嵌套失败 |
| `01_split_by_tree.py` | source manifest | tree/section split、inner folds | tree/section 泄漏、分层配置非法 |
| `02_generate_crops.py` | section split、原图/曲线 | crop/reference manifest、GT-free annotation | 输出位于源目录、低前景、ring 不足、路径冲突 |
| `03_build_evidence.py` | GT-free annotation、crop RGB、config | NPZ+JSON、evidence index | 连续弧/维度/有限性/parent budget 失败 |
| `04_run_localization.py` | evidence index、同 hash config | locate JSON、result index、provenance | config/evidence hash、数值、搜索或状态门失败 |
| `05b_run_uncertainty.py` | evidence、定位 result index、immutable baseline | uncertainty sidecar/index | 节点 iid bootstrap 禁止；旧/未注册 baseline 禁止；新尾/模态触发重审 |
| `05_run_contributions.py` | evidence、定位 result index、baseline、manifest GT | contribution records、minus results | result-index/baseline/evidence hash、partition、POINT-to-POINT、重求解失败 |
| `22_run_development_partition_audit.sh` | 同一冻结 baseline、尺度/相位 registry | 独立稳定性 contribution 根 | 结果盲子样本、分区质量守恒、主结果不回写 |
| `06*_contribution*` | train records 或 frozen calibrator | OOF/概率/校准指标 | tree 泄漏、全局或任一训练 fold 缺少三类、config hash/阈值/class order 不同，概率阈值不在 `[0,1]` |
| `06d_assess_target_domain.py` | `analysis_role=TARGET_REFERENCE_ONLY` 的 full reference result、manifest GT | target-domain records | reference 角色、尺度/偏差恒等式、GT 顺序或可靠 POINT 条件失败则拒绝/UNKNOWN |
| `07_evaluate.py` | crop manifest、result/evidence/uncertainty/target-domain index | 定位统计表、`denominators.json` | 缺失结果仍入分母；不允许把 crop 当独立 tree；登记输入/输出哈希不一致 |
| `07b_evaluate_contributions.py` | contribution root/index、**必需 `--crop-manifest`**、可选 partition-audit root/index、config | 贡献统计表、`contribution_evaluation_index.json` | manifest 分母不完整、lineage/config/source-result-index 不一致、未注册或被改写统计产物 |
| `07c_evaluate_runtime.py` | crop manifest、stage indexes、config | 五个运行时/搜索 CSV、`runtime_denominator.json` | stage index 不完整、result-index 绑定不一致、输出哈希漂移 |
| `08_visualize.py` | 登记结果、只读图像路径、定位 denominator | cohort 图、case card、`visualization_index.json` | 绘图不重拟合、不从 PNG 反推数值、输入/输出哈希不一致 |
| `08b_visualize_contributions.py` | contribution index、**必需 `--crop-manifest`**、config | cohort/overlay、`contribution_visualization_index.json` | manifest lineage、source-result-index 或 PNG 内容哈希不一致 |
| `09_audit.py` | config、manifest、result/contribution/uncertainty index 与全部产物 | combined audit | schema/path/content hash/lineage/坐标/恒等式任一硬合同失败则整体 FAIL |
| `10_build_report.py` | 定位/运行时/贡献/图形/baseline 登记索引、三份 combined audit | Markdown、同名 `.md.index.json` | 任一必需输入缺失或哈希不一致、审计重复/畸形/非 PASS、三份中没有恰一份非空 target-domain 审计；报告 sidecar 不能省略 |
| `13_freeze_config.py` / `14_validate_frozen_config.py` | config、decision record、source bundle、crop/reference manifests | frozen config/lock validation | 配置正文/manifest 或其逐条内容哈希漂移、路径非绝对/不存在、源码束或 decision record 漂移 |

手动冻结与 sealed 总控不接收 prepared/sealed 路径：二者必须由输入配置的
`paths` 段解析；冻结配置默认写入 `paths.output_root/racpith_v1_frozen.json`。

预置 tests 现包括：共同中心恒等式、VarPro envelope gradient、解析远场极限、radius 一阶条件、点复制预算不变、删除不重归一化、tree split 不泄漏、GT-free crop annotation、重叠网格和 ring-bootstrap 等预算。遵照用户要求，本轮没有执行这些 tests。

所有 Python、shell、fixture 和真实数据入口都只允许以后由用户在生产 conda `py311` / Python 3.11 环境中手动执行。本轮静态审计不导入模块、不加载数据，也不以本机试跑替代生产验证。当前几何主线仅允许 NumPy/SciPy float64 CPU；GPU 未启用，除非另行在生产 `py311` 完成 CPU/cuda:0 对中心、目标/profile、状态、支持区及贡献标签的等价 replay，并将 GPU 设计为显式 opt-in 的同算法加速路径。
