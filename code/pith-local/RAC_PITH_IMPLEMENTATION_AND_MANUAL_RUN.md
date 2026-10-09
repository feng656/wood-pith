# RAC-Pith v2 实现说明与生产端手动运行手册

> 对应方案：`pith_localization_integrated_final.md`  
> 对应评估规范：`pith_localization_integrated_final_result_analysis_plan.md`  
> 目标环境：conda `py311`、Python 3.11  
> 本轮状态：只完成源码与接口的静态推理审计；没有加载 UruDendro4，没有导入/执行/调试项目代码，也没有生成任何实验数值。

## 1. 实现边界

本实现解决两个相互隔离的问题：

1. 从局部 RGB crop 和人工年轮弧定位树髓；
2. 在估计器冻结后，通过完整 delete-refit 判断 ring、visible arc、subarc 对 GT 误差的正、负或中性作用，并同时报告方向、距离、排模态、高杠杆和冲突等功能角色。

坐标始终来自不可变的 All-Arc 估计。贡献分析不会回写权重、删除弧或生成“修正后坐标”；Safe-Prune 默认不存在于主线。

UruDendro4 的官方资料说明该版本含 102 个 Pinus taeda 横切面；本实现按 sample stem 恢复 24 棵 biological tree，并以 `T?_B?_N?` 作为分组单位，而不是把重叠 crop 当独立样本。资料入口：

- [UruDendro 作者仓库](https://github.com/hmarichal93/uruDendro)
- [UruDendro4 Zenodo 记录](https://zenodo.org/records/15653340)

## 2. 数据输入与只读约束

数据路径与输出路径只有一个默认来源：`configs/racpith_v1.json` 的 `paths`：

```json
"paths": {
  "dataset_root": "/root/2026/dataset/UruDendro4/UruDendro4",
  "output_root": "./outputs",
  "prepared_subdirectory": "prepared",
  "development_subdirectory": "development",
  "sealed_subdirectory": "sealed"
}
```

其中 `./outputs` 固定相对于项目根目录解析，与启动命令时所在目录无关。默认目录布局为：

```text
/root/2026/dataset/UruDendro4/UruDendro4/
├── pith_location.txt
├── annotations/annual_rings/*.json
└── images_no_background/*.{jpg,jpeg,png}

PROJECT_ROOT/outputs/
├── prepared/
├── development/
└── sealed/
```

`configs/urudendro4_split*.json` 只保存划分协议，不再重复保存数据路径。

索引器执行严格的 image—annotation—pith 三方 `section_id` join，并检查：

- 官方完整集应为 102 个 section、24 棵 tree；
- 每个 treatment/block cell 应有两棵树；
- LabelMe 中非空的图像尺寸声明、非空 `imagePath`、闭合 polygon 和图像文件一致；官方文件中空字符串形式的 `imageWidth/imageHeight` 与 `imagePath` 视为“未提供”，此时分别以实际解码图像尺寸和已完成 stem join 的文件身份为准，并仍严格检查所有 polygon 点的范围；
- 年轮按面积和嵌套几何得到由内到外顺序；
- pith 文本仅接受明确白名单格式，包括常见的 `Code,cx,cy`；
- 无表头时，`auto` 只有在 `(x,y)` 与 `(y,x)` 中恰有一个通过边界和最内环检查时才接受，否则要求显式给出 `xy` 或 `yx`。

所有派生图像、标注、manifest 和结果写到独立输出目录。原始数据不会被改名、移动或覆盖。

特别地：

- `crop_manifest.jsonl` 保存 GT，供实验分层、贡献真值和最终评价使用；
- `racpith.crop_annotation.v1` 不含任何 pith 坐标；
- evidence NPZ/JSON 不含 GT；
- `RacPithEstimator.fit(evidence, active_mask=None)` 没有 GT 参数。

## 3. 核心实现不是演示版

`src/racpith/numerics.py` 和 `src/racpith/estimator.py` 实现完整主线：

- 连续弧轻度平滑和预注册求积；
- 每条 parent ring 等预算，插值和切段不增权；
- 半径无关的凸共同中心 IRLS 主初值；
- 正确的切向零投影第二初值；
- 每 ring 严格一维 robust-radius 求根；
- 无外部物理框的二维 pseudo-Huber VarPro；
- 解析 `J_infinity(phi)` 全方向扫描及角度加密 replay；
- 条件 compact-distance profile；
- 必要时确定性二维方向—逆距离 compact audit；
- `POINT/RANGE/RAY/AXIS/MULTIMODAL/REJECT` 互斥保守状态机；
- 非有限结果转成 JSON `null`，不写 NaN/Infinity。

`POINT` 不是“优化器返回了一个坐标”的同义词；有限—远场间隔、局部曲率、支持区闭合、profile、模态和搜索加密不通过时，必须输出更保守的状态。

评估结构化输出时，不再用“GT 到 crop 中心的半径是否落入区间”冒充支持区覆盖。`07_evaluate.py` 重新载入哈希绑定的同一 EvidenceBundle，在 **固定 GT 中心** 处只 profile 每条 ring 的半径，计算与定位器完全相同的鲁棒 VarPro 目标：

```text
support_contains_gt := J_profiled(GT) <= global_support_reference + support_delta
```

该步骤不优化 GT 中心、不改变结果、不回写状态，只是事后覆盖评分；保存解也会用同一接口重评分并与结果 JSON 中的 objective 核对。径向区间命中另存为 `radial_interval_contains_gt`，仅作诊断，不能替代二维/投影支持判定。

## 4. 重叠裁窗

`configs/urudendro4_crops.json` 预注册三个相对尺度和 65% overlap。每个完整横切面先生成候选网格，再按：

- 前景比例；
- 至少三条可见 parent ring；
- 连续弧最小长度；
- pith 到 crop 的 inside / near / mid / far 距离层；

进行确定性、结果盲的平衡选择。所有 crop 继承源 tree 的 split。输出同时保留：

- `crop_candidate_manifest.jsonl`：全部候选和拒绝原因；
- `crop_manifest.jsonl`：正式选中分母；
- `crop_section_audit.jsonl`：逐 section 计数；
- `crop_generation_audit.json`：配置、manifest 哈希和总体分母。

曲线裁剪使用源 polygon 折线段与半开 crop 矩形相交，不使用填充 polygon 的交集边界，因此不会把 ROI 边缘伪造成年轮。

## 5. 结构化不确定性

`scripts/05b_run_uncertainty.py` 只做两类 GT-blind replay：

1. 整条 parent ring bootstrap；重复抽到的 ring 会复制并重编号，每个 draw 重新分配相等 parent budget；
2. 沿 visible arc 的相关法向曲线扰动，并同步更新切向。

不做独立节点 bootstrap。输出状态频率、POINT 中心样本、协方差、经验稳定椭圆、新长尾/新模态标志和 `adjudicated_state`。若 replay 发现新长尾或模态，sidecar 标记 `requires_search_reaudit=true`，最终统计不会继续把原始结果当作 POINT。

不确定性入口强制同时提供定位 `result_index.jsonl`；sidecar 和 uncertainty index 保存该索引的 SHA-256，并继续绑定具体 baseline JSON 与 evidence 双文件哈希。因此目录中的旧预测不能成为 replay 基线。

椭圆在独立树覆盖校准前只称“经验稳定域”，不能称为严格 95% 置信区间。

## 6. 正负贡献

对注册组 `G`：

```text
All-Arc = M(E)
Minus-G = M(E \ G)
```

Minus-G 使用同一冻结配置，重新运行初值、radius profile、VarPro、远场、条件 profile、compact audit 和状态机；保留节点沿用原 `base_weight`，删除后不重归一化。

贡献入口同样强制使用定位 result index。每个贡献 index 和逐组记录同时绑定 result-index SHA、具体 All-Arc baseline SHA 与 evidence 双文件 SHA；未登记预测不会进入删除重求解。

只有 full 与 minus 都为 POINT 时，才计算：

```text
C_GT = error(Minus-G, GT) - error(All-Arc, GT)
```

- `C_GT > 0`：保留该弧使结果更接近 GT；
- `C_GT < 0`：保留该弧使结果偏离 GT；
- 死区内：中性。

每条记录同时含：

- `gt_point_label`：单次点估计符号，仅供诊断；
- `gt_label`：配对曲线/GT 扰动区间越过死区后的正式标签；
- `contrib_cf`：由未参与两次拟合的 witness rings 给出的无 GT 外部化贡献；
- `roles`：可辨识性、方向、范围、排模态、高杠杆、冲突或冗余角色。

默认对每棵树按距离层做结果盲的确定性贡献审计子样本，避免对数万 crop、数百删除组和所有 replay 做不可控的指数级计算。设置 `--max-crops-per-tree 0` 可对全部 crop 运行；这只改变预注册计算范围，不改变单个贡献的核心算法。

尺度/相位稳定性不从主分区自我证明。`22_run_development_partition_audit.sh` 在独立输出根目录中，对每棵树结果盲抽取默认 1 个 crop，完整计算 5%/10%/20% × phase 0/0.5 的 delete-refit，并用默认 16 次配对扰动判断同一弧位置的标签是否稳定。该审计关闭重复的 CF 计算，也绝不回写主贡献、All-Arc 坐标或配置。可用 `RACPITH_PARTITION_AUDIT_REPLAYS` 和 `RACPITH_PARTITION_AUDIT_CROPS_PER_TREE` 显式覆盖计算预算，覆盖值必须在实验登记中记录。

## 7. 生产端命令

以下命令只应由用户在已安装依赖的生产端手动执行；本地静态审计、文档审阅或源码浏览都不得隐式触发这些入口，也不得通过导入模块来代替手动运行。生产环境固定为 conda `py311` / Python 3.11，脚本内部使用：

```text
conda run --no-capture-output -n py311 python ...
```

### 7.1 环境检查

```bash
bash scripts/manual/00_check_environment.sh ./outputs/prepared/environment.json
```

### 7.2 一次完成 development 链

在项目根目录直接运行唯一推荐命令：

```bash
bash scripts/manual/run_all_development.sh
```

该命令从同一个 `configs/racpith_v1.json` 同时解析数据路径、prepared 路径和
development 路径，默认分别写入 `./outputs/prepared` 与
`./outputs/development`。总控脚本不再接受位置路径参数：继续使用旧式多路径命令会立即
失败，避免结果写入另一目录。使用另一份完整配置时设置
`RACPITH_CONFIG=/absolute/path/to/config.json`；并行数可用
`RACPITH_WORKERS=8` 设置，坐标顺序诊断可用 `RACPITH_PITH_ORDER=xy|yx|auto` 设置。

例如使用 8 个 worker，仍不重复填写任何路径：

```bash
RACPITH_WORKERS=8 bash scripts/manual/run_all_development.sh
```

底层 Python 入口仍保留显式路径覆盖用于诊断，并在数据/crop audit 中登记
`dataset_root_override_used` 或 `prepared_root_override_used`；带路径覆盖的结果不能进入正式冻结。

若无表头 pith 坐标顺序仍歧义，索引器会失败；检查审计后把最后一个参数明确改为 `xy` 或 `yx`，不能靠猜测继续。

也可逐段运行：

```bash
bash scripts/manual/00_prepare_data.sh
bash scripts/manual/10_run_development_core.sh
bash scripts/manual/15_run_development_uncertainty.sh
bash scripts/manual/20_run_development_contributions.sh
bash scripts/manual/22_run_development_partition_audit.sh
bash scripts/manual/30_analyze_development.sh
```

`run_all_development.sh` 不会冻结配置，也不会打开 sealed_test。

### 7.3 冻结

先完成 development 分析并写出参数取舍记录，再执行：

```bash
bash scripts/manual/40_freeze_after_development.sh \
  configs/racpith_v1.json \
  ./outputs/development/DEVELOPMENT_DECISION_RECORD.md \
  DEVELOPMENT_RUN_ID \
  CODE_REVISION
```

冻结文件记录 decision record、section manifest、local-crop manifest、full-section reference manifest、crop audit、crop config、独立 `paths` 子配置、冻结配置正文、freeze 元数据和全部可执行 Python/shell 源码束的 SHA-256，同时保存 code revision 与 development/sealed tree ID。冻结动作不负责自动调参；它只封存已经由 development 树确定的配置。裁剪只绑定 `paths` 子配置哈希，因此 development 阶段正常调整求解阈值不会要求重新裁剪；但数据或输出路径变化必须重建 prepared 产物。校验时会分别重算这些哈希，配置参数或冻结身份被静默改写会直接失败。

脚本从输入配置解析 `prepared` 与输出根，默认将冻结配置写到
`./outputs/racpith_v1_frozen.json`；只有需要另存冻结副本时，才在命令末尾追加一个
`OUTPUT_CONFIG` 参数。

冻结后，应在新的输出目录用 frozen config 重跑 train 链，再拟合贡献校准器：

```bash
bash scripts/manual/25_fit_contribution_calibrator.sh \
  ./outputs/frozen_train \
  ./outputs/racpith_v1_frozen.json
```

校准器只接受预注册的 tree-level `inner_fold`，要求每个训练 fold 都包含三类 GT 标签；`--minimum-probability` 必须位于 `[0,1]`。应用阶段还会核对 class order、阈值、模型文件 SHA-256 与 metadata，任何不一致都拒绝输出。

### 7.4 单次 sealed_test

```bash
export RACPITH_CALIBRATOR_MODEL=/path/to/contribution_calibrator.joblib
export RACPITH_CALIBRATOR_METADATA=/path/to/contribution_calibrator.json

bash scripts/manual/50_run_sealed_once.sh \
  ./outputs/racpith_v1_frozen.json \
  RAC_PITH_V1_SEALED_001 \
  ./outputs/development/DEVELOPMENT_DECISION_RECORD.md \
  8
```

该脚本从 frozen config 的同一 `paths` 段解析 `./outputs/prepared` 与
`./outputs/sealed`，命令行不再接受这两个目录，避免 prepared 来源和 sealed 输出位置
与冻结配置不一致。也可省略最后的 worker 数并通过 `RACPITH_WORKERS` 设置。

封存脚本会：

- 校验冻结配置、decision record、源码束、split/crop/reference/audit 哈希；
- 创建 one-time opening marker；
- 只允许相同 `run_id + config_hash` 的中断续跑；
- 依次运行 evidence、核心、不确定性、贡献、基线、统计、可视化和审计；
- 分别要求主定位/正式贡献、分区敏感性贡献、完整截面目标域三份 combined audit 全部 PASS，并检查恰有一份审计含非空 target-domain 分母；任一失败都不生成报告；
- 在报告正文登记审计摘要，并在 `.md.index.json` 中绑定三份审计文件和全部实际输入 index/denominator 的 SHA-256；
- 不允许 provisional 配置打开 sealed_test。

## 8. 主要输出

```text
RUN_ROOT/
├── evidence/evidence_index.jsonl
├── core/
│   ├── provenance.json
│   ├── result_index.jsonl
│   └── predictions/<split>/*.json
├── uncertainty/
│   ├── uncertainty_index.jsonl
│   └── per_crop/<split>/*.json
├── contributions/
│   ├── contribution_index.jsonl
│   ├── per_crop/*.jsonl
│   └── minus_results/<crop_id>/*.json
├── contribution_partition_audit/
│   ├── contribution_index.jsonl
│   ├── per_crop/*.jsonl
│   └── minus_results/<crop_id>/*.json
├── analysis/
│   ├── crop_metrics.csv
│   ├── tree_metrics.csv
│   ├── dataset_summary.csv
│   ├── tree_cluster_bootstrap_intervals.csv
│   ├── distance_stratum_summary.csv
│   ├── state_tree_metrics.csv
│   ├── state_specific_summary.csv
│   ├── target_domain_summary.csv
│   ├── runtime_summary.csv
│   ├── stage_crop_runtime.csv
│   ├── stage_tree_runtime.csv
│   ├── localization_search_diagnostics.csv
│   ├── search_trigger_summary.csv
│   ├── risk_coverage.csv
│   ├── denominators.json
│   ├── target_domain_tree_metrics.csv
│   ├── target_domain_crop_descriptive.csv
│   └── runtime_denominator.json
├── baseline_analysis/
│   ├── baseline_summary.csv
│   ├── baseline_paired_tree_differences.csv
│   └── baseline_denominator.json
├── contribution_analysis/
│   ├── contribution_evaluation_index.json
│   └── *.csv / contribution_evaluation_status.json
├── figures/
│   ├── visualization_index.json
│   ├── cohort/
│   └── samples/
├── contribution_figures/
│   ├── contribution_visualization_index.json
│   ├── cohort/
│   └── overlays/
├── audit/
│   ├── combined_audit.json
│   ├── partition_contribution_audit.json
│   └── target_reference_audit.json
├── RAC_Pith_*_report.md
└── RAC_Pith_*_report.md.index.json
```

主结论以 tree-level effect 和 tree-cluster bootstrap 区间为准。crop ECDF 只作描述；数千重叠 crop 不会被伪装成数千独立样本。

裁窗层先建立内容绑定：`crop_manifest.jsonl` 和 `full_section_reference_manifest.jsonl` 的每条已物化记录都保存 `crop_image_sha256` 与 `crop_annotation_sha256`。前者绑定实际 crop PNG，后者绑定不含 GT 的 `racpith.crop_annotation.v1`；完整截面 reference 的图像哈希绑定只读源图，标注哈希绑定新生成的 GT-free reference annotation，且 reference 行固定标记 `analysis_role=TARGET_REFERENCE_ONLY`。evidence 构建会在读取前重新计算两者并拒绝内容漂移，不能只相信路径或总 manifest 哈希。候选清单包含未物化候选，因此不把所有候选误写成具有这两个产物哈希。

各 stage index 均保存配置哈希和单 worker 耗时；定位 index 额外保存预测内容哈希，贡献/不确定性 index 保存 sidecar 或记录哈希。正式定位统计、目标域、基线和可视化都以 index 为登记依据，并核对 config、tree/section/split lineage 与内容哈希；目录扫描只用于发现并拒绝未注册旧文件。分析表写完后由 `denominators.json` 登记定位统计输出哈希；`runtime_denominator.json` 另行登记 `stage_crop_runtime.csv`、`stage_tree_runtime.csv`、`runtime_summary.csv`、`localization_search_diagnostics.csv` 和 `search_trigger_summary.csv` 的 SHA-256，并绑定实际使用的各 stage index。运行时 P50/P90/P95 排除 `RESUMED` 和未执行行，同时仍把失败、跳过与缺失保留在成功率分母中。

贡献链也不以目录扫描为权威。`07b_evaluate_contributions.py` 必须显式接收 `--crop-manifest ./outputs/prepared/crops/crop_manifest.jsonl`，其 `contribution_evaluation_index.json` 绑定 config、crop manifest、正式 contribution index、可选 partition-audit index、唯一 source result-index SHA，以及每个贡献统计表/状态文件的内容哈希。`contribution_visualization_index.json` 绑定同一 config、crop manifest、contribution index 和 source result-index SHA，并登记 cohort PNG 与逐 crop overlay 的哈希。最终 `10_build_report.py` 仅接受外层与内层 artifact audit 都为 PASS、且 finding 结构合法的 `racpith.combined_audit.v1`；手动总控固定传入主定位/正式贡献、分区敏感性贡献、完整截面目标域三份审计。脚本在报告旁生成 `RAC_Pith_*_report.md.index.json`，登记报告自身 SHA-256，以及三份审计、`denominators.json`、`runtime_denominator.json`、两个 visualization/evaluation index 和可选 baseline denominator 的输入哈希；没有该 sidecar、三份审计未全部通过或任一登记输入不一致时，不得把 Markdown 当作已封存报告。

冻结与审计都会逐条重新计算 crop/reference 图像和 GT-free annotation 的当前内容哈希，并要求路径为绝对且文件存在；不是只核对 manifest 文件自身。B0–B5 每个方法独立捕获异常，一个基线失败只登记该方法为 `CRASH`，不会伪造其余方法失败或破坏 manifest×method 完整分母。

Evidence NPZ 使用固定 ZIP 元数据和固定数组顺序生成，因此相同数组可得到逐字节一致的 SHA-256；载入时所有数组在关闭归档前复制到内存。`03_build_evidence.py --resume` 只复用范围、配置、lineage、文件哈希和 bundle 身份全部一致的完整索引；真实构建异常记为 `FAIL`，清单预先判定不适用单独记为 `MANIFEST_INELIGIBLE`。

## 9. CPU、GPU 与并行

当前生产主线固定使用 NumPy/SciPy float64 CPU，GPU 路径未启用。原因是核心只有二维中心和一维 radius/profile，主要收益来自 crop 级多进程，而不是把小型优化器迁移到 GPU。这样也避免 CPU 与 GPU 出现目标函数或精度不同的隐形算法分叉。

生产端通过 `WORKERS` 参数并行 crop。不得仅因机器存在 GPU 或环境安装了 PyTorch 就自动切换设备。只有后续确认图像特征或大批 profile 成为瓶颈，并在生产 `py311` 中完成 CPU/cuda:0 对中心、目标/profile、状态、支持区和贡献标签的数值一致性 replay 后，才允许新增显式 opt-in 的 PyTorch GPU0 加速；GPU 只能加速同一算法，不能形成另一套目标函数、精度默认值或状态门。

## 10. 当前不能声称的结果

因为本轮没有在生产 `py311` 环境运行，所以当前不能声称：

- UruDendro4 实际 `pith_location.txt` dialect 已被成功解析；
- 任一依赖已成功导入；
- 任一数值优化已收敛；
- 任一 fixture 或测试已通过；
- provisional 阈值已经校准；
- 定位误差、覆盖率、False-POINT 或贡献分类达到任何数值。

只有生产端运行产生的三份 `combined_audit.json` 均为 PASS，且冻结 sealed-test 报告满足预注册门时，才允许形成效果结论。当前 `configs/racpith_v1.json` 明确是 provisional；即使几何状态为 POINT，也不会被标为生产可用。
