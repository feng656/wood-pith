# RAC-Pith v2 修复后手动运行与结果分析手册

> 适用范围：已合并本轮修复后的代码（证据 resume 接受已记录的确定性 FAIL、
> 定位阶段进度输出、开发流程默认并行 `min(nproc,16)`、全链路 `--resume` 断点续跑）。
>
> 运行环境固定为 conda `py311` / Python 3.11，脚本内部统一使用
> `conda run --no-capture-output -n py311 python ...`。
>
> 所有路径默认值只来自 `configs/racpith_v1.json::paths`；`./outputs`
> 相对项目根目录解析。原始数据集全程只读。

## 0. 本轮修复对运行方式的影响

| 修复项 | 位置 | 运行含义 |
|---|---|---|
| 裁剪产物缓存 | `scripts/02_generate_crops.py --resume` | patch 图像、GT-free 标注、髓心坐标（`pith_source_px`/`pith_crop_px`）在首次生成后持久化到 `outputs/prepared/crops/`；缓存命中时不再读取源图、不重新裁剪 |
| 证据 resume 修复 | `scripts/03_build_evidence.py` | 已记录的确定性 `FAIL` 不再触发全量重建；仅当索引缺失或输入漂移时才重建 |
| 定位断点续跑 | `scripts/04_run_localization.py --resume` | 已完成且哈希一致的逐裁剪预测被直接复用；中断后重跑不重复拟合 |
| 进度输出 | `scripts/04_run_localization.py` | 每约 1% 打印 `completed/total`、elapsed 与 ETA |
| 默认并行 | `scripts/manual/10_run_development_core.sh`、`run_all_development.sh` | 默认 `min(nproc,16)` 个进程；并行时自动设置 `OMP_NUM_THREADS=1` 等防线程超额订阅 |
| provenance 归档 | 手动操作 | 旧的 `outputs/development/core/provenance.json` 已归档为 `provenance.archived-20260907.json`，新运行写入新 provenance |

注意：修改任何 `src/racpith/**/*.py`、`scripts/**/*.py`、`scripts/manual/*.sh` 或
`pyproject.toml` 都会改变源码束哈希，旧的 `--resume` provenance 校验会拒绝续跑。
此时必须先归档对应的旧 `provenance.json`（不要删除预测文件），再重新启动。

## 1. 运行顺序总览

```text
00 环境检查
00 数据准备（裁剪缓存，可反复重跑）
10 开发核心（证据 → 定位 → 基线 → 目标参考证据/定位）
15 不确定性 replay
20 正式贡献（ring/arc/subarc 精确删除重拟合）
22 分区敏感性审计
30 统计分析 + 可视化 + 三份审计 + 开发报告
40 冻结（仅当 30 的审计全部 PASS）
25 拟合贡献校准器（冻结后重跑 train 链的输出根上）
50 单次 sealed_test
```

一键等价命令（开发链 00→30）：

```bash
bash scripts/manual/run_all_development.sh
```

以下按分步骤给出，便于观察每个阶段和分析中间结果。

## 2. 分步骤运行

### 步骤 0：环境检查

```bash
mkdir -p ./outputs/prepared
bash scripts/manual/00_check_environment.sh ./outputs/prepared/environment.json
```

预期输出：`environment PASS: Python 3.11, float64 CPU geometry dependencies available`。

### 步骤 1：数据准备（索引 + 分树 + 裁剪缓存）

```bash
bash scripts/manual/00_prepare_data.sh
```

行为说明：

- 重建 `data_index` 与 `split` 清单（用于验证数据集与切分协议）；
- `02_generate_crops.py --resume` 命中既有裁剪缓存时只验证哈希与
  髓心坐标一致性（约十几秒），缓存失效才重新裁剪全部 patch。

预期输出示例：

```text
indexed 102 sections / 24 trees; pith dialect=comma, order=xy
split 24 trees; assignment_sha256=...; leakage_free=True
reused 36753 cached crops from 102 sections; pith coordinates loaded from manifests
```

快速核验缓存：

```bash
wc -l ./outputs/prepared/crops/crop_manifest.jsonl          # 36753 条选中裁剪
wc -l ./outputs/prepared/crops/crop_candidate_manifest.jsonl
```

### 步骤 2：开发核心（证据 → 定位 → 基线 → 目标参考）

```bash
bash scripts/manual/10_run_development_core.sh
```

或显式控制并行数（16 为默认上限，内存约 62GiB 的机器建议 8–16）：

```bash
RACPITH_WORKERS=8 bash scripts/manual/10_run_development_core.sh
```

关键点：

- 所有阶段带 `--resume`；已完成的证据与预测按哈希复用，直接跳到缺口继续；
- 定位阶段持续打印进度与 ETA，例如
  `localization: 1608/26872 crops; elapsed=0.15 h; eta=2.31 h`；
- 单线程全量曾需约 23 天；16 进程下预计十几小时量级，实际以 ETA 为准；
- 中断后直接重跑同一命令即可续跑，不会重复拟合。

### 步骤 3：结构化不确定性（GT-blind replay）

```bash
bash scripts/manual/15_run_development_uncertainty.sh
```

要求步骤 2 的定位 `result_index.jsonl` 已存在。输出
`outputs/development/uncertainty/uncertainty_index.jsonl` 及逐裁剪 sidecar。

### 步骤 4：正负贡献（精确 delete-refit）

```bash
bash scripts/manual/20_run_development_contributions.sh
```

输出 `outputs/development/contributions/contribution_index.jsonl`、逐裁剪
`per_crop/*.jsonl` 与逐组 `minus_results/<crop_id>/*.json`。

`C_GT > 0` 表示保留该弧/子弧使结果更接近 GT；`C_GT < 0` 表示使结果偏离；
正式标签 `gt_label` 须经配对扰动区间越过死区，单次点符号仅为 `gt_point_label`。

### 步骤 5：分区尺度/相位敏感性审计（独立输出根）

```bash
bash scripts/manual/22_run_development_partition_audit.sh
```

预算可用环境变量显式覆盖并需在实验登记中记录：

```bash
RACPITH_PARTITION_AUDIT_REPLAYS=16 \
RACPITH_PARTITION_AUDIT_CROPS_PER_TREE=1 \
bash scripts/manual/22_run_development_partition_audit.sh
```

### 步骤 6：统计分析、可视化、审计与开发报告

```bash
bash scripts/manual/30_analyze_development.sh
```

依次执行目标域评估、定位统计、基线对比、运行时统计、可视化、贡献统计、
贡献图、三份 combined audit，最后生成
`outputs/development/RAC_Pith_development_report.md` 及 `.md.index.json`。

三份审计任一为 FAIL 时脚本以非零码退出且不生成报告；先检查
`outputs/development/audit/` 下的失败明细。

## 3. 结果分析顺序

### 3.1 定位完成度与状态分布

```bash
wc -l ./outputs/development/core/result_index.jsonl   # 应为 27351（26872 PASS + 479 上游失败）
```

```bash
conda run --no-capture-output -n py311 python - <<'PY'
import json
from collections import Counter
from pathlib import Path
rows = [json.loads(l) for l in
        Path("outputs/development/core/result_index.jsonl").open() if l.strip()]
print("status:", Counter(r["status"] for r in rows))
print("state :", Counter(r["state"] for r in rows))
done = [r["runtime_seconds"] for r in rows if r["status"] == "FINISHED"]
done.sort()
if done:
    n = len(done)
    print(f"fit runtime s: p50={done[n//2]:.2f} p90={done[int(n*0.9)]:.2f} "
          f"p99={done[int(n*0.99)]:.2f} max={done[-1]:.2f}")
PY
```

### 3.2 搜索开销分布（确认昂贵分支占比）

```bash
conda run --no-capture-output -n py311 python - <<'PY'
import json, glob
from collections import Counter
c = Counter()
for p in glob.glob("outputs/development/core/predictions/train/*.json"):
    d = json.load(open(p)).get("diagnostics", {})
    c[(bool(d.get("profile_triggered")),
      bool(d.get("compact_audit_required")))] += 1
print("(profile_triggered, compact_required) -> count")
for k, v in sorted(c.items()):
    print(k, v)
PY
```

### 3.3 统计主表（步骤 6 完成后）

按以下顺序阅读：

1. `outputs/development/analysis/dataset_summary.csv`：总体分母与主指标；
2. `outputs/development/analysis/tree_metrics.csv`：tree 级效应（主结论依据）；
3. `outputs/development/analysis/tree_cluster_bootstrap_intervals.csv`：tree 聚类 bootstrap 区间；
4. `outputs/development/analysis/distance_stratum_summary.csv`：按 pith 距离层；
5. `outputs/development/analysis/state_specific_summary.csv`、`risk_coverage.csv`：状态与风险分层；
6. `outputs/development/analysis/runtime_summary.csv`、`search_trigger_summary.csv`：工程成本与搜索触发；
7. `outputs/development/baseline_analysis/baseline_summary.csv` 与
   `baseline_paired_tree_differences.csv`：B5 与 B0–B4 的树级配对差异；
8. `outputs/development/contribution_analysis/*.csv`：ring/arc/subarc 正负贡献统计
   （配合 `contribution_evaluation_index.json` 核对 lineage）。

### 3.4 审计判定

```bash
conda run --no-capture-output -n py311 python - <<'PY'
import json
for name in ("combined_audit", "partition_contribution_audit",
             "target_reference_audit"):
    p = f"outputs/development/audit/{name}.json"
    d = json.load(open(p))
    print(name, "->", d.get("overall_status", d.get("status", "?")))
PY
```

三份全部 PASS 后，`outputs/development/RAC_Pith_development_report.md`
才是可引用的开发报告；定位主结论以 tree-level 效应和 tree-cluster bootstrap
区间为准，crop ECDF 仅作描述。

## 4. 冻结与 sealed_test（仅在开发链全部通过后）

### 4.1 冻结配置

```bash
bash scripts/manual/40_freeze_after_development.sh \
  configs/racpith_v1.json \
  ./outputs/development/DEVELOPMENT_DECISION_RECORD.md \
  racpith-development \
  <CODE_REVISION>
```

### 4.2 冻结后重跑 train 链并拟合贡献校准器

```bash
# 使用冻结配置在新输出根重跑开发链后：
bash scripts/manual/25_fit_contribution_calibrator.sh \
  ./outputs/frozen_train \
  ./outputs/racpith_v1_frozen.json
```

### 4.3 单次 sealed_test

```bash
export RACPITH_CALIBRATOR_MODEL=/path/to/contribution_calibrator.joblib
export RACPITH_CALIBRATOR_METADATA=/path/to/contribution_calibrator.json

bash scripts/manual/50_run_sealed_once.sh \
  ./outputs/racpith_v1_frozen.json \
  RAC_PITH_V1_SEALED_001 \
  ./outputs/development/DEVELOPMENT_DECISION_RECORD.md
```

sealed_test 只允许相同 `run_id + config_hash` 中断续跑一次；provisional
配置无法打开。

## 5. 常见运行问题的处置

| 现象 | 原因 | 处置 |
|---|---|---|
| `FileExistsError: refusing to overwrite derived crop artifacts` | 旧的裁剪保护路径（修复前） | 确保 `00_prepare_data.sh` 调用传 `--resume`；缓存会哈希校验后复用 |
| 定位 `refusing to resume localization with changed provenance` | 修改了源码/配置，源码束哈希变化 | 归档 `outputs/development/core/provenance.json` 后重跑；若配置内容真的变了，旧预测不可复用，须重拟合 |
| 证据阶段突然全量重建 | 索引缺失、清单或内容哈希漂移 | 检查 `evidence_index.jsonl` 是否被删除/移动；确认 `crop_manifest.jsonl` 未变 |
| 单样本 `CRASH` 行 | 单个裁剪拟合异常（默认不中止） | 汇总进 `result_index.jsonl`；审计阶段统一判定；需要严格模式时加 `--fail-on-any-error` |
| 希望从头重算某阶段 | 缓存/断点过于保守 | 删除对应阶段的输出目录或索引文件后重跑同一命令；不要混用不同 run_id 的目录 |
