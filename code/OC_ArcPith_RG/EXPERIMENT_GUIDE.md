# 调整后实验指南：围绕最终目标重新组织

最终目标有三个：

1. **定位**：从局部年轮线反演真实髓心；
2. **可行度**：判断当前 crop 是否支持有限点、仅方向、多个模式，或应拒绝；
3. **贡献度**：量化每条年轮/微弧对定位、可观测性和冲突的贡献。

两组 Step 3 已经说明：不能再把 “GT 不是所有候选中的最低 loss” 统一写成 `NOT_IDENTIFIABLE`。调整后的流程把 Step 3 拆成两类门：**全局可辨识性**与**局部 GT 驻点/模型偏置**。

## Step A — 先验证数据分组

```bash
python tools/validate_manifest.py data/manifest.jsonl
```

必须确认 `tree_id` 是 biological tree。不同 section/cross-section/crop 不能冒充独立树。

## Step B — 新 Step 3：M0 与 M1-RPC 候选筛选

```bash
python experiments/00_screen.py --config config.yaml --manifest data/manifest.jsonl
```

输出：

- `results/step3_v2/candidate_m0.jsonl`
- `results/step3_v2/candidate_m1rpc.jsonl`
- `results/step3_v2/m0_screen.json`
- `results/step3_v2/m1rpc_screen.json`

### B1. 候选平移改成规范尺度

不再固定 10/30/60 px，而采用 crop 规范坐标比例：默认 0.01/0.025/0.05/0.10。这样 400 px 与 1200 px crop 的难度可比较；若有可靠物理标尺，可把这些值改成统一 mm/FOV 比例。

### B2. 两个独立判断

**Global geometry** 只看 direction / distance / reverse / random：

- 若这些也无法排除，才判 `NO_GO_GLOBAL_GEOMETRY`；
- 若 global 几何可辨，但 translation 在 GT 附近更低，输出 `BIAS_REPAIR_REQUIRED`，不再误写为“完全不可辨识”。

### B3. M1-RPC

旧 additive M1 不再作为主修复项。新模型在每个候选髓心下：

1. 将相邻 parent rings 转换成候选中心的极坐标剖面；
2. 在重合角区间插值 `r_k(theta)`；
3. 比较 `d log r_k / d theta`；
4. 加入真实 ring-order 的正序软约束；
5. near-infinity 时自动衰减该项。

这正对准当前失败模式：错误平移会给不同半径的 ring 引入不同的角向畸变，因此不能只靠静态最近邻点对。

## Step C — Local Bias Audit

```bash
python experiments/01_bias_audit.py --config config.yaml --manifest data/manifest.jsonl
```

输出：

- `results/bias_audit/m0_bias.jsonl`
- `results/bias_audit/m1rpc_bias.jsonl`
- `results/bias_audit/repair_decision.json`

对每个 crop，在 GT 周围做二维 coarse-to-fine profile，得到真实局部最优点，而不是只检查有限的 10/30/60 px 候选。

必查：

- `bias_norm / bias_px`；
- radial / tangential bias；
- `loss_gt - loss_local`；
- bias 随 crop size、真实 pith 距离、tree 的规律；
- M1-RPC 是否在 tree-level 至少 70% 的树上把 bias 降低 20% 以上。

若 M1-RPC 不能稳定降低 bias：**不要增加 lambda 强行通过**，保留 M0 为基线，转入更显式的共享形变 nuisance 模型研究。

## Step D — Blind Inversion + tree-group CV

```bash
python experiments/02_blind_inversion.py --config config.yaml --manifest data/manifest.jsonl
```

代码用 biological tree 做 group folds。训练折只用于选择 `m1_lambda_shape`；留出树上重新执行完整多初值 M1-RPC。

输出：

- `results/blind/metrics_m0.jsonl`
- `results/blind/metrics_m1rpc_heldout.jsonl`
- `results/blind/tuning_lambda_*.jsonl`
- `results/blind/g2.json`

G2 只有同时满足以下条件才保留 M1-RPC：

- 至少 4/5 folds 的 finite-point tree median 改善；
- overall median point improvement 达到预注册值；
- far-field axis angle 不明显恶化。

否则使用 M0 baseline，不把“PSD 更大”当成“定位更正确”。

## Step E — 可行度 / Observability

```bash
python experiments/03_observability.py --config config.yaml --manifest data/manifest.jsonl
```

联合 profile 使用 `(phi,kappa)`，其中 `kappa=0` 表示 point at infinity。输出 POINT / AXIS / MULTIMODAL / REJECT。

下一轮主实验应按：

- crop size/FOV；
- true pith distance / FOV；
- visible ring count；
- angular diversity；
- tree

分层统计，而不是把 96 crop 当成 96 个独立生物样本。

## Step F — 年轮贡献度

```bash
python experiments/04_contribution.py --config config.yaml --manifest data/manifest.jsonl
```

对每个 parent ring 删除后重新优化，并且**不重新分配其他 ring 的冻结预算**。输出：

- projective solution shift；
- finite point shift；
- fixed-coordinate local Hessian logdet drop；
- GT error increase when deleted（仅实验期）；
- 后续可扩展到 micro-arc/block。

解释必须拆开：

- **support**：该 ring 自己的观测质量/覆盖；
- **information**：删除后可观测性变差多少；
- **leverage/shift**：删除后解移动多少；
- **GT help**：真实误差变好还是变坏；
- **conflict**：高 leverage 但 GT help 为负，说明其可能把解拉向错误方向。

## 推荐的下一轮实验矩阵

主实验优先采用“策略 B 的多尺度/多位置思想”，但要冻结设计并按树汇总：

- FOV：至少 3 档规范尺度；
- radial distance：inside / near-outside / mid / far；
- 每个 biological tree 每个 strata 尽量等量抽样；
- 同一完整截面的 radial sequence 作为相关序列，不当独立 N。

模型比较顺序：

1. B0 homogeneous/normal baseline；
2. M0 absolute tangent；
3. M1-add legacy（只作消融）；
4. **M1-RPC（本次新增）**；
5. 若 M1-RPC 仍有稳定 translation bias，再研究显式 shared-deformation nuisance；不要直接上 RGB 坐标回归。
