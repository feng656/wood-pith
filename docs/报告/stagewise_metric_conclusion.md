# ArcPith-GT v4 分阶段指标结论

- 固定 manifest 分母：**5182 crops**
- 独立 biological trees：**4** (`T0, T2, T4, T6`)
- 物理尺度：**缺失 mm_per_pixel，所有物理毫米指标不可计算**
- 目标域：**TARGET_UNKNOWN，不能认证局部公共径向中心等同生物学髓心**
- Safe-Prune：**OFF**

| Stage | 数据集级结论 | 核心已满足项 | 阻止正式 PASS 的证据缺口 |
|---|---|---|---|
| 0 | **PASS_WITH_RISK** | 坐标与 lineage 硬合同 | 目标域认证、变换等变 replay、mm scale |
| 1 | **EXPLORATORY** | 连续弧数值不变量 | 重复标注 calibration、overlay/平滑与证据守恒 replay |
| 2 | **EXPLORATORY** | 六源 seed provenance | 同目标 reference basin、消融与 loss map |
| 3 | **EXPLORATORY** | 部分 crop 的 certificate | 585 个 certificate 失败，合计 591 个不可几何解释；独立 reference search |
| 4 | **EXPLORATORY** | 状态记录与 profile 路由 | 独立 calibration、profile/search replay |
| 5 | **PASS_WITH_RISK** | 状态-执行映射与坐标不改写 | common-bias capture/false-veto 与风险—覆盖验证 |
| 6 | **EXPLORATORY** | exact delete-refit 的大部分 certificate | mm GT 贡献、独立 ranking 与 scale/phase stability |
| 7 | **EXPLORATORY** | F-All 固定与 Safe OFF | 至少 30 棵独立 validation trees、Safe 的 tail/False-POINT/coverage 门 |

## 总结性判定

当前运行证明了该冻结管线可完整生成阶段产物，并保留失败/REJECT 样本；它不构成已校准的生物学髓心定位或 Safe-Prune 部署验证。所有 `EXPLORATORY` 与 `PASS_WITH_RISK` 结论均不得提升为 `PASS`。

各阶段的完整指标表和三层判定位于相应 `stage N grayscale/stage_analysis.md`；其中逐项说明了哪些数值已验证、哪些科学证据缺失，以及这些限制对下游解释的影响。
