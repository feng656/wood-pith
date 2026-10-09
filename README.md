# 树髓定位（Tree Pith Localization）

在木材横截面图像上定位**髓心（pith，树木生长中心）**的完整实验仓库，含 4 条实验线：传统几何方法对比、自研 ArcPith 年轮弧联合反演管线、分层评价与子弧可解释性、深度学习（2D-OA-Pith）。

> 全部实验的总纲见 [树髓定位实验总体整理.md](./树髓定位实验总体整理.md)。

## 四条实验线速览

| # | 实验线 | 技术路线 | 核心结果（中位误差） |
|---|---|---|---|
| 1 | 传统几何方法对比 | 圆拟合 / 同心回归 / H4 混合策略 / CNN | **90 px**（H4 最优混合） |
| 2 | ArcPith 几何管线 | 7 阶段年轮弧联合反演（OC_ArcPith_RG） | 分阶段验证 EXPLORATORY / PASS_WITH_RISK |
| 3 | 分层评价 + 子弧解释 | 四类分组、三层指标、子弧几何解释、delete-refit 实测 | 总体 **74.7 px**；髓心在图内 **16.8 px** |
| 4 | 2D-OA-Pith 深度学习 | 几何 arc 提取 + 学习头 + 联合后验收尾 | 150 轮收敛；24 张 val 接受 1 张（166 px，保守拒识） |

- 可宣称精度：髓心在图内 / 近髓时中位误差 **16.8 / 24.9 px**（约 1%~1.3% 截面半径）；
- 误差随髓心远离图像单调上升（远髓 193.1 px）——近直线年轮弧上髓心反演数学病态；
- 子弧级可解释性的诚实结论：单个 10% 子弧删除不改变定位结果（贡献 <1 px），可测信号只在环级；
- 数据集无 mm/px 物理标定，所有误差以像素报告。

## 仓库结构

```
├── 树髓定位实验总体整理.md   # 实验总纲（4 条实验线全部结论/方法/指标）
├── 数据文件/       # 各实验线全部指标 CSV/JSON（含 5182 crops 逐条明细）
├── 示例图/         # 解释图、全量可视化、训练曲线示例
├── docs/
│   ├── 报告/       # ArcPith v5-slim 验证报告、执行清单、去留决议等
│   └── 髓心最终结果报告/   # 误差分层评价与子弧段解释三份报告
├── 实验记录_2026-08-18/   # 2D-OA-Pith 接入真实数据的全程记录
├── mokume_audit/   # 第三方 U-Net 权重审计（无法复现年轮检测，F1=0.072）
└── code/
    ├── pith-local/     # 分层评价 + 子弧解释工程（scripts 20-26、src/racpith）
    ├── OC_ArcPith_RG/  # 7 阶段年轮弧联合反演管线（oc_arcpith_rg 包）
    ├── 2D-OA-Pith/     # 深度学习髓心反演工程（含 38 MB 适配数据）
    └── wood-block/     # 实验线 1 传统方法对比脚本（含老师代码 teacher-code/gpt_ring_recon_v5.py，经授权发布）
```

> 路径对照：总纲附录中的 `代码/pith-local/pith-local/` → 本仓库 `code/pith-local/`；`OC_ArcPith_RG_72h_algorithm_package_adjusted 2/` → `code/OC_ArcPith_RG/`。

## 数据

- 主数据为公开数据集 **UruDendro4**（CC BY 4.0，4 棵树 102 截面，年轮标注 + 髓心真值）；
- `code/2D-OA-Pith/data/` 内含适配后的 manifest（约 38 MB），可直接复现训练链路；
- 模型权重：2D-OA-Pith 150 轮 `best.pt` 通过 [GitHub Releases](../../releases) 发布；
- ArcPith 全量 5182 crops 的图像本体未入库；68 万条子弧解释记录（628 MB）未入库，需要请联系作者。

## 许可证

代码与文档：[MIT](./LICENSE)。数据部分遵循 UruDendro4 的 CC BY 4.0 许可证（Marichal et al. 2024/2025）。
