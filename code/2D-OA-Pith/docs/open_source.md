# 研究与开源代码审计（截至 2026-08）

在截至 2026-08 的本次公开论文与官方/作者仓库检索范围内，未发现一个现成项目同时处理“单张任意局部裁剪、髓心可远在图外、极细线标注不确定性、有限/无穷远混合后验、校准置信区和弧段贡献”。这是有限检索结论，不是系统综述或“全球首次”声明。

下表是基线/依据候选审计，不表示当前仓库已完成这些项目的 adapter、数据下载或复现结果。拟复现时必须锁定论文版本、仓库 commit、访问日期、环境与权重。

## 木材领域

| 项目 | 许可/框架 | 可复用价值 | 不可直接承担的部分 |
|---|---|---|---|
| [APD / APD-PCL](https://github.com/hmarichal93/apd)、[论文](https://arxiv.org/abs/2404.01952) | MIT；Python | 结构张量方向、法线累积、PCLines、冷启动与主要髓心基线 | 面向完整截面；PCL 参数范围回避近无穷远；无概率后验和弧贡献 |
| [DeepCS-TRD](https://github.com/hmarichal93/deepcstrd)、[论文](https://arxiv.org/abs/2504.16242) | MIT；PyTorch | U-Net 年轮前端、切块推理、跨树种基线 | 连接流程依赖已有髓心，不能解决远图外逆问题 |
| [INBD](https://github.com/alexander-g/INBD)、[论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Gillert_Iterative_Next_Boundary_Detection_for_Instance_Segmentation_of_Tree_Rings_CVPR_2023_paper.pdf) | 代码 MPL-2.0；所配数据/标注 CC BY-NC-SA 4.0；PyTorch | 年轮实例顺序和逐轮边界基线 | 从髓心开始，局部图外条件不成立；文件级 copyleft 要隔离；NC/SA 数据不得默认用于商业训练 |
| [CS-TRD/IPOL](https://github.com/hmarichal93/cstrd_ipol)、[论文](https://www.ipol.im/pub/art/2025/485/) | MIT；Python/C | 边缘连接、拓扑整理和完整年轮评估 | 已知髓心、完整截面、不可微 |
| [TRAS](https://github.com/hmarichal93/tras)、[论文](https://arxiv.org/abs/2605.08025) | 仓库 GPL-3.0-only | 交互标注、人工修订和统一基线外壳 | 强 copyleft；核心检测仍继承完整圆盘/已有髓心假设 |
| [UruDendro](https://github.com/hmarichal93/uruDendro)、[论文](https://link.springer.com/article/10.1186/s13595-025-01296-5) | 代码、数据和标注需分开核查 | 多专家完整轮线、髓心与标注离散带；可从完整截面自行生成髓心图外的配对裁剪 | 数据本身不是现成的“图外配对裁剪集”；必须按树级先划分再裁剪 |
| [UruDendro4](https://arxiv.org/abs/2511.20935) | 数据 CC BY 4.0 | 102 个多高度截面，外部泛化 | 仍是完整截面年轮检测数据 |
| [Automatic Pith Detection with Deep Learning](https://arxiv.org/abs/2512.00625) | arXiv 预印本；未见对应完整官方工程 | 比较 YOLOv9、U-Net、Swin、DeepLabV3、Mask R-CNN 的全截面髓心检测 | 目标是图内可见髓心，不支持任意图外坐标/方向—逆距离后验 |
| [ACO/IPOL](https://www.ipol.im/pub/art/2022/338/) | IPOL 文章与源码包；集成前核查实际源包许可 | 异常条件下的传统候选基线 | 不是概率远场模型 |
| [合成原木端面](https://github.com/dagbjornberg/Image-Generation-Log_Ends) | 未发现明确仓库 LICENSE；不复制/集成 | 作为裂纹、节和外观异常的方法参考 | 不能代替真实远场与置信校准；论文许可不自动授权仓库代码/资产 |

表中仅链接 arXiv 的版本按预印本处理；投稿或冻结基线时应重新核对是否已有同行评审版、更新代码或变更许可。

## 细线、等变和不确定性

| 项目 | 许可 | 用途 |
|---|---|---|
| [clDice](https://github.com/jocpae/clDice) | MIT | 可微骨架与连通损失；不能单独约束位置 |
| [Boundary Loss](https://github.com/LIVIAETS/boundary-loss) | MIT | 距离边界损失；距离图必须从不确定共识线产生 |
| [UAED](https://github.com/ZhouCX117/UAED_MuGE) | 未发现明确仓库 LICENSE；不复制/集成 | 只作多标注边缘均值/方差建模的论文依据 |
| [Faithful Heteroscedastic Regression](https://proceedings.mlr.press/v206/stirn23a.html) | 论文/作者实现需分别核查 | 方差目标与均值/共享表征梯度解耦的训练依据 |
| [Probabilistic U-Net](https://github.com/SimonKohl/probabilistic_unet) | Apache-2.0 | 拓扑存在多种合理解释时的随机分割消融 |
| [Stochastic Segmentation Networks](https://github.com/biomedia-mira/stochastic_segmentation_networks) | MIT | 弧内相关的掩膜不确定性 |
| [escnn](https://github.com/QUVA-Lab/escnn) | BSD-3-Clause-Clear | C4/C8/SO(2) 等变编码器；表示类型和编译依赖需单独工程化 |
| [Kornia](https://github.com/kornia/kornia) | Apache-2.0 | 仿射、图像微分与有效区域；任意角插值不等于严格等变 |
| [TorchUncertainty](https://github.com/torch-uncertainty/torch-uncertainty) | Apache-2.0 | 集成、MC Dropout、校准/选择性风险指标 |
| [PyPose](https://github.com/pypose/pypose) | Apache-2.0 | 可选 GN/LM 后端；EM、Schur 和混合分支仍需自研 |
| [Theseus](https://github.com/facebookresearch/theseus) | MIT | 可微优化参考；版本/CUDA 约束使其不宜成为唯一后端 |
| [Ellipsoidal Conformal MTR](https://github.com/M-Soundouss/EllipsoidalConformalMTR) | 未发现明确仓库 LICENSE；不复制/集成 | 只作二维椭圆 conformal 方法参考 |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | BSD-3-Clause | 分位数和覆盖率逻辑参考；多峰/无界集合需自定义 |

## 集成和许可证原则

- APD、DeepCS-TRD、CS-TRD、INBD、TRAS 只能在各自许可允许的独立环境中作为冻结基线；当前仓库尚未提供这些 adapter；
- 核心包仅采用自身实现及 MIT/BSD/Apache 兼容依赖；
- 不把 GPL-3.0-only TRAS 代码复制到 MIT 核心；
- MPL-2.0 INBD 保持独立进程/环境或保留原文件许可；
- MiSCS 等带 `NC/SA` 的数据不能默认用于商业训练；
- `segmentation_models.pytorch` 的具体编码器和预训练权重分别审计；
- 未声明 LICENSE 的公开仓库默认**没有复制、修改或再分发授权**，只能阅读论文/界面作为方法参考；
- 代码、数据、标注和预训练权重分别核查；代码许可不自动覆盖数据或权重；
- 论文文本与仓库许可证不一致时，以所用 commit 的实际文件为准，并在 `THIRD_PARTY_NOTICES` 或等价记录中保存 SPDX ID、commit、访问日期和必要声明。
