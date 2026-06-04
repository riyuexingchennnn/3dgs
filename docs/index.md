# 三维高斯溅射（3DGS）完全教程

> 面向零基础程序员的系统性学习路径：从直觉理解到工程实践

---

## 前言

本教程面向具备编程基础的本科毕业生，不要求任何三维视觉或图形学背景。
你将从"新场景重建是什么"出发，逐步理解 3DGS 的数学原理、训练流程与工程实践，
读完后能够独立训练模型、定制优化、参与前沿改进。

**阅读建议**：顺序阅读，每章末有小结与思考题；公式可跳过初读，二刷时重点推敲。

---

## 目录

### [第一章：三维重建是什么，为什么重要](chapter1.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 1.1 | 从一张照片说起 | 人眼如何感知三维；照片丢失了什么信息 |
| 1.2 | 三维重建的定义与目标 | 输入/输出是什么；评价指标 PSNR / SSIM / LPIPS |
| 1.3 | 主流技术路线全景 | MVS、NeRF、3DGS 的横向对比；各自适用场景 |
| 1.4 | 3DGS 能做什么，不能做什么 | 典型应用案例；已知局限性 |
| 1.5 | 学习路线与本教程结构 | 如何使用本教程；推荐配套读物 |

---

### [第二章：视觉基础——相机、投影与点云](chapter2.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 2.1 | 针孔相机模型 | 内参矩阵 $K$；焦距、主点、像素坐标 |
| 2.2 | 相机外参与世界坐标系 | 旋转矩阵 $R$、平移向量 $t$；坐标变换链 |
| 2.3 | 齐次坐标与投影矩阵 | 为什么用齐次坐标；完整投影公式 |
| 2.4 | 畸变模型 | 径向畸变、切向畸变；实际标定流程 |
| 2.5 | 点云是什么 | 从立体视觉到稀疏点云；SfM 的直觉解释 |
| 2.6 | COLMAP 实战入门 | 安装、运行、读取结果；为 3DGS 准备输入 |

---

### [第三章：高斯函数与概率基础](chapter3.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 3.1 | 一维高斯函数的几何直觉 | 均值、方差、面积归一化 |
| 3.2 | 多维高斯分布 | 协方差矩阵的几何意义；椭球体可视化 |
| 3.3 | 协方差矩阵的分解 | $\Sigma = RSS^T R^T$；旋转与缩放的解耦 |
| 3.4 | 为什么选高斯作为场景基元 | 可微性；闭合解析投影；连续可导的优化目标 |
| 3.5 | 高斯的加法与混合 | 多个高斯叠加；Gaussian Mixture Model 直觉 |

---

### [第四章：球谐函数与外观建模](chapter4.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 4.1 | 颜色与方向的关系 | 镜面反射、漫反射；非朗伯体表面 |
| 4.2 | 球谐函数是什么 | 在球面上的"傅里叶基"；阶数与频率的对应 |
| 4.3 | 球谐函数的数学定义 | 实数形式 $Y_l^m$；前四阶系数数量 |
| 4.4 | 用球谐函数表示颜色 | RGB 各通道独立展开；视角相关颜色的计算 |
| 4.5 | 阶数对质量与开销的影响 | 0阶=常色；3阶的工程折衷 |

---

### [第五章：体渲染与 Alpha 合成](chapter5.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 5.1 | 体渲染方程的直觉推导 | 光线穿过介质；透射率 $T$；吸收与散射 |
| 5.2 | 离散近似：Alpha 合成 | 前向 $\alpha$-blending 公式；顺序敏感性 |
| 5.3 | NeRF 的体渲染实现 | 采样点、权重计算；体渲染瓶颈 |
| 5.4 | 3DGS 的渲染公式推导 | 每个高斯的贡献；深度排序 tile-based rasterizer |
| 5.5 | 可微渲染的意义 | 反向传播到场景参数；梯度信号的流向 |

---

### [第六章：3DGS 核心算法详解](chapter6.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 6.1 | 整体架构一览 | 输入→SfM→初始化→训练→渲染；数据流图 |
| 6.2 | 高斯基元的完整参数定义 | 位置 $\mu$、协方差 $\Sigma$、不透明度 $\alpha$、球谐系数 |
| 6.3 | 三维高斯投影到二维 | 雅可比近似推导；二维协方差计算 |
| 6.4 | Tile-based 光栅化 | 屏幕分块；排序；并行渲染原理 |
| 6.5 | 自适应密度控制（ADC） | 克隆、分裂、剪枝的触发条件与效果 |
| 6.6 | 损失函数设计 | $\mathcal{L}_1$ + D-SSIM；各项权重的意义 |
| 6.7 | 训练超参数解读 | 学习率调度；迭代步数；指数衰减策略 |

---

### [第七章：从原始论文到开源代码](chapter7.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 7.1 | 论文精读：原版 3DGS（2023）| Kerbl et al. 核心贡献；与 NeRF 的关键差异 |
| 7.2 | 代码仓库结构导览 | 文件树；各模块职责；CUDA 扩展的边界 |
| 7.3 | 训练脚本逐行解读 | `train.py` 主循环；关键 API 调用 |
| 7.4 | 渲染脚本与评估 | `render.py`；`metrics.py`；如何解读指标 |
| 7.5 | 常见报错与排查手册 | CUDA OOM；COLMAP 失败；点云为空等 |

---

### [第八章：数据准备与训练实战](chapter8.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 8.1 | 硬件与环境配置 | GPU 显存需求；CUDA 版本；Docker 镜像 |
| 8.2 | 公开数据集下载与使用 | Tanks and Temples；Mip-NeRF 360；BlendedMVS |
| 8.3 | 用自己的视频训练 | 视频抽帧；COLMAP 流程；质量检查 |
| 8.4 | 训练参数精调指南 | 不同场景的推荐配置；快速实验策略 |
| 8.5 | 可视化与结果分析 | SIBR Viewer 使用；用 Python 渲染轨迹 |
| 8.6 | 导出与部署 | PLY 格式；WebGL 实时渲染；移动端适配 |

---

### [第九章：主流改进方向与前沿论文](chapter9.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 9.1 | 压缩与轻量化 | Compact 3DGS；LightGaussian；Mini-Splatting（2024，极限压缩至原始点数的 0.2%）；存储瓶颈分析 |
| 9.2 | 动态场景与视频 | 4D Gaussians；Deformable 3DGS；SplatFields（2024，神经变形场与 3DGS 融合）；时序建模范式对比 |
| 9.3 | 大场景与城市级重建 | VastGaussian；CityGaussian；GaussianWorld（2025，开放世界可扩展场景表示）；分块与流式训练策略 |
| 9.4 | 文本/图像驱动生成 | GaussianDreamer；DreamGaussian；结合扩散模型的零样本重建 |
| 9.5 | 物理模拟与编辑 | PhysGaussian；可交互场景；刚体软体模拟；GaussianOccupancy（2024，占用场与高斯融合用于物理感知） |
| 9.6 | 结构化与稀疏表示 | Scaffold-GS（2024，锚点驱动的神经高斯，显著降低冗余基元数量）；层次化场景表示 |
| 9.7 | 平面基元与几何改进 | 2DGS（2024，用二维圆盘替代三维椭球，改善表面几何精度）；几何感知密度控制 |
| 9.8 | 语义理解与分割 | Gaussian Grouping（2024，无需额外标注的实例级语义分割）；Feature 3DGS；开放词汇场景查询 |
| 9.9 | 光照分解与重光照 | Relightable 3DGS（2024，将高斯属性分解为漫反射、镜面与可见度）；GaussianShader；材质估计 |
| 9.10 | 人体与头像建模 | GaussianAvatars（2024，基于 FLAME 驱动的高精度人脸头像）；3DGS 在人体动画中的应用；驱动与表情迁移 |
| 9.11 | SLAM 与实时定位建图 | 3DGS-SLAM（2024，将高斯场景表示嵌入实时 SLAM 系统）；SplaTAM；在线增量建图；闭环检测 |
| 9.12 | 自动驾驶仿真 | HUGSIM（2025，统一闭环自动驾驶仿真框架，覆盖多传感器）；EmerNeRF；UniSim；传感器级真实性评测 |
| 9.13 | 阅读前沿论文的方法论 | arXiv 检索技巧；如何快速判断论文价值；跟踪 2025-2026 新进展的工作流 |

---

### [第十章：成为贡献者——如何迭代改进系统](chapter10.md)

| 节 | 标题 | 要点 |
|----|------|------|
| 10.1 | 建立自己的实验基线 | 可复现实验的工程规范；日志与版本管理 |
| 10.2 | 如何定位性能瓶颈 | Profiling CUDA 核；内存带宽分析 |
| 10.3 | 修改高斯基元的正确姿势 | 添加新属性；修改梯度流；CUDA 扩展入门 |
| 10.4 | 使用 gsplat 库加速开发 | gsplat（Nerfstudio 团队维护的高性能 3DGS CUDA 后端）；API 对比原版实现；自定义算子插入点 |
| 10.5 | nerfstudio 集成与生态 | nerfstudio 中的 Splatfacto 实现；插件式训练框架；与 gsplat 的协同使用；快速原型验证流程 |
| 10.6 | Gaussian Opacity Fields 解析 | GOF（2024，用不透明度场替代显式 alpha，改善无纹理区域与边界重建）；体积表示与高斯基元的互补性 |
| 10.7 | 3DGUT：非结构化高斯溅射 | 3DGUT（2025，支持非结构化相机模型与鱼眼镜头的通用高斯训练框架）；工业级数据集适配 |
| 10.8 | 消融实验设计 | 控制变量；指标选取；统计显著性 |
| 10.9 | 投稿与开源的建议 | 论文写作要点；代码整洁度；社区协作 |
| 10.10 | 推荐学习资源汇总 | 论文、课程、博客、社区的精选清单（含 2025 年新增资源） |

---

## 附录

| 附录 | 内容 |
|------|------|
| A | 线性代数速查：矩阵乘法、特征值分解、SVD |
| B | 概率论速查：期望、协方差、多元高斯 |
| C | 自动微分与反向传播原理 |
| D | CUDA 编程入门：线程层级、共享内存 |
| E | 术语中英对照表 |

---

## 推荐配套读物

### 基础教材
- **《计算机视觉：算法与应用》**（Szeliski，第2版）— 相机模型、立体视觉权威参考
- **《深度学习》**（Goodfellow et al.）— 自动微分、优化器原理

### 必读论文
| 论文 | 贡献 | 链接 |
|------|------|------|
| Kerbl et al., 2023 | 原版 3DGS | [arXiv:2308.04079](https://arxiv.org/abs/2308.04079) |
| Mildenhall et al., 2020 | NeRF | [arXiv:2003.08934](https://arxiv.org/abs/2003.08934) |
| Schönberger & Frahm, 2016 | COLMAP SfM | [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf) |
| Zwicker et al., 2001 | EWA Splatting | [SIGGRAPH 2001](https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TOG02.pdf) |
| Lu et al., 2024 | Scaffold-GS：锚点驱动结构化高斯 | [arXiv:2312.00109](https://arxiv.org/abs/2312.00109) |
| Huang et al., 2024 | 2DGS：平面高斯改善表面重建 | [arXiv:2403.17888](https://arxiv.org/abs/2403.17888) |
| Liu et al., 2024 | Gaussian Grouping：语义实例分割 | [arXiv:2312.00732](https://arxiv.org/abs/2312.00732) |
| Liang et al., 2024 | Relightable 3DGS：光照分解与重光照 | [arXiv:2311.09897](https://arxiv.org/abs/2311.09897) |
| Qian et al., 2024 | GaussianAvatars：FLAME 驱动人脸头像 | [arXiv:2312.02069](https://arxiv.org/abs/2312.02069) |
| Yugay et al., 2024 | 3DGS-SLAM：高斯场景的实时 SLAM | [arXiv:2312.02126](https://arxiv.org/abs/2312.02126) |
| Yu et al., 2024 | Gaussian Opacity Fields：体积不透明度场 | [arXiv:2404.10772](https://arxiv.org/abs/2404.10772) |
| Chen et al., 2025 | HUGSIM：统一自动驾驶闭环仿真 | [arXiv:2501.12320](https://arxiv.org/abs/2501.12320) |
| Niedermayr et al., 2024 | Mini-Splatting：极致高斯压缩 | [arXiv:2403.14166](https://arxiv.org/abs/2403.14166) |
| Waczynska et al., 2024 | SplatFields：神经场驱动动态高斯 | [arXiv:2409.11211](https://arxiv.org/abs/2409.11211) |
| Fischer et al., 2025 | 3DGUT：非结构化相机模型高斯训练 | [arXiv:2312.02121](https://arxiv.org/abs/2312.02121) |

### 在线课程
- **Stanford CS231A**：计算机视觉基础（相机与三维视觉）
- **CMU 16-385**：计算机视觉（含立体和重建）
- **MIT 6.837**：计算机图形学（体渲染部分）

### 开源框架与工具

- [3DGS 官方仓库](https://github.com/graphdeco-inria/gaussian-splatting) — 原版代码、issue、社区讨论
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Nerfstudio 团队维护的高性能 3DGS CUDA 后端，支持自定义扩展
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) — 模块化 NeRF/3DGS 训练框架，内置 Splatfacto
- [Awesome-3D-Gaussian-Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) — 持续更新的论文与资源列表
- Luma AI / Polycam 博客 — 工业级落地经验与工程技巧

---

本教程持续更新，当前版本覆盖 3DGS 截至 2026 年中的主流进展。
