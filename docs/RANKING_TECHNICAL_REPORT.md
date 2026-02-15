# 精排阶段技术白皮书 (Ranking Layer Technical Report)

## 1. 精排阶段目标
作为推荐系统的最后一环，精排（Ranking）负责对召回出的几百个候选电影进行极其精准的打分。其核心挑战在于：
- **个性化深度**：超越热门和类别，挖掘用户对导演、演员及剧情细节的微观偏好。
- **动态感知**：理解用户最近 5 次观影行为对当前决策的影响。
- **多目标权衡**：同时优化“是否观看（点击）”与“看完后的评分”。

## 2. 核心架构：UnifiedRanker Pro
本项目实现了一个工业级的深度学习模型，融合了三大核心技术：

| 模块 | 技术方案 | 解决的问题 |
| :--- | :--- | :--- |
| **序列建模 (DIN)** | Deep Interest Network (Attention) | 捕捉用户动态兴趣，将聚光灯打在与目标电影最相关的历史行为上。 |
| **特征交叉 (DCN-V2)** | Deep & Cross Network V2 | 显式建模高阶特征交叉（如：用户年龄 x 导演 x 题材），挖掘隐藏关联。 |
| **多专家系统 (MMoE)** | Multi-gate Mixture-of-Experts | 共享底层特征，通过不同专家门控同时优化 Click (Listwise) 和 Rating (MSE) 任务。 |

## 3. 数据工程逻辑

### 3.1 三段式时间隔离 (Three-Stage Isolation)
为了彻底解决推荐系统中常见的“数据泄露”导致的 AUC 虚高问题，我们将数据在时间轴上切分为：
1. **History (0% - 60%)**: 仅用于计算用户画像、偏好导演/演员索引。
2. **Train (60% - 80%)**: 模型训练区。其特征提取仅允许参考 History 区的信息。
3. **Val (80% - 100%)**: 评估验证区。确保模型在“完全陌生的未来”进行预测。

### 3.2 特征配方 (Feature Mix)
- **Dense**: 1024 维 BGE-M3 语义向量、TMDb 业务特征（预算、票房、时长）、用户统计特征。
- **Sparse**: 电影 ID、导演 ID、演员 ID、题材索引。
- **Match**: 用户-导演命中、用户-演员匹配数、题材 Jaccard 相似度、剧情余弦距离。

## 4. 训练策略：Listwise Preference Learning
本项目弃用了传统的 Pointwise (BCE Loss)，转而采用 **Listwise** 建模：
- **逻辑**: 将 1 个正样本和 4 个负样本视为一个小组。
- **优化器**: Softmax Cross-Entropy。
- **目标**: 最小化正确电影在组内的排序误差，直接优化 NDCG/Top-1 Acc。

## 5. 效果分析
- **Baseline (XGBoost)**: 依靠强力特征工程达到 0.67 AUC。
- **UnifiedRanker Pro**: 引入序列特征与 MMoE，具备更高的泛化能力与多目标预估精度。

## 6. 后续演进
- **重排阶段 (Reranking)**：引入多样性（Diversity）约束与打散算法（MMR）。
- **实时 Serving**：将特征引擎与模型导出为 TorchScript 进行高性能推理。
