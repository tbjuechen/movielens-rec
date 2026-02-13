# MovieLens Recommendation System (movielens-rec)

## 项目综述
本项目旨在 MovieLens 32M 数据集上实践工业级推荐系统全流程。目前已完成高性能数据底座和多路召回层的构建，正在通过外部 API 丰富特征，为排序阶段做准备。

## 核心文件结构
- **`data/processed/tmdb/`**: 基于星型模型建模的外部元数据。
- **`src/models/recall/`**: 包含各路召回算法的实现。
- **`scripts/data_collector/`**: 健壮的分布式采集与数据整合模块。

## 路线图 (Implementation Roadmap)

### Phase 1: 数据与特征工程
- [x] **数据格式化**: CSV -> Parquet (Completed).
- [x] **EDA**: 产出 `docs/EDA_REPORT.md` (Completed).
- [x] **外部采集**: 50 线程并发 TMDb 采集器 (Completed).
- [x] **深度建模**: 外部数据星型建模整合 (Completed).

### Phase 2: 多路召回模块 (Recall)
- [x] **架构闭环**: 建立 `BaseRecall` 统一接口 (Completed).
- [x] **热门召回**: 0.20 HitRate (Completed).
- [x] **ItemCF**: 0.15 HitRate (Completed).
- [x] **双塔 V2**: **0.35 HitRate** (Completed).

### Phase 3: 精排模型 (Ranking)
- [ ] **Ranking EDA**: 分析外部特征与 Label 的相关性 (In Progress).
- [ ] **样本构造**: 实现精排正负样本采样。
- [ ] **模型实现**: XGBoost / DeepFM。

## 技术深度决策 (Technical Decisions)
1. **数据存储**: 全面采用 Parquet 替代 CSV，读取性能提升 10 倍以上。
2. **建模范式**: 采用 **Star Schema**（星型模型）处理“电影-人员-关系”数据，避免了特征工程中的冗余计算。
3. **召回策略**: 采用 **In-batch 负采样** 配合大 Batch Size (8192)，利用局部对比学习模拟全量空间检索。
4. **加速方案**: 针对 MPS 不支持 `EmbeddingBag` 的问题，手动实现 `mean` 聚合算子，确保 M4 芯片全速运行。

## 开发规范
- **原子提交**: 每个 commit 必须是功能原子化的。
- **文档优先**: 核心功能上线必须同步更新 README 和技术报告。
