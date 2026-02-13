# MovieLens Recommendation System (movielens-rec)

## 项目综述
本项目旨在 MovieLens 32M 数据集上实践工业级推荐系统全流程。目前已完成高性能数据底座和多路召回层的构建，正在通过外部 API 丰富特征，为排序阶段做准备。

## 核心文件结构
- **`data/processed/tmdb/`**: 基于星型模型建模的外部元数据。
- **`src/models/recall/`**: 包含各路召回算法的实现。
- **`src/features/`**: 核心特征工程模块（含语义嵌入与序列处理）。
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
- [/] **Ranking EDA**: 分析外部特征与 Label 的相关性 (Completed).
- [ ] **语义增强**: 调用 API 获取剧情简介的语义向量 (Planned).
- [ ] **特征工程**: 构建用户画像、物品画像及历史行为序列 (Planned).
- [ ] **样本构造**: 实现精排正负样本采样。
- [ ] **模型实现**: XGBoost / DeepFM。

## 排序阶段待办清单 (Ranking Phase To-Do List)

### 3.1 语义向量化 (Phase 3.1)
- [ ] **实现 `api_embedder.py`**: 支持批量获取 Embedding，内置 **Matryoshka 降维** (1536 -> 256)。
- [ ] **实现断点续传**: 自动保存已获取的向量，防止网络中断。

### 3.2 特征底座建设 (Phase 3.2)
- [ ] **构建 `User_Profile`**: 整合用户平均分、题材偏好、活跃度等。
- [ ] **构建 `Item_Profile`**: 整合电影题材、TMDb 时长/预算、导演/主演影响力、语义向量等。
- [ ] **行为序列处理**: 实现 Time-sorted 滑动窗口，提取用户**最近 N 次**观影历史。

### 3.3 特征交叉引擎 (Phase 3.3)
- [ ] **开发 `Ranking_Feature_Engine`**:
    - 计算“用户-导演”命中、题材 Jaccard 相似度。
    - 计算当前电影与用户历史序列的 Embedding 余弦距离。

### 3.4 模型训练与评估 (Phase 3.4)
- [ ] **负采样**: 1:10 曝光未点击采样 + 随机负采样。
- [ ] **模型开发**: 训练第一个 XGBoost 排序器作为强基准。

## 技术深度决策 (Technical Decisions)
1. **建模范式**: 采用 **Star Schema**（星型模型）处理“电影-人员-关系”数据。
2. **硬件加速**: 针对 MPS 手动实现 `mean` 聚合算子，支持 M4 芯片。
3. **语义特征存储**: 采用 **“全量存储、运行时切片”** 策略。使用 OpenAI `text-embedding-3` 获取并持久化全量 1536 维向量作为黄金数据源，精排引擎加载时自动切分至 256 维以优化计算性能。

## 开发规范
- **原子提交**: 每个 commit 必须是功能原子化的。
- **文档优先**: 核心功能上线必须同步更新 README 和技术报告。
