# Gemini Instructions for MovieLens Recommender

## 1. 基础规范 (Core Standards)
- **沟通语言**: 始终使用 **中文** 进行交流。
- **环境约束**: 必须使用 Conda 环境 `movielens-rec`。所有 shell 指令应以 `conda run -n movielens-rec` 启动。
- **Git 规范**: 采用 **原子化提交 (Atomic Commits)**。每一项新功能、优化或修复必须单独提交，并附带清晰的中文或英文注释。
- **文件格式**: 大规模中间数据必须使用 **Parquet** 格式，禁止使用未经压缩的 CSV 存储中间宽表。

## 2. 数据处理准则 (Data Engineering)
- **防数据穿越**: 所有的用户特征 (Profile) 必须基于训练集 (`train_data`) 计算。验证集和测试集仅作为 Target 使用，严禁泄露未来信息。
- **性能优先**: 针对千万级数据 (MovieLens-32M)，必须优先使用 **Pandas 向量化 (Vectorization)** 或 NumPy 数组操作，严禁在训练循环中使用 slow-apply 或逐行循环。
- **特征工程**: 
    - 针对长尾分布特征 (Revenue, Activity, Vote Count)，必须进行 **Log 变换**。
    - 针对 Release Year 等时间特征，必须进行 **分箱 (Binning)** 处理。

## 3. 召回模型架构准则 (Recall Architecture)
- **双塔模型 (Dual-Tower)**:
    - **权重共享**: User 塔的历史行为序列 Embedding 与 Item 塔的物品 ID Embedding 必须 **强共享**。
    - **混合损失**: 采用 **InfoNCE (简单负样本)** + **BPR (困难负样本)** 的混合 Loss。
    - **采样策略**: 必须同时包含 In-batch 负采样、全局均匀随机采样和热门物品困难负采样。
    - **Log-Q 纠偏**: 必须实现 Log-Q Correction 以消除热度偏置。纠偏公式顺序必须为 `(Score / Tau) - LogQ`。
    - **时序感知**: 用户历史行为聚合必须包含 **时间衰减 (Time-Decay)** 加权。

## 4. 评估准则 (Evaluation)
- **召回评估**: 核心指标为 **Recall@50** 和 **Recall@100**。
- **评估链路**: 必须包含全量向量索引构建 (FAISS) 环节，以模拟真实的线上检索性能。

---
*本文件定义的指令具有最高优先级，覆盖所有通用开发原则。*
