# TMDb 外部元数据建模与表结构说明

本篇文档详细说明了经过整合建模后的 TMDb 外部元数据。所有数据均已对齐 MovieLens `movieId`，采用 **星型模型 (Star Schema)** 存储，位于 `data/processed/tmdb/` 目录下。

## 1. 核心表结构

### 1.1 电影详情主表 (`tmdb_movies.parquet`)
- **movieId** (int): 关联 MovieLens 的主键。
- **tmdbId** (int): TMDb 原始 ID。
- **overview** (text): 剧情简介，用于后续文本 Embedding。
- **runtime** (int): 电影时长（分钟）。
- **budget / revenue** (long): 制作预算与全球票房。
- **vote_average / vote_count**: TMDb 社区的评分与评价人数。
- **original_language**: 电影原始语种（如 en, fr, zh）。

### 1.2 剧情简介语义向量表 (`tmdb_embeddings_full.parquet`)
- **movieId**: 主键。
- **embedding**: 全量高维向量 (如 BGE-M3 1024 维)。

### 1.3 人员维度表 (`tmdb_persons.parquet`)
- **personId** (int): TMDb 唯一人员 ID。
- **name** (text): 人员姓名。
- **gender**: 性别 (0:未知, 1:女性, 2:男性)。

### 1.4 关系表 (`tmdb_movie_cast` / `tmdb_movie_crew`)
- 分别描述演员（主演）和幕后团队（导演/编剧）的关联。

---

## 2. 采集与处理效率优化

为了在 M4 芯片环境下全速收割海量元数据，本项目设计了**分层处理策略**：

### 2.1 采集机制
- **并发控制**: 采用 `ThreadPoolExecutor`。
    - **元数据抓取**: 50 线程（网络 I/O 密集型）。
    - **Embedding 获取**: 3 线程（API 频率/厂商风控限制）。
- **断点续传**: 以本地 JSON 缓存和 Parquet 索引为基准，支持秒级任务恢复。

### 2.2 数据分层逻辑 (Tiered Processing)
针对 Embedding 采集，我们区分了通信层和持久化层：
- **API Batch (64)**: 受限于厂商 Payload 限制，单次 HTTP 请求聚合 64 条文本，规避 413 错误。
- **Checkpoint Chunk (500)**: 为了平衡磁盘 I/O 压力与数据安全，每累积 500 条向量执行一次 Parquet/CSV 覆盖写盘。

---

## 3. 对排序阶段 (Ranking) 的价值
- **特征交叉**: 计算用户偏好的导演/演员与当前候选电影的重合度。
- **语义匹配**: 计算用户历史剧情简介 Embedding 与候选电影剧情 Embedding 的余弦相似度。
- **业务画像**: 识别用户是喜欢“高预算大片”还是“高分独立电影”。

## 4. 文件格式说明
- **Parquet**: 生产环境使用，支持高效列式读取。
- **CSV**: 办公/分享环境使用，可直接通过 Excel 打开查看。
