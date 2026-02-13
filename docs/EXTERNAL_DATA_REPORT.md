# 外部数据采集与建模报告 (External Data Report)

## 1. 采集背景
MovieLens 原始数据集虽然庞大，但特征深度有限（仅 ID、题材、年份）。为了在排序阶段实现更精准的推荐，我们引入了 TMDb (The Movie Database) 的全量元数据。

## 2. 数据建模：星型模型 (Star Schema)
为了避免海量演职员信息带来的数据冗余，我们将采集到的 JSON 平铺为以下四张结构化 Parquet 表：

| 表名 | 核心字段 | 业务用途 |
| :--- | :--- | :--- |
| **tmdb_movies** | `overview`, `budget`, `runtime`, `vote_average` | 提供文本语义特征、商业规模特征。 |
| **tmdb_persons** | `personId`, `name`, `gender`, `popularity` | 人员维度库，用于去重和演员人气分析。 |
| **tmdb_movie_cast** | `movieId`, `personId`, `order` | 电影-演员关系，`order` 可用于提取前 N 名主演。 |
| **tmdb_movie_crew** | `movieId`, `personId`, `job` | 电影-团队关系，用于提取导演、编剧等核心主创。 |

## 3. 核心特征工程方向 (Future Work)

### 3.1 文本语义理解
- **Overview Embedding**: 使用预训练模型将剧情简介转化为向量，计算用户历史偏好与候选集的语义相似度。

### 3.2 强力交叉特征
- **Director Match**: 用户打过高分的电影中，是否存在同一导演的作品。
- **Cast Power**: 统计用户看过的电影中主演的平均“咖位”（TMDb Popularity）。
- **Production Scale**: 区分用户偏好独立文艺片（低 Budget）还是商业大片（高 Budget）。

## 4. 采集效率优化
- **并发机制**: 采用 `ThreadPoolExecutor` 开启 **50 线程** 并发抓取。
- **断点续传**: 以 `tmdbId.json` 文件为原子缓存，自动跳过已成功抓取的项目。
- **容错处理**: 自动处理 HTTP 429 (限流) 重试逻辑。
