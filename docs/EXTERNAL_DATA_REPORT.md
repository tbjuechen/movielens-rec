# TMDb 外部元数据建模与表结构说明

本篇文档详细说明了经过整合建模后的 TMDb 外部元数据。所有数据均已对齐 MovieLens `movieId`，采用 **星型模型 (Star Schema)** 存储，位于 `data/processed/tmdb/` 目录下。

## 1. 核心表结构

### 1.1 电影详情主表 (`tmdb_movies.parquet`)
这是描述电影内容和业务属性的核心表。
- **movieId** (int): 关联 MovieLens 的主键。
- **tmdbId** (int): TMDb 原始 ID。
- **overview** (text): 剧情简介，用于后续文本 Embedding。
- **runtime** (int): 电影时长（分钟）。
- **budget / revenue** (long): 制作预算与全球票房。
- **vote_average / vote_count**: TMDb 社区的评分与评价人数。
- **original_language**: 电影原始语种（如 en, fr, zh）。

### 1.2 人员维度表 (`tmdb_persons.parquet`)
存储所有出现在电影中的唯一演职员信息。
- **personId** (int): TMDb 唯一人员 ID。
- **name** (text): 人员姓名。
- **gender** (int): 性别 (0:未知, 1:女性, 2:男性)。
- **profile_path**: 人员头像图片的相对路径。

### 1.3 演员关系表 (`tmdb_movie_cast.parquet`)
描述“谁演了哪部戏”。
- **movieId** (int): 关联电影。
- **personId** (int): 关联人员。
- **character** (text): 角色名。
- **order** (int): 演员排序。**0 代表第一主演**，通常取 `order < 5` 作为核心特征。

### 1.4 团队关系表 (`tmdb_movie_crew.parquet`)
描述“谁执导/编写了哪部戏”。
- **movieId** (int): 关联电影。
- **personId** (int): 关联人员。
- **job** (text): 职位（如 Director, Screenplay, Producer）。
- **department**: 所属部门。

---

## 2. 表关联示例 (SQL/Pandas 逻辑)

若要找出“某个用户看过的所有电影的导演”，操作逻辑如下：
1. 从 `ratings` 表根据 `userId` 找到其看过的 `movieId`。
2. 将 `movieId` 与 `tmdb_movie_crew` 进行 **Inner Join**。
3. 过滤 `job == 'Director'`。
4. 将结果与 `tmdb_persons` 关联获取导演姓名。

---

## 3. 对排序阶段 (Ranking) 的价值
- **特征交叉**：计算用户偏好的导演/演员与当前候选电影的重合度。
- **语义匹配**：计算用户历史剧情简介 Embedding 与候选电影剧情 Embedding 的余弦相似度。
- **业务画像**：识别用户是喜欢“高预算大片”还是“高分独立电影”。

## 4. 文件格式说明
- **Parquet**: 生产环境使用，支持高效读取。
- **CSV**: 办公/分享环境使用，可直接通过 Excel 打开查看。
