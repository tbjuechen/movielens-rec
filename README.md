# MovieLens-Rec-V2

基于 MovieLens 32M 和 TMDB 元数据的离线推荐系统项目，覆盖了完整的召回、候选池构造、排序训练与离线评估流程。当前仓库已经不是早期的“只做召回 demo”，而是一套可以顺着脚本跑通的数据与模型流水线。

## 项目目标

- 用 `MovieLens-32M` 构建一个接近工业推荐系统分层架构的离线实验环境。
- 在召回层同时支持向量召回、协同过滤和简单统计召回。
- 通过离线构造候选池，让排序模型尽量贴近真实线上分发场景。
- 提供可复现实验路径，方便继续调参、替换模型和补充特征。

## 当前实现概览

### 召回层

- `DualTower` 双塔召回
  - 用户塔和物品塔分别编码
  - 支持 in-batch negative、global negative、hard negative
  - 损失函数为 `InfoNCE + BPR`
  - 使用 `FAISS` 做向量检索评估
- `ItemCF`
- `UserCF`
- `PopularityRecall`
- `GenreRecall`
- `RecallMerger`
  - 按配置权重融合多路召回结果

### 排序层

- `RankingModel`
  - 结构为 `DCNv2 + MMoE`
  - 双目标学习
    - `pCTR`
    - `pRating`
- 最终分数由 `pCTR` 和 `pRating` 组合得到
- 排序训练样本不是随机负采样，而是基于召回候选池构造

### 数据与特征

- 时间顺序切分，避免泄漏
- 用户画像来自训练集
  - 平均评分
  - 活跃度
  - Top Genres
  - 历史序列
- 物品画像融合 MovieLens 和 TMDB
  - 年份
  - 类型
  - 票房
  - 预算
  - TMDB 评分与投票数
- 类别特征和连续特征都通过 `FeatureEncoder` 持久化到 `feature_store`

## 目录结构

```text
movielens-rec-v2/
├── config.sample.yaml
├── docs/
│   └── design_spec.md
├── scripts/
│   ├── 00_prepare_tmdb.py
│   ├── 01_process_data.py
│   ├── 02_build_features.py
│   ├── 03_train_models.py
│   ├── 04_evaluate_e2e.py
│   ├── 05_build_ranking_data.py
│   ├── 06_train_ranking.py
│   └── 07_evaluate_ranking.py
├── src/
│   ├── config/
│   ├── data_pipeline/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   │   ├── ranking/
│   │   └── recall/
│   └── pipeline/
└── requirements.txt
```

## 环境准备

建议使用 Python 3.10。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你使用 Conda，也可以：

```bash
conda create -n movielens-rec python=3.10 -y
conda activate movielens-rec
pip install -r requirements.txt
```

## 数据准备

### 1. 放置 MovieLens 原始数据

默认配置下，原始 MovieLens 数据放在：

```text
data/raw/ml-32m/
```

至少需要这些文件：

- `movies.csv`
- `ratings.csv`
- `links.csv`

如果你后续还会扩展标签或分析任务，也可以一起放：

- `tags.csv`

### 2. 放置 TMDB JSON 缓存

`00_prepare_tmdb.py` 会读取：

```text
data/raw/tmdb_cache/*.json
```

脚本会把这些 JSON 合并成：

```text
data/processed/tmdb_features.parquet
```

仓库当前不包含抓取 TMDB 的脚本，所以这里默认你已经提前准备好了缓存文件。

## 配置说明

项目启动时会优先读取根目录下的 `config.yaml`。如果不存在，则自动回退到 `config.sample.yaml`。

也就是说：

- 想直接试跑：保留 `config.sample.yaml` 即可
- 想自定义路径、batch size、召回 K、排序参数：新建 `config.yaml`

主要配置项如下：

- `paths`
  - 原始数据目录
  - 处理后数据目录
  - 特征库目录
  - 模型权重目录
- `recall`
  - 双塔 embedding 维度、温度参数、召回 top_k
- `training`
  - 召回训练 batch size、epoch、负采样规模
- `merger_weights`
  - 多路召回融合权重
- `ranking`
  - 排序模型结构、训练参数、候选池大小、评估 topK
- `features`
  - 用户历史长度、genre 截断长度、时间衰减参数

## 推荐执行顺序

### 阶段 1：准备 TMDB 特征

```bash
python scripts/00_prepare_tmdb.py
```

产物：

- `data/processed/tmdb_features.parquet`

### 阶段 2：清洗数据并构建用户/物品画像

```bash
python scripts/01_process_data.py
```

这一步会完成：

- 按用户时间序列切分 `train/val/test`
- 基于训练集构建 `user_profile`
- 基于训练集构建 `item_profile`
- 生成简单召回所需倒排和热度文件

主要产物：

- `data/processed/train_data.parquet`
- `data/processed/val_data.parquet`
- `data/processed/test_data.parquet`
- `data/processed/user_profile.parquet`
- `data/processed/item_profile.parquet`
- `data/feature_store/genre_to_items.json`
- `data/feature_store/popularity_list.json`

### 阶段 3：构建编码器与特征字典

```bash
python scripts/02_build_features.py
```

这一步会把：

- `userId`
- `movieId`
- `genres`
- 连续特征归一化器

保存到 `feature_store`，供召回和排序共同使用。

### 阶段 4：训练召回模型

训练全部召回模型：

```bash
python scripts/03_train_models.py --model all
```

也可以分开训练：

```bash
python scripts/03_train_models.py --model dual_tower
python scripts/03_train_models.py --model item_cf
python scripts/03_train_models.py --model user_cf
```

主要产物：

- `data/model_weights/dual_tower.pth`
- `data/model_weights/item_sim_matrix.pkl`
- `data/model_weights/user_sim_matrix.pkl`

### 阶段 5：评估召回链路

```bash
python scripts/04_evaluate_e2e.py
```

这一步会做多路召回、融合和离线指标统计，重点看：

- `Recall@K`
- `NDCG@K`

### 阶段 6：构建排序训练候选池

```bash
python scripts/05_build_ranking_data.py
```

这一步很关键。它不是简单地给排序模型喂正负样本，而是：

- 先用 `ItemCF + Popularity + GenreRecall` 做 CPU 候选召回
- 再用 `DualTower + FAISS` 补充向量召回结果
- 最后把多路结果融合成排序训练/评估候选池

主要产物：

- `data/processed/ranking_candidate_pool.parquet`
- `data/processed/ranking_val_candidates.parquet`
- `data/processed/ranking_test_candidates.parquet`

### 阶段 7：训练排序模型

```bash
python scripts/06_train_ranking.py
```

主要产物：

- `data/model_weights/ranking_model.pth`

### 阶段 8：评估排序效果

验证集：

```bash
python scripts/07_evaluate_ranking.py --set val
```

测试集：

```bash
python scripts/07_evaluate_ranking.py --set test
```

重点指标包括：

- `HR@K`
- `NDCG@K`
- `MRR`

## 一条完整跑通命令

如果数据都准备好了，最常见的本地流程就是：

```bash
python scripts/00_prepare_tmdb.py
python scripts/01_process_data.py
python scripts/02_build_features.py
python scripts/03_train_models.py --model all
python scripts/04_evaluate_e2e.py
python scripts/05_build_ranking_data.py
python scripts/06_train_ranking.py
python scripts/07_evaluate_ranking.py --set val
python scripts/07_evaluate_ranking.py --set test
```

## 核心产物说明

### `data/processed`

- 原始交互切分结果
- 用户画像、物品画像
- 排序训练候选池

### `data/feature_store`

- 类别词表
- 连续特征 scaler
- 热门物品列表
- genre 倒排索引

### `data/model_weights`

- 双塔权重
- ItemCF 相似度矩阵
- UserCF 相似度矩阵
- 排序模型权重

## 评估视角

### 召回评估

关注“目标物品是否能进候选池”：

- `Recall@50`
- `Recall@100`
- `NDCG@50`

### 排序评估

关注“候选池内能否把目标排到更前面”：

- `HR@10/20/50`
- `NDCG@10/20/50`
- `MRR`

## 已知前提和注意事项

- `05_build_ranking_data.py` 默认依赖前面训练好的召回模型，尤其是 `DualTower` 和 `ItemCF`。
- 仓库内目前只有 TMDB 合并脚本，没有在线抓取 TMDB 的下载脚本。
- `faiss-cpu` 已写进依赖；如果你准备在 GPU 环境做更大规模实验，可以自行替换为适合环境的版本。
- 当前很多参数是按“大机器”思路配置的，`config.sample.yaml` 里默认值对普通笔记本可能偏大，必要时请降低：
  - `training.batch_size`
  - `training.inbatch_neg_size`
  - `ranking.batch_size`
  - `ranking.num_workers`
- `src/features/feature_builder.py` 目前是预留接口，实际特征构建逻辑主要由 `01_process_data.py` 和 `02_build_features.py` 承担。

## 后续扩展建议

- 增加 TMDB 数据抓取与校验脚本
- 把 `04/07` 的评估结果统一落盘成报表
- 增加实验配置版本管理
- 为训练和评估补充更明确的日志与 checkpoint 策略

## 参考文档

- 设计草稿见 [docs/design_spec.md](docs/design_spec.md)
- 核心配置见 [config.sample.yaml](config.sample.yaml)
