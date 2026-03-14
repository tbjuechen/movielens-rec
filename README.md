# MovieLens Recommendation System v2

本项目是一个基于 [MovieLens-32M](https://grouplens.org/datasets/movielens/) 和 TMDB 外部数据的工业级推荐系统实验平台。旨在模拟搜推实习面试中的召回与排序实战场景。

## 核心特性
- **数据工程**：处理 3200 万级交互数据，融合 TMDB 票房、语言、演职员等多维特征。
- **召回模型**：
    - **双塔模型 (Dual-Tower)**：支持 In-batch 负采样、全局随机采样、热门负采样。
    - **混合损失函数**：结合 InfoNCE (全域对比) 与 BPR (困难对排序)。
    - **特征纠偏**：应用 Log-Q Correction 修正采样偏差。
    - **时序感知**：用户历史序列采用时间衰减加权 (Time-Decay Weighting)。
- **高性能组件**：特征向量化处理、Parquet 存储格式、PyTorch 加速训练。

## 项目结构
```bash
.
├── data/               # 原始数据与处理后的 Parquet 宽表 (Git 忽略)
├── docs/               # 详细的技术方案与实验报告
│   └── design_spec.md  # 初始设计方案与特征工程定义
├── scripts/            # 一键执行流水线 (01-04 脚本)
├── src/                # 核心源代码 (特征工程、模型、数据流)
└── requirements.txt    # 项目依赖
```

## 快速开始

### 1. 环境配置
推荐使用 Conda 环境：
```bash
conda activate movielens-rec
# 或手动安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
1. 下载 [MovieLens-32M](https://grouplens.org/datasets/movielens/) 并解压到 `data/raw/ml-32m`。
2. 将 TMDB 的 JSON 缓存放置在 `data/raw/tmdb_cache`。

### 3. 一键执行流水线 (Commands)
按顺序执行以下脚本完成端到端流程：

#### Step 0: 聚合 TMDB 缓存数据 (加速 IO)
将 8 万个小 JSON 文件合并为高效的 Parquet 格式：
```bash
conda run -n movielens-rec python -c "
import os, json, pandas as pd; from tqdm import tqdm; \
tmdb_dir = 'data/raw/tmdb_cache/'; \
files = [f for f in os.listdir(tmdb_dir) if f.endswith('.json')]; \
data_list = [json.load(open(os.path.join(tmdb_dir, f))) for f in tqdm(files)]; \
df = pd.DataFrame([{ 'tmdb_id': d.get('id'), 'imdb_id': d.get('imdb_id'), 'original_language': d.get('original_language'), 'budget': d.get('budget', 0), 'revenue': d.get('revenue', 0), 'runtime': d.get('runtime', 0), 'vote_average': d.get('vote_average', 0.0), 'vote_count': d.get('vote_count', 0), 'tmdb_genres': [g['name'] for g in d.get('genres', [])] } for d in data_list]); \
df.to_parquet('data/processed/tmdb_features.parquet', index=False)"
```

#### Step 1: 数据清洗与特征预处理 (生成宽表)
执行 MovieLens 与 TMDB 数据合并、特征 Log 变换、年份分箱、数据集 Leave-One-Out 切分：
```bash
conda run -n movielens-rec python src/data_pipeline/preprocessor.py
```

#### Step 2: 启动双塔召回模型训练
训练带 Log-Q 纠偏、混合 Loss (InfoNCE + BPR)、时间衰减加权的召回模型：
```bash
conda run -n movielens-rec python scripts/03_train_models.py
```

#### Step 3: 召回指标评测 (待开发)
基于训练好的权重生成全量物品向量库，在测试集上计算 Recall@50：
```bash
conda run -n movielens-rec python scripts/04_evaluate_e2e.py
```

## 技术文档索引
- [技术方案与字段定义 (docs/design_spec.md)](docs/design_spec.md)
- [指标评估报告 (待生成)](docs/evaluation_report.md)

---
2026.3.14 - 搜推实习项目准备
