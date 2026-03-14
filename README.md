# MovieLens Recommendation System v2

本项目是一个基于 [MovieLens-32M](https://grouplens.org/datasets/movielens/) 和 TMDB 外部数据的工业级推荐系统召回模型方案。它模拟了真实业务中处理千万级数据量、多路召回融合以及深度兴趣建模的实战场景。

## 🚀 技术亮点

### 1. 召回算法 (Retrieval Strategies)
- **深度双塔 (Neural Dual-Tower)**: 
    - **共享嵌入层**: 用户历史序列与物品 ID 强共享 Embedding，增强语义一致性。
    - **混合损失 (Mixed Loss)**: 同时使用 InfoNCE (全域对比) 和 BPR (困难对排序)。
    - **采样纠偏 (Log-Q Correction)**: 修正 In-batch 负采样带来的流行度偏差 (Popularity Bias)。
    - **时序加权 (Time-Decay)**: 采用指数衰减函数对用户最近观看历史进行动态加权。
- **协同过滤 (Collaborative Filtering)**: 实现并优化了带 IUF/IIF 惩罚的 **ItemCF** 和 **UserCF**。
- **冷启动召回**: 包含 **Genre-based (标签)** 和 **Popularity (热门)** 兜底通道。

### 2. 工程优化 (Engineering)
- **高性能 IO**: 将 8.6 万个小文件与 32M 行 CSV 聚合为 **Parquet** 宽表，读取速度提升 100x。
- **向量化处理**: 特征编码与纠偏计算全量采用 **Pandas/NumPy 向量化映射**。
- **混合负采样**: 每个 Batch 同时包含 In-batch、Global Uniform 和 Popular Hard Negatives。

## 📂 项目结构
```bash
.
├── data/               # 数据存储 (processed/ 内存放生成的宽表)
├── docs/               # 技术文档与实验报告
├── scripts/            # 执行流水线 (01-04 入口脚本)
├── src/
│   ├── data_pipeline/  # 预处理与 Dataset 构建
│   ├── features/       # 特征编码器
│   ├── models/         # 召回模型 (DualTower, ItemCF, UserCF, Merger)
│   └── pipeline/       # 离线评估链路
└── GEMINI.md           # 项目开发规范与共识
```

## 🛠 快速开始

### 1. 环境准备
```bash
conda activate movielens-rec
pip install -r requirements.txt
```

### 2. 数据流水线 (Pipeline)

#### Step 0: 聚合 TMDB 数据
将 8 万个小 JSON 合并为 Parquet 格式：
```bash
conda run -n movielens-rec python -c "import os, json, pandas as pd; from tqdm import tqdm; tmdb_dir = 'data/raw/tmdb_cache/'; files = [f for f in os.listdir(tmdb_dir) if f.endswith('.json')]; data_list = [json.load(open(os.path.join(tmdb_dir, f))) for f in tqdm(files)]; df = pd.DataFrame([{ 'tmdb_id': d.get('id'), 'imdb_id': d.get('imdb_id'), 'original_language': d.get('original_language'), 'budget': d.get('budget', 0), 'revenue': d.get('revenue', 0), 'runtime': d.get('runtime', 0), 'vote_average': d.get('vote_average', 0.0), 'vote_count': d.get('vote_count', 0), 'tmdb_genres': [g['name'] for g in d.get('genres', [])] } for d in data_list]); df.to_parquet('data/processed/tmdb_features.parquet', index=False)"
```

#### Step 1: 预处理与切分
生成宽表、进行特征 Log 变换、完成用户时间线 Leave-One-Out 切分：
```bash
conda run -n movielens-rec python scripts/01_process_data.py
```

#### Step 2: 特征词表固化
统计全局词表、拟合归一化器，固化 Encoders：
```bash
conda run -n movielens-rec python scripts/02_build_features.py
```

#### Step 3: 模型训练
支持三路训练/离线计算：
```bash
# 训练双塔神经网络 (支持 GPU/MPS 加速)
conda run -n movielens-rec python scripts/03_train_models.py --model dual_tower

# 计算 ItemCF / UserCF 相似度矩阵
conda run -n movielens-rec python scripts/03_train_models.py --model item_cf
conda run -n movielens-rec python scripts/03_train_models.py --model user_cf
```

#### Step 4: 指标评测 (开发中)
构建 FAISS 索引并计算测试集 Recall@50/NDCG@50。

---
2026.3.14 - 搜推实习准备项目
