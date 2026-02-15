# MovieLens 全流程推荐系统 (movielens-rec)

本项目基于 MovieLens 32M 数据集，构建一个从原始数据处理、多路召回、精细化排序到端到端推理的工业级推荐系统。

## 1. 环境配置

本项目支持 macOS (MPS) 和 NVIDIA (CUDA) 硬件加速。

### 创建环境并安装依赖
```bash
conda create -y -n movielens-rec python=3.10
conda activate movielens-rec
# 安装全量依赖
conda install -y -c pytorch -c conda-forge pandas pyarrow loguru jupyter matplotlib seaborn scikit-learn pytorch faiss-cpu tqdm scipy python-dotenv beautifulsoup4 requests openai
# 注册 Jupyter 内核
python -m ipykernel install --user --name movielens-rec --display-name "Python 3.10 (movielens-rec)"
```

### 外部数据配置 (TMDb / OpenAI)
在项目根目录下创建 `.env` 文件并填入：
```text
TMDB_API_KEY=xxx
OPENAI_API_KEY=xxx
OPENAI_BASE_URL=xxx
EMBEDDING_MODEL=BAAI/bge-m3
```

## 2. 数据工程流水线

### 2.1 基础预处理 (MovieLens)
```bash
PYTHONPATH=. python scripts/run_preprocessing.py
PYTHONPATH=. python scripts/prepare_two_tower_data.py
```

### 2.2 深度特征采集 (TMDb & Embedding)
```bash
# 1. 50 线程并发采集 TMDb 元数据 JSON
PYTHONPATH=. python scripts/data_collector/run_batch_crawl.py

# 2. 星型建模整合为 Parquet 关联表
PYTHONPATH=. python scripts/data_collector/process_tmdb_json.py

# 3. 语义向量化 (获取 1024 维剧情简介向量)
PYTHONPATH=. python scripts/data_collector/generate_text_embeddings.py
```

## 3. 召回层 (Recall)

支持多路并发召回，核心指标 **HitRate@50** 已达到 **0.35**。

### 3.1 训练召回模型
```bash
# 1. 训练热门召回 (Popularity)
PYTHONPATH=. python scripts/train_recall.py --model popularity --input data/processed/two_tower/train.parquet

# 2. 训练物品协同过滤 (ItemCF)
PYTHONPATH=. python scripts/train_recall.py --model itemcf --input data/processed/two_tower/train.parquet

# 3. 训练多模态双塔模型 (Two-Tower V2)
PYTHONPATH=. python scripts/train_recall.py --model two_tower --epochs 5 --batch_size 8192
```

### 3.2 一键对比评估
```bash
PYTHONPATH=. python scripts/evaluate_recall.py
```

## 4. 排序层 (Ranking)

本项目实现了从基础 XGBoost 到顶级深度模型 (UnifiedRanker Pro) 的全进化。

### 4.1 训练精排模型
```bash
# 1. 训练 XGBoost 基准模型 (AUC 约 0.67)
PYTHONPATH=. python scripts/train_ranker_xgboost.py

# 2. 训练 UnifiedRanker Pro (DIN + DCN-V2 + MMoE)
# 支持 Listwise 偏好学习与多任务预测
PYTHONPATH=. python scripts/train_ranker_mmoe.py
```

---

## 项目架构与规范

- **星型模型**：外部元数据按 `Movies`, `Persons`, `Cast`, `Crew` 四张表存储。
- **分层处理**：语义向量采集采用 API Batch (64) 与 Checkpoint Chunk (500) 的分层策略。
- **MPS 优化**：针对 Apple 芯片手动实现 pooling 算子。

## 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源协议。
