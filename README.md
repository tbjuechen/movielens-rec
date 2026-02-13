# MovieLens 全流程推荐系统 (movielens-rec)

本项目基于 MovieLens 32M 数据集，构建一个包含召回、排序、重排和推理流水线的完整工业级推荐系统。

## 1. 环境配置

本项目使用 Conda 管理虚拟环境，支持 macOS (MPS) 和 NVIDIA (CUDA) 加速。

### 创建环境并安装依赖
```bash
conda create -y -n movielens-rec python=3.10
conda activate movielens-rec
# 安装基础及深度学习依赖
conda install -y -c pytorch -c conda-forge pandas pyarrow loguru jupyter matplotlib seaborn scikit-learn pytorch faiss-cpu tqdm scipy python-dotenv beautifulsoup4 requests
# 注册 Jupyter 内核
python -m ipykernel install --user --name movielens-rec --display-name "Python 3.10 (movielens-rec)"
```

### 外部数据配置 (TMDb)
1. 在项目根目录下创建 `.env` 文件。
2. 填入你的 TMDb API Key:
   ```text
   TMDB_API_KEY=your_api_key_here
   ```

## 2. 数据处理流水线

在运行任何模型之前，必须按顺序执行以下预处理脚本。

### 2.1 基础格式转换 (CSV -> Parquet)
```bash
PYTHONPATH=. python scripts/run_preprocessing.py
```

### 2.2 双塔模型特征工程 (V2)
```bash
PYTHONPATH=. python scripts/prepare_two_tower_data.py
```

### 2.3 外部元数据采集 (可选，用于 Ranking 特征增强)
```bash
# 1. 批量抓取 (支持断点续传，默认 50 线程)
PYTHONPATH=. python scripts/data_collector/run_batch_crawl.py

# 2. 星型建模整合 (JSON -> 4张关联 Parquet 表)
PYTHONPATH=. python scripts/data_collector/process_tmdb_json.py
```

## 3. 召回模块 (Recall)

本项目支持多路召回通道，所有模型均继承自 `BaseRecall` 接口。

### 3.1 热门召回 (Popularity)
```bash
PYTHONPATH=. python scripts/train_recall.py --model popularity --input data/processed/two_tower/train.parquet
```

### 3.2 物品协同过滤 (ItemCF)
```bash
PYTHONPATH=. python scripts/train_recall.py --model itemcf --input data/processed/two_tower/train.parquet
```

### 3.3 多模态双塔模型 (Two-Tower V2)
```bash
PYTHONPATH=. python scripts/train_recall.py --model two_tower --epochs 5 --batch_size 8192
```

## 4. 模型评估与测试

### 4.1 召回率对比 (Hit Rate@K)
```bash
PYTHONPATH=. python scripts/evaluate_recall.py
```

### 4.2 案例验证
```bash
# 验证双塔模型效果
PYTHONPATH=. python examples/test_recall_two_tower.py
```

---

## 模型特征说明 (V2)

| 塔 (Tower) | ID 特征 | 内容特征 | 统计特征 |
| :--- | :--- | :--- | :--- |
| **User Tower** | userId (128d) | Top-3 题材偏好 | 平均分、活跃度 |
| **Item Tower** | movieId (128d) | 电影题材 (Multi-label) | 年份、平均分、流行度 |

## 开发规范

- **原子提交**：每个提交只实现一个功能点。
- **星型建模**：外部采集的数据采用电影、人员、关系表分离的星型模型存储。
- **文档规范**：新脚本上线后，必须同步更新 `README.md` 和 `GEMINI.md`。
