# MovieLens 全流程推荐系统 (movielens-rec)

本项目基于 MovieLens 32M 数据集，构建一个包含召回、排序、重排和推理流水线的完整工业级推荐系统。

## 1. 环境配置

本项目使用 Conda 管理虚拟环境，支持 macOS (MPS) 和 NVIDIA (CUDA) 加速。

### 创建环境并安装依赖
```bash
conda create -y -n movielens-rec python=3.10
conda activate movielens-rec
# 安装基础及深度学习依赖
conda install -y -c pytorch -c conda-forge pandas pyarrow loguru jupyter matplotlib seaborn scikit-learn pytorch faiss-cpu tqdm scipy
# 注册 Jupyter 内核
python -m ipykernel install --user --name movielens-rec --display-name "Python 3.10 (movielens-rec)"
```

## 2. 数据处理流水线

在运行任何模型之前，必须按顺序执行以下预处理脚本。

### 2.1 基础格式转换 (CSV -> Parquet)
提取电影年份，清洗题材，提升读取速度。
```bash
PYTHONPATH=. python scripts/run_preprocessing.py
```

### 2.2 双塔模型特征工程 (V2)
执行时间切分 (80/10/10)、ID 重映射，并生成**富特征表**（包含题材偏好、平均评分、活跃度等）。
```bash
PYTHONPATH=. python scripts/prepare_two_tower_data.py
```

## 3. 召回模块 (Recall)

本项目支持多路召回通道，所有模型均继承自 `BaseRecall` 接口。

### 3.1 热门召回 (Popularity)
非个性化基准路，支持过滤用户已看过的电影。
```bash
PYTHONPATH=. python scripts/train_recall.py --model popularity --input data/processed/two_tower/train.parquet
```

### 3.2 物品协同过滤 (ItemCF)
基于共现关系的个性化召回，使用 `scipy` 稀疏矩阵加速计算。
```bash
PYTHONPATH=. python scripts/train_recall.py --model itemcf --input data/processed/two_tower/train.parquet
```

### 3.3 多模态双塔模型 (Two-Tower V2)
深度学习召回核心，融合 ID、题材 (Embedding Bag) 和统计特征。
```bash
# 建议配置：batch_size 越大，负采样效果越好
PYTHONPATH=. python scripts/train_recall.py --model two_tower --epochs 5 --batch_size 8192
```

## 4. 模型评估与测试

### 4.1 召回率对比 (Hit Rate@K)
在测试集上一键对比三路召回的命中率。
```bash
PYTHONPATH=. python scripts/evaluate_recall.py
```

### 4.2 案例验证
在 `examples/` 目录下运行脚本，查看针对具体用户的真实推荐标题。
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
- **MPS 兼容**：深度学习算子优先考虑 macOS 芯片的兼容性（如避免使用原生 EmbeddingBag）。
- **文档规范**：新脚本上线后，必须同步更新 `README.md` 和 `GEMINI.md`。
