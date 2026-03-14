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

### 3. 运行流水线
按顺序执行以下脚本完成端到端流程：

```bash
# 01. 数据预处理：清洗、特征提取、数据集 Leave-One-Out 切分
python src/data_pipeline/preprocessor.py

# 02. 特征构建：生成特征词表、归一化器 (可选，训练脚本内包含)
python scripts/02_build_features.py

# 03. 召回模型训练：启动带纠偏的双塔模型训练
python scripts/03_train_models.py

# 04. 召回评测：评估 Recall@50 等核心指标 (正在开发中)
python scripts/04_evaluate_e2e.py
```

## 技术文档索引
- [技术方案与字段定义 (docs/design_spec.md)](docs/design_spec.md)
- [指标评估报告 (待生成)](docs/evaluation_report.md)

---
2026.3.14 - 搜推实习项目准备
