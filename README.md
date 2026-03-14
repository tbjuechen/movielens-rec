# movielens 推荐项目

2026.3.6 搜推实习项目准备

## 一、项目结构

```jsx
movielens_recsys/
├── data/                      # 数据存放目录 (需在 .gitignore 中忽略)
│   ├── raw/                   # 原始 MovieLens 数据 
│   ├── processed/             # 清洗切分后的离线数据集 (train/val/test)
│   ├── feature_store/         # 【模拟线上】预处理好的静态特征字典
│   └── model_weights/         # 训练好的模型权重与索引
├── notebooks/                 # Jupyter Notebooks (实验与探索)
├── src/                       # 核心源代码目录
│   ├── config/                # 全局配置
│   │   └── settings.py        # 路径、特征维度、召回K值等
│   ├── data_pipeline/         # 离线数据处理流
│   │   ├── preprocessor.py    # 清洗去噪、数据集切分
│   │   └── dataset.py         # 构建 Dataset/DataLoader
│   ├── features/              # 特征工程模块
│   │   ├── encoder.py         # 类别编码、连续值归一化
│   │   └── feature_builder.py # 提取特征并落盘到 feature_store
│   ├── models/                # 模型层 (仅保留抽象接口)
│   │   ├── recall/            
│   │   │   ├── base.py        # 召回基类 (定义 fit, retrieve 标准接口)
│   │   │   └── merger.py      # 多路召回的结果融合与截断
│   │   └── ranking/           
│   │       └── base.py        # 排序基类 (定义 fit, predict 标准接口)
│   ├── evaluation/            # 评估模块
│   │   └── metrics.py         # 评估指标实现
│   └── pipeline/              # 端到端推断与评测链路
│       ├── retriever.py       # 召回调度器 (面向 recall/base.py 编程)
│       ├── feature_fetcher.py # 特征组装器 (从 feature_store 拉取数据)
│       ├── ranker.py          # 排序调度器 (面向 ranking/base.py 编程)
│       └── evaluator.py       # 链路总控，输出最终指标报表
├── scripts/                   # 一键执行流水线
│   ├── 01_process_data.py     # 运行清洗与切分
│   ├── 02_build_features.py   # 生成并存储特征库
│   ├── 03_train_models.py     # 触发模型训练 (召回+排序)
│   └── 04_evaluate_e2e.py     # 执行端到端评测
├── requirements.txt           
└── README.md
```

## 二、数据准备

- MovieLens-32M https://grouplens.org/datasets/movielens/
    - movies.csv 电影表：movieID, title, genres
    - ratings.csv 交互表：userID, movieID, rating, ts
    - tags.csv 交互表：userID, movieID, tag, ts
    - links.csv 电影外部数据库表：movieID, imdbID, tmdbID
- IMDB https://www.imdb.com/

通过爬虫脚本和IMDB官方api获取额外的电影特征

| **字段名 (Field)** | **数据类型 (Type)** | **描述 (Description)** |
| --- | --- | --- |
| **`id`** | Number | 平台内部标识符 |
| **`imdb_id`** | String | IMDB 标识符 |
| **`title`** | String | 片名 |
| **`original_title`** | String | 原片名 |
| **`overview`** | String | 剧情简介 |
| **`genres`** | Array[Object] | 类型数组 |
| **`credits`** | Object | 演职人员表 |
| **`keywords`** | Object | 关键词 |
*(...更多详见IMDB)*

## 三、召回模型

### 3.1 召回通道
- 双塔模型
- ItemCF
- UserCF
- 热门召回
- 标签召回

### 3.2 双塔模型
- 用户塔
    - user_id：用户id嵌入
    - avg_rating：用户历史平均打分
    - genres：用户top3类别兴趣
    - history：历史交互item加权和
    - activity：用户活跃度
- 物品塔
    - item_id：物品id嵌入
    - genres：物品类别嵌入和
    - release_year：发布时间
    - language：对白语言
    - avg_rating：movielens平均评分
    - imdb_avg_rating：imdb平均评分
    - revenue：电影票房
- 采样方式
    - in-batch负采样 + 全集采样
    - 困难负样本 (热门未交互物品, 交互低分物品)
- 损失函数
    - infoNCE
    - BPR
- 评测指标
    - recal@50
