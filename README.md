# MovieLens 全流程推荐系统 (movielens-rec)

本项目基于 MovieLens 32M 数据集，构建一个包含召回、排序、重排和推理流水线的完整推荐系统。

## 1. 环境配置

本项目使用 Conda 管理虚拟环境。

### 创建环境并安装依赖
```bash
conda create -y -n movielens-rec python=3.10
conda activate movielens-rec
# 安装基础依赖
conda install -y -c conda-forge pandas pyarrow loguru jupyter matplotlib seaborn
# 注册 Jupyter 内核
python -m ipykernel install --user --name movielens-rec --display-name "Python 3.10 (movielens-rec)"
```

## 2. 数据准备与预处理

在运行任何模型之前，需要将原始 CSV 数据转换为高效的 Parquet 格式。

### 执行预处理脚本
```bash
# 确保在项目根目录下
PYTHONPATH=. conda run -n movielens-rec python scripts/run_preprocessing.py
```
该操作会将 `data/ml-32m/` 下的文件处理后存入 `data/processed/`。

## 3. 数据探索 (EDA)

推荐在 VS Code 中打开 `notebooks/01_eda_and_viz.ipynb` 进行交互式分析。
确保选择的内核为 `Python 3.10 (movielens-rec)`。

## 4. 召回模块 (Recall)

### 数据准备 (双塔模型专用)
双塔模型需要进行 ID 连续化编码及时间切分：
```bash
PYTHONPATH=. conda run -n movielens-rec python scripts/prepare_two_tower_data.py
```

### 热门召回 (Popularity Recall)
统计全局最热门的电影榜单。

**执行训练/统计脚本**:
```bash
# 训练热门召回模型
PYTHONPATH=. conda run -n movielens-rec python scripts/train_recall.py --model popularity

# 训练 ItemCF 召回模型
PYTHONPATH=. conda run -n movielens-rec python scripts/train_recall.py --model itemcf

# 训练双塔召回模型
PYTHONPATH=. conda run -n movielens-rec python scripts/train_recall.py --model two_tower --epochs 5
```

## 5. 测试与演示

可以使用 `examples/` 目录下的脚本快速验证模型效果：

```bash
# 测试热门召回效果
PYTHONPATH=. conda run -n movielens-rec python examples/test_recall_popularity.py

# 测试 ItemCF 个性化召回效果
PYTHONPATH=. conda run -n movielens-rec python examples/test_recall_itemcf.py

# 测试双塔模型召回效果
PYTHONPATH=. conda run -n movielens-rec python examples/test_recall_two_tower.py
```

---

## 开发规范

- **原子提交**：每个提交只实现一个功能点。
- **日志规范**：统一使用 `loguru`。
- **文档规范**：新脚本上线后，必须同步更新 `README.md` 中的命令示例。
