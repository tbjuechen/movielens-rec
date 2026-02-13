# MovieLens 全流程推荐系统 (movielens-rec)

本项目基于 MovieLens 32M 数据集，构建一个从原始数据处理、多路召回、精细化排序到端到端推理的工业级推荐系统。

## 1. 环境配置

本项目支持 macOS (MPS) 和 NVIDIA (CUDA) 硬件加速。

### 创建环境并安装依赖
```bash
conda create -y -n movielens-rec python=3.10
conda activate movielens-rec
# 安装全量依赖
conda install -y -c pytorch -c conda-forge pandas pyarrow loguru jupyter matplotlib seaborn scikit-learn pytorch faiss-cpu tqdm scipy python-dotenv beautifulsoup4 requests
# 注册 Jupyter 内核
python -m ipykernel install --user --name movielens-rec --display-name "Python 3.10 (movielens-rec)"
```

## 2. 数据工程流水线

### 2.1 基础预处理
将 32M CSV 转换为高效的 Parquet 格式，并完成时间切分 (80/10/10) 与 ID 重映射。
```bash
PYTHONPATH=. python scripts/run_preprocessing.py
PYTHONPATH=. python scripts/prepare_two_tower_data.py
```

### 2.2 外部元数据采集 (TMDb)
通过 TMDb API 补全电影的剧情、关键词、导演、主演等深度特征。
1. **配置 Key**：在 `.env` 中填入 `TMDB_API_KEY=xxx`。
2. **高效抓取**：使用 50 线程并发抓取全量 JSON。
   ```bash
   PYTHONPATH=. python scripts/data_collector/run_batch_crawl.py
   ```
3. **星型建模**：将散乱的 JSON 整合为结构化的电影、人员、关系表。
   ```bash
   PYTHONPATH=. python scripts/data_collector/process_tmdb_json.py
   ```

## 3. 召回层 (Recall)

支持多路并发召回，核心指标 **HitRate@50** 已达到 **0.35**。

- **热门召回 (Popularity)**：带“已看过滤”的全局基准路。
- **协同过滤 (ItemCF)**：基于 `scipy` 稀疏矩阵的个性化关联路。
- **多模态双塔 (Two-Tower V2)**：融合 ID、题材和统计特征的深度学习核心路。

```bash
# 一键评估召回率
PYTHONPATH=. python scripts/evaluate_recall.py
```

## 4. 排序层 (Ranking) - 开发中
利用采集到的 TMDb 深度特征，通过特征交叉与 CTR 预估实现精细化排序。

---

## 项目架构与规范

- **星型模型**：外部元数据按 `Movies`, `Persons`, `Cast`, `Crew` 四张表存储，确保数据无冗余。
- **原子提交**：严格遵循一个提交解决一个问题的原则。
- **MPS 优化**：针对 Apple 芯片手动实现 pooling 算子，绕过 PyTorch 兼容性限制。
