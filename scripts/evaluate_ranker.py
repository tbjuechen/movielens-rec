import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from src.features.ranking_feature_engine import RankingFeatureEngine
from src.data_loader_listwise import ListwiseRankingDataset
from src.models.ranking.deep_model import UnifiedDeepRanker

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg: return 0.
    return dcg_at_k(r, k) / idcg

def evaluate_ranker():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = Path("saved_models/unified_ranker")
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"

    # 1. 加载元数据与模型
    logger.info(f"正在从 {model_dir} 加载模型...")
    with open(model_dir / "model_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    
    model = UnifiedDeepRanker(
        meta['feature_map'], 
        embedding_dim=meta['embedding_dim']
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))
    model.eval()

    # 2. 准备验证集 (严格隔离最后 20%)
    samples_df = pd.read_parquet(ranking_dir / "ranking_samples_prototype.parquet")
    samples_df = samples_df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(samples_df)
    history_ratings = samples_df.iloc[:int(n*0.6)]
    history_ratings = history_ratings[history_ratings['label'] == 1][['userId', 'movieId', 'rating', 'timestamp']]
    
    val_samples = samples_df.iloc[int(n*0.8):].copy()
    
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(history_ratings)
    
    logger.info("提取验证集特征矩阵...")
    val_df = engine.build_feature_matrix(val_samples)
    item_profile = pd.read_parquet(ranking_dir / "item_profile_ranking.parquet")
    
    # 3. 数据加载器 (Listwise)
    val_ds = ListwiseRankingDataset(val_df, item_profile)
    # 覆盖元数据
    val_ds.mid_map = meta['mid_map']
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # 4. 指标计算
    metrics = {'top1_acc': [], 'mrr': [], 'ndcg5': []}
    
    logger.info("开始性能评估...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            B, K = batch['movieId'].shape
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ['click_label', 'rating_label']}
            
            outputs = model(inputs)
            logits = outputs['click'].view(B, K) # (Batch, 5)
            
            for i in range(B):
                scores = logits[i].cpu().numpy()
                # 真实的正样本在索引 0
                # 公平排名计算：处理平分
                num_strictly_greater = (scores > scores[0]).sum()
                num_equal = (scores == scores[0]).sum()
                rank = num_strictly_greater + (num_equal + 1) / 2.0
                
                metrics['top1_acc'].append(1 if rank <= 1.5 else 0)
                metrics['mrr'].append(1.0 / rank)
                
                # NDCG 计算
                pred_order = np.argsort(-scores)
                binary_rel = [1 if idx == 0 else 0 for idx in pred_order]
                metrics['ndcg5'].append(ndcg_at_k(binary_rel, 5))

    # 5. 输出报告
    logger.success("性能评估完成。")
    print("\n" + "="*40)
    print("精排模型性能报告 (Validation Set)")
    print("-" * 40)
    print(f"Top-1 Accuracy: {np.mean(metrics['top1_acc']):.4f}")
    print(f"Mean RR (MRR):  {np.mean(metrics['mrr']):.4f}")
    print(f"NDCG@5:         {np.mean(metrics['ndcg5']):.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_ranker()
