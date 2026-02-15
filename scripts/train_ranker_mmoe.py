import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from src.features.ranking_feature_engine import RankingFeatureEngine
from src.data_loader_listwise import ListwiseRankingDataset
from src.models.ranking.deep_model import UnifiedDeepRanker

def train_mmoe():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    
    # 1. 准备数据
    samples_df = pd.read_parquet(ranking_dir / "ranking_samples_prototype.parquet")
    samples_df = samples_df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(samples_df)
    history_end, train_end = int(n * 0.6), int(n * 0.8)
    
    # 修正警告：先取切片，再在切片内过滤
    history_df = samples_df.iloc[:history_end]
    history_ratings = history_df[history_df['label'] == 1][['userId', 'movieId', 'rating', 'timestamp']]
    engine.initialize(history_ratings)
    
    train_df = engine.build_feature_matrix(samples_df.iloc[history_end:train_end])
    val_df = engine.build_feature_matrix(samples_df.iloc[train_end:])
    item_profile = pd.read_parquet(ranking_dir / "item_profile_ranking.parquet")
    
    # 2. Dataset & Loader
    train_ds = ListwiseRankingDataset(train_df, item_profile)
    val_ds = ListwiseRankingDataset(val_df, item_profile)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    # 3. 初始化『完全体』模型
    feature_map = {
        'sparse': {'movieId': train_ds.vocab_size},
        'dense': 4
    }
    model = UnifiedDeepRanker(
        feature_map, 
        embedding_dim=128,  # 加大参数量
        num_experts=4,      # MMoE 专家
        tasks=['click', 'rating']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    click_criterion = nn.CrossEntropyLoss()
    rating_criterion = nn.MSELoss()

    # 4. 训练
    for epoch in range(5):
        model.train()
        total_loss, correct_top1 = 0, 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                B, K = batch['movieId'].shape
                inputs = {k: v.to(device) for k, v in batch.items() if k not in ['click_label', 'rating_label']}
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # A. Click Loss (Listwise)
                click_logits = outputs['click'].view(B, K)
                click_loss = click_criterion(click_logits, batch['click_label'].to(device))
                
                # B. Rating Loss (Pointwise MSE)
                rating_preds = outputs['rating'] # (B*K)
                rating_labels = batch['rating_label'].view(-1).to(device)
                # 只有正样本计算 Rating Loss (可选策略)
                rating_loss = rating_criterion(rating_preds, rating_labels)
                
                # 联合优化
                loss = click_loss + 0.1 * rating_loss
                loss.backward()
                optimizer.step()
                
                # 统计
                preds = torch.argmax(click_logits, dim=1)
                correct_top1 += (preds == 0).sum().item()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct_top1/((pbar.n+1)*B):.4f}"})

        logger.success(f"Epoch {epoch+1} 训练结束。")

if __name__ == "__main__":
    train_mmoe()
