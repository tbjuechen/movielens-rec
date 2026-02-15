import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from src.features.ranking_feature_engine import RankingFeatureEngine
from src.data_loader_listwise import ListwiseRankingDataset
from src.models.ranking.deep_model import UnifiedDeepRanker

def train_click_only():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    
    # 1. 准备数据
    samples_df = pd.read_parquet(ranking_dir / "ranking_samples_prototype.parquet")
    samples_df = samples_df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(samples_df)
    history_end, train_end = int(n * 0.6), int(n * 0.8)
    history_ratings = samples_df.iloc[:history_end]
    history_ratings = history_ratings[history_ratings['label'] == 1][['userId', 'movieId', 'rating', 'timestamp']]
    
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(history_ratings)
    
    logger.info("提取特征矩阵...")
    train_df = engine.build_feature_matrix(samples_df.iloc[history_end:train_end])
    val_df = engine.build_feature_matrix(samples_df.iloc[train_end:])
    item_profile = pd.read_parquet(ranking_dir / "item_profile_ranking.parquet")
    
    # 2. Dataset & Loader
    train_ds = ListwiseRankingDataset(train_df, item_profile)
    val_ds = ListwiseRankingDataset(val_df, item_profile)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    # 3. 初始化模型
    feature_map = {
        'sparse': {'movieId': train_ds.vocab_size},
        'dense': len(train_ds.dense_cols)
    }
    model = UnifiedDeepRanker(feature_map, embedding_dim=128).to(device)
    
    # 使用较大的学习率打破数值平衡
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练
    for epoch in range(5):
        model.train()
        total_loss, correct_top1, total_samples = 0, 0, 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                B, K = batch['movieId'].shape
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'click_label'}
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                logits = outputs['click'].view(B, K)
                loss = criterion(logits, batch['click_label'].to(device))
                
                if torch.isnan(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 修复统计逻辑：累加样本总数
                preds = torch.argmax(logits, dim=1)
                correct_top1 += (preds == 0).sum().item()
                total_loss += loss.item()
                total_samples += B
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct_top1/total_samples:.4f}"})

        logger.success(f"Epoch {epoch+1} 训练结束。")

    # 5. 保存
    model_dir = Path("saved_models/unified_ranker")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    import pickle
    with open(model_dir / "model_meta.pkl", "wb") as f:
        pickle.dump({
            'feature_map': feature_map,
            'embedding_dim': 128,
            'dense_cols': train_ds.dense_cols,
            'mid_map': train_ds.mid_map
        }, f)
    logger.success(f"模型已保存至 {model_dir}")

if __name__ == "__main__":
    train_click_only()
