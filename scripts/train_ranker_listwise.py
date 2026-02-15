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

def train_listwise():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    
    # 1. 加载样本
    samples_df = pd.read_parquet(ranking_dir / "ranking_samples_prototype.parquet")
    samples_df = samples_df.sort_values('timestamp').reset_index(drop=True)
    
    # 隔离逻辑
    n = len(samples_df)
    history_end = int(n * 0.6)
    train_end = int(n * 0.8)
    
    history_ratings = samples_df.iloc[:history_end]
    history_ratings = history_ratings[history_ratings['label'] == 1][['userId', 'movieId', 'rating', 'timestamp']]
    
    train_samples = samples_df.iloc[history_end:train_end].copy()
    val_samples = samples_df.iloc[train_end:].copy()

    # 2. 特征引擎
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(history_ratings)
    
    logger.info("生成 Listwise 特征矩阵...")
    train_df = engine.build_feature_matrix(train_samples)
    val_df = engine.build_feature_matrix(val_samples)
    
    item_profile = pd.read_parquet(ranking_dir / "item_profile_ranking.parquet")
    
    # 3. Listwise Dataset
    train_ds = ListwiseRankingDataset(train_df, item_profile, neg_ratio=4)
    val_ds = ListwiseRankingDataset(val_df, item_profile, neg_ratio=4)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    # 4. 模型初始化 (模型架构保持 Pointwise，但在训练循环中 Reshape)
    feature_map = {
        'sparse': {'movieId': train_ds.vocab_size},
        'dense': 4
    }
    model = UnifiedDeepRanker(feature_map, embedding_dim=32, tasks=['click']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # Listwise 的核心

    # 5. 训练
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct_top1 = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                B, K = batch['movieId'].shape # B=Batch, K=5 (1+4)
                
                # Reshape: (B, 5, ...) -> (B*5, ...)
                inputs = {
                    'movieId': batch['movieId'].view(-1).to(device),
                    'seq_history': batch['seq_history'].view(B*K, -1).to(device),
                    'dense_feats': batch['dense_feats'].view(B*K, -1).to(device)
                }
                label = batch['label'].to(device) # 全 0
                
                optimizer.zero_grad()
                outputs = model(inputs) # (B*5,)
                
                # 把分值拉回 (B, 5)
                logits = outputs['click'].view(B, K)
                
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                
                # 统计 Top-1 命中率 (即预测对 0 的次数)
                preds = torch.argmax(logits, dim=1)
                correct_top1 += (preds == 0).sum().item()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item(), 'top1_acc': correct_top1 / ((pbar.n + 1) * B)})

        # 验证... (省略相似逻辑)
        logger.success(f"Epoch {epoch+1} 完成！")

if __name__ == "__main__":
    train_listwise()
