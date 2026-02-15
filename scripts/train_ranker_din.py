import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from scripts.generate_feature_profiles import generate_profiles
from src.features.ranking_feature_engine import RankingFeatureEngine
from src.data_loader_ranking import RankingDataset
from src.models.ranking.deep_model import UnifiedDeepRanker

def train_din():
    # 0. 配置
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    
    # 1. 准备数据 (复用三段式逻辑)
    logger.info("加载并切分样本...")
    samples_df = pd.read_parquet(ranking_dir / "ranking_samples_prototype.parquet")
    samples_df = samples_df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(samples_df)
    history_end = int(n * 0.6)
    train_end = int(n * 0.8)
    
    history_df = samples_df.iloc[:history_end]
    train_samples = samples_df.iloc[history_end:train_end].copy()
    val_samples = samples_df.iloc[train_end:].copy()

    # 2. 特征工程 (Isolation)
    logger.info("生成特征矩阵...")
    history_ratings = history_df[history_df['label'] == 1][['userId', 'movieId', 'rating', 'timestamp']]
    # generate_profiles(ref_ratings=history_ratings) # 假设画像已生成，跳过以节省时间
    
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(history_ratings)
    
    # 生成宽表 (包含 semantic_sim, user_avg_rating 等 Dense 特征)
    logger.info("提取训练集特征 (此步较慢)...")
    train_df = engine.build_feature_matrix(train_samples)
    val_df = engine.build_feature_matrix(val_samples)
    
    # 3. 构造 PyTorch Dataset
    # 需要先加载 item_profile 以获取全量 movieId 映射
    item_profile = pd.read_parquet(ranking_dir / "item_profile_ranking.parquet")
    
    train_ds = RankingDataset(train_df, item_profile)
    val_ds = RankingDataset(val_df, item_profile)
    
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    
    # 4. 初始化模型
    feature_map = {
        'sparse': {'movieId': train_ds.vocab_size},
        'dense': 4 # [user_avg, vote_avg, sem_sim, genre_match]
    }
    
    model = UnifiedDeepRanker(feature_map, embedding_dim=32, tasks=['click']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # 5. 训练循环
    logger.info("开始深度训练...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                # Move to device
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
                label = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs['click'], label)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # 验证
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
                label = batch['label']
                
                logits = model(inputs)['click']
                probs = torch.sigmoid(logits)
                
                preds.extend(probs.cpu().numpy())
                labels.extend(label.numpy())
        
        val_auc = roc_auc_score(labels, preds)
        logger.success(f"Epoch {epoch+1} Val AUC: {val_auc:.4f}")

if __name__ == "__main__":
    train_din()
