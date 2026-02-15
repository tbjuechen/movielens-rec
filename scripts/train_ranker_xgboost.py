import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
from loguru import logger
from scripts.generate_feature_profiles import generate_profiles
from src.features.ranking_feature_engine import RankingFeatureEngine

def train_ranker_v4_robust():
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    model_dir = Path("saved_models/ranking_xgboost_v4")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载全量样本并按时间排序
    logger.info("加载样本并执行三段式时间切分...")
    samples_df = pd.read_parquet(ranking_dir / "ranking_samples_prototype.parquet")
    samples_df = samples_df.sort_values('timestamp').reset_index(drop=True)
    
    # 三段式切分比例：60% 历史, 20% 训练, 20% 验证
    n = len(samples_df)
    history_end = int(n * 0.6)
    train_end = int(n * 0.8)
    
    # 历史区：仅用于构建画像和索引
    history_df = samples_df.iloc[:history_end]
    # 训练区：用于训练模型
    train_samples = samples_df.iloc[history_end:train_end].copy()
    # 验证区：用于评估模型
    val_samples = samples_df.iloc[train_end:].copy()

    # 2. 【核心隔离】基于『历史区』生成画像和索引
    logger.info(f"正在基于历史区 (前 60%) 记录重制画像，确保训练集完全不可见...")
    history_ratings = history_df[history_df['label'] == 1][['userId', 'movieId', 'rating', 'timestamp']]
    generate_profiles(ref_ratings=history_ratings)

    # 3. 初始化引擎 (绑定历史数据)
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(history_ratings) 
    
    # 4. 提取特征
    logger.info("提取训练集和验证集特征 (基于历史偏好)...")
    X_train_full = engine.build_feature_matrix(train_samples)
    X_val_full = engine.build_feature_matrix(val_samples)

    # 5. 特征选择
    feature_cols = [
        'user_avg_rating', 'user_rating_std', 'user_rating_count_log',
        'year', 'runtime', 'budget_log', 'revenue_log', 'vote_average', 'vote_count',
        'is_director_match', 'actor_match_count', 'rating_diff', 'semantic_sim'
    ]

    X_train = X_train_full[feature_cols].fillna(0)
    y_train = X_train_full['label']
    X_val = X_val_full[feature_cols].fillna(0)
    y_val = X_val_full['label']

    # 6. 训练 XGBoost
    # 增加正则化，防止过拟合
    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'eta': 0.1,
        'eval_metric': ['auc', 'logloss'],
        'tree_method': 'hist',
        'lambda': 1,  # L2 正则
        'alpha': 0.5, # L1 正则
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    logger.info("开始训练健壮版模型...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=15,
        verbose_eval=10
    )

    # 7. 评估输出
    y_pred = model.predict(dval)
    auc = roc_auc_score(y_val, y_pred)
    logger.success(f"三段式隔离后的真实 AUC: {auc:.4f}")
    
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values(by='importance', ascending=False)
    
    print("\n" + "="*30)
    print("Robust Feature Importance (Gain):")
    print(importance_df.head(10))
    print("="*30 + "\n")

    model.save_model(model_dir / "ranker_robust.json")

if __name__ == "__main__":
    train_ranker_v4_robust()
