import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
from loguru import logger
import pickle
from scripts.generate_feature_profiles import generate_profiles
from src.features.ranking_feature_engine import RankingFeatureEngine

def train_ranker_v3():
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    model_dir = Path("saved_models/ranking_xgboost_v3")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载样本
    logger.info("加载原型样本数据...")
    samples_df = pd.read_parquet(ranking_dir / "ranking_samples_prototype.parquet")

    # 2. 严格时间切分
    logger.info("执行基于时间戳的训练验证集切分...")
    samples_df = samples_df.sort_values('timestamp')
    split_idx = int(len(samples_df) * 0.8)
    
    train_samples = samples_df.iloc[:split_idx].copy()
    test_samples = samples_df.iloc[split_idx:].copy()

    # 3. 【核心修正】基于训练集样本重新生成画像 (Feature Isolation)
    # 获取训练集涉及的打分记录作为参考
    logger.info("正在基于训练集数据重新生成画像底座，防止统计泄露...")
    # 只取 train_samples 中 label=1 (即真实打分记录) 的数据来计算画像
    train_ratings = train_samples[train_samples['label'] == 1][['userId', 'movieId', 'rating', 'timestamp']]
    generate_profiles(ref_ratings=train_ratings)

    # 4. 初始化引擎并提取特征
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize(train_ratings) 
    
    logger.info("开始提取训练集和验证集特征...")
    train_matrix = engine.build_feature_matrix(train_samples)
    test_matrix = engine.build_feature_matrix(test_samples)

    # 5. 训练
    feature_cols = [
        'user_avg_rating', 'user_rating_std', 'user_rating_count_log',
        'year', 'runtime', 'budget_log', 'revenue_log', 'vote_average', 'vote_count',
        'is_director_match', 'actor_match_count', 'rating_diff', 'semantic_sim'
    ]

    X_train = train_matrix[feature_cols].fillna(0)
    y_train = train_matrix['label']
    X_test = test_matrix[feature_cols].fillna(0)
    y_test = test_matrix['label']

    logger.info(f"开始训练隔离后的模型... 特征数: {len(feature_cols)}")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': ['auc', 'logloss'],
        'tree_method': 'hist',
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'val')],
        early_stopping_rounds=15,
        verbose_eval=10
    )

    # 6. 最终评估与重要性输出
    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    logger.success(f"最终隔离评估 AUC: {auc:.4f}")
    
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values(by='importance', ascending=False)
    
    print("\n" + "="*30)
    print("Final Feature Importance (Gain):")
    print(importance_df.head(10))
    print("="*30 + "\n")

    model.save_model(model_dir / "ranker_v3.json")
    logger.info(f"模型已保存至 {model_dir}")

if __name__ == "__main__":
    train_ranker_v3()
