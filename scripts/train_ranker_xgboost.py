import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
from pathlib import Path
from loguru import logger
import pickle
from src.features.ranking_feature_engine import RankingFeatureEngine

def train_ranker_v2():
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    model_dir = Path("saved_models/ranking_xgboost_v2")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载样本
    logger.info("加载原型样本数据...")
    samples_df = pd.read_parquet(ranking_dir / "ranking_samples_prototype.parquet")

    # 2. 严格的时间切分 (Time-based Split)
    # 取每个用户时间戳最晚的 20% 作为验证集
    logger.info("执行基于时间戳的训练验证集切分...")
    samples_df = samples_df.sort_values('timestamp')
    split_idx = int(len(samples_df) * 0.8)
    
    train_df = samples_df.iloc[:split_idx].copy()
    test_df = samples_df.iloc[split_idx:].copy()

    # 3. 初始化特征引擎 (仅基于训练集数据构建索引，防止泄露)
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    # ⚠️ 修正：传入训练集打分记录，让引擎只基于训练集构建“最爱主创”索引
    engine.initialize(train_df) 
    
    logger.info("提取训练集特征...")
    train_matrix = engine.build_feature_matrix(train_df)
    logger.info("提取测试集特征...")
    test_matrix = engine.build_feature_matrix(test_df)

    # 4. 特征选择
    feature_cols = [
        'user_avg_rating', 'user_rating_std', 'user_rating_count_log',
        'year', 'runtime', 'budget_log', 'revenue_log', 'vote_average', 'vote_count',
        'is_director_match', 'actor_match_count', 'rating_diff', 'semantic_sim'
    ]

    X_train = train_matrix[feature_cols].fillna(0)
    y_train = train_matrix['label']
    X_test = test_matrix[feature_cols].fillna(0)
    y_test = test_matrix['label']

    # 5. 训练 XGBoost
    logger.info(f"开始训练修正后的模型... 样本数: {len(X_train)}")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': ['auc', 'logloss'],
        'tree_method': 'hist'
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'val')],
        early_stopping_rounds=10,
        verbose_eval=20
    )

    # 6. 评估
    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    logger.success(f"修正后的 Offline AUC: {auc:.4f}")
    
    # 特征重要性分析
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values(by='importance', ascending=False)
    
    print("\n" + "="*30)
    print("Top Feature Importance (Gain):")
    print(importance_df.head(10))
    print("="*30 + "\n")

    if auc > 0.88:
        logger.warning("AUC 依然偏高，可能存在特征强关联或 Easy Negatives 问题。")

    # 保存
    model.save_model(model_dir / "ranker_v2.json")
    logger.info(f"模型已保存至 {model_dir}")

if __name__ == "__main__":
    train_ranker_v2()
