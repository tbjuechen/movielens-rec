import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, classification_report
from pathlib import Path
from loguru import logger
import pickle
from src.features.ranking_feature_engine import RankingFeatureEngine

def train_ranker():
    data_dir = Path("data/processed")
    ranking_dir = data_dir / "ranking"
    model_dir = Path("saved_models/ranking_xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载样本数据
    logger.info("加载原型样本数据...")
    samples_path = ranking_dir / "ranking_samples_prototype.parquet"
    if not samples_path.exists():
        logger.error("样本集不存在，请先运行 scripts/prepare_ranking_dataset.py")
        return
    samples_df = pd.read_parquet(samples_path)

    # 2. 初始化特征引擎并生成特征矩阵
    engine = RankingFeatureEngine(data_dir=str(data_dir))
    engine.initialize()
    
    # 提取特征
    full_df = engine.build_feature_matrix(samples_df)

    # 3. 特征选择与预处理
    # 定义进入模型的特征列
    feature_cols = [
        'user_avg_rating', 'user_rating_std', 'user_rating_count_log',
        'year', 'runtime', 'budget_log', 'revenue_log', 'vote_average', 'vote_count',
        'is_director_match', 'actor_match_count', 'rating_diff', 'semantic_sim'
    ]
    
    X = full_df[feature_cols].fillna(0)
    y = full_df['label']

    logger.info(f"训练特征维度: {X.shape}")
    logger.info(f"正样本比例: {y.mean():.2%}")

    # 4. 划分数据集 (由于是排序，通常按时间切分，这里先用随机切分验证逻辑)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 训练 XGBoost 模型
    logger.info("开始训练 XGBoost 模型...")
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.1,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': ['auc', 'logloss'],
        'nthread': 8,
        'tree_method': 'hist' # 针对大规模数据加速
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    watchlist = [(dtrain, 'train'), (dtest, 'val')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=watchlist,
        early_stopping_rounds=20,
        verbose_eval=50
    )

    # 6. 评估与保存
    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred)
    
    logger.success(f"训练完成！")
    logger.info(f"Offline AUC: {auc:.4f}")
    logger.info(f"Offline LogLoss: {loss:.4f}")

    # 保存模型和特征元数据
    model.save_model(model_dir / "ranker.json")
    with open(model_dir / "features.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    logger.info(f"模型已保存至 {model_dir}")

if __name__ == "__main__":
    train_ranker()
