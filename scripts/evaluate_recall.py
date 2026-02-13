import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from src.models.recall.popularity import PopularityRecall
from src.models.recall.itemcf import ItemCFRecall
from src.models.recall.two_tower import TwoTowerRecall
from src.evaluation import calculate_hit_rate

def evaluate_all():
    data_dir = Path("data/processed/two_tower")
    test_path = data_dir / "test.parquet"
    
    # 1. 加载测试集和映射表
    logger.info("Loading test data and mappings...")
    df_test = pd.read_parquet(test_path)
    with open(data_dir / "user_map.pkl", "rb") as f:
        user_map = pickle.load(f)
    with open(data_dir / "movie_map.pkl", "rb") as f:
        movie_map = pickle.load(f)
    
    inv_user_map = {v: k for k, v in user_map.items()}
    inv_movie_map = {v: k for k, v in movie_map.items()}

    # 构造 Ground Truth
    ground_truth_dict = df_test.groupby('userId')['movieId'].apply(set).to_dict()
    test_users = list(ground_truth_dict.keys())
    
    # 随机采样 1000 个测试用户进行评估
    np.random.seed(42)
    sample_users = np.random.choice(test_users, min(1000, len(test_users)), replace=False)
    gt_list = [ground_truth_dict[uid] for uid in sample_users]
    
    # 2. 评估各路模型
    top_k = 50
    models_to_eval = [
        ("Popularity", "saved_models/popularity_recall.pkl"),
        ("ItemCF", "saved_models/itemcf_recall.pkl"),
        ("Two-Tower", "saved_models/two_tower_model")
    ]

    results_summary = []

    for name, path in models_to_eval:
        if not Path(path).exists():
            continue
            
        logger.info(f"Setting up {name} Recall...")
        
        if name == "Popularity":
            model = PopularityRecall()
            model.load(path)
            raw_top_mids = [mid for mid, _ in model.recall(0, top_n=top_k)]
            internal_top_mids = [movie_map[mid] for mid in raw_top_mids if mid in movie_map]
            preds = [internal_top_mids for _ in sample_users]
            
        elif name == "ItemCF":
            model = ItemCFRecall()
            model.load(path)
            preds = []
            for uid_int in tqdm(sample_users, desc=f"Evaluating {name}"):
                uid_raw = inv_user_map[uid_int]
                raw_rec = model.recall(uid_raw, top_n=top_k)
                preds.append([movie_map[mid] for mid, _ in raw_rec if mid in movie_map])
                
        elif name == "Two-Tower":
            model = TwoTowerRecall()
            model.load(path)
            preds = []
            for uid_int in tqdm(sample_users, desc=f"Evaluating {name}"):
                rec_internal = model.recall(uid_int, top_n=top_k)
                preds.append([mid_int for mid_int, _ in rec_internal])

        hr = calculate_hit_rate(preds, gt_list)
        results_summary.append((name, hr))
        logger.success(f"{name} HitRate@{top_k}: {hr:.4f}")

    # 3. 输出最终汇总表格
    print("\n" + "="*40)
    print(f"{'Recall Model':<15} | {'HitRate@' + str(top_k):<10}")
    print("-" * 40)
    for name, score in results_summary:
        print(f"{name:<15} | {score:<10.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    evaluate_all()
