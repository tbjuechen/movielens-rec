import pandas as pd
import pickle
from src.models.recall.two_tower import TwoTowerRecall
from pathlib import Path
from loguru import logger

def test_two_tower():
    # 1. 加载映射表和元数据
    data_dir = Path("data/processed/two_tower")
    with open(data_dir / "user_map.pkl", "rb") as f:
        user_map = pickle.load(f)
    with open(data_dir / "movie_map.pkl", "rb") as f:
        movie_map = pickle.load(f)
    
    # 建立反向映射用于显示
    inv_movie_map = {v: k for k, v in movie_map.items()}
    
    df_movies = pd.read_parquet("data/processed/movies.parquet")
    movie_dict = dict(zip(df_movies['movieId'], df_movies['title']))

    # 2. 加载模型
    model_path = "saved_models/two_tower_v2"
    if not Path(model_path).exists():
        logger.error(f"Model directory not found at {model_path}!")
        return
    
    model = TwoTowerRecall()
    model.load(model_path)

    # 3. 选择测试用户
    # 我们选一个在训练集中比较活跃的用户
    df_train = pd.read_parquet(data_dir / "train.parquet")
    test_user_id_internal = df_train['userId'].value_counts().index[0]
    
    # 找到其对应的原始 UserID
    inv_user_map = {v: k for k, v in user_map.items()}
    original_user_id = inv_user_map[test_user_id_internal]

    # 4. 显示用户历史
    history_mids_internal = df_train[df_train['userId'] == test_user_id_internal]['movieId'].tolist()
    print("\n" + "="*60)
    print(f"【用户历史】Original User ID: {original_user_id} | 观影数量: {len(history_mids_internal)}")
    print("-" * 60)
    for mid_int in history_mids_internal[:10]:
        original_mid = inv_movie_map[mid_int]
        print(f"- {movie_dict.get(original_mid, 'Unknown')}")
    if len(history_mids_internal) > 10:
        print("  ...")

    # 5. 执行召回
    results_internal = model.recall(user_id=test_user_id_internal, top_n=10)

    # 6. 打印结果
    print("\n" + "="*60)
    print(f"【双塔模型召回结果】Top 10")
    print("=" * 60)
    for i, (mid_int, score) in enumerate(results_internal):
        original_mid = inv_movie_map.get(mid_int, -1)
        title = movie_dict.get(original_mid, 'Unknown')
        print(f"{i+1}. [ID: {original_mid:6}] Score: {score:8.3f} | Title: {title}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_two_tower()
