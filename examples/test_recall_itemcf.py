import pandas as pd
from src.models.recall.itemcf import ItemCFRecall
from pathlib import Path
from loguru import logger

def test_itemcf():
    # 1. 加载数据用于显示
    processed_dir = Path("data/processed")
    movies_path = processed_dir / "movies.parquet"
    if not movies_path.exists():
        logger.error("Please run preprocessing first!")
        return
    df_movies = pd.read_parquet(movies_path)
    movie_dict = dict(zip(df_movies['movieId'], df_movies['title']))

    # 2. 加载训练好的模型
    model_path = "saved_models/itemcf_recall.pkl"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Please run training first!")
        return
    
    model = ItemCFRecall()
    model.load(model_path)

    # 3. 选择一个用户进行测试
    # 选一个有一定观影记录的用户
    available_users = [uid for uid, hist in model.user_history.items() if len(hist) > 5]
    if not available_users:
        test_user_id = list(model.user_history.keys())[0]
    else:
        test_user_id = available_users[0]
    
    # 4. 显示该用户的观影历史
    history_ids = model.user_history.get(test_user_id, [])
    print("\n" + "="*60)
    print(f"【用户历史】User ID: {test_user_id} | 观影数量: {len(history_ids)}")
    print("-" * 60)
    for mid in history_ids[:10]:
        print(f"- {movie_dict.get(mid, 'Unknown')}")
    if len(history_ids) > 10:
        print("  ...")

    # 5. 执行召回
    results = model.recall(user_id=test_user_id, top_n=10)

    # 6. 打印推荐结果
    print("\n" + "="*60)
    print(f"【ItemCF 个性化推荐结果】Top 10")
    print("=" * 60)
    for i, (mid, score) in enumerate(results):
        title = movie_dict.get(mid, 'Unknown')
        print(f"{i+1}. [ID: {mid:6}] Score: {score:8.3f} | Title: {title}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    test_itemcf()
