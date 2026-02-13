import pandas as pd
from src.models.recall.popularity import PopularityRecall
from pathlib import Path
from loguru import logger

def test_popularity():
    # 1. 加载电影元数据（用于显示标题）
    movies_path = Path("data/processed/movies.parquet")
    if not movies_path.exists():
        logger.error("Please run preprocessing first!")
        return
    df_movies = pd.read_parquet(movies_path)
    movie_dict = dict(zip(df_movies['movieId'], df_movies['title']))

    # 2. 加载模型
    model_path = "saved_models/popularity_recall.pkl"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Please run training script first!")
        return
    
    model = PopularityRecall()
    model.load(model_path)

    # 3. 执行召回
    # 热门召回对所有用户都一样，这里 user_id 随便填
    results = model.recall(user_id=1, top_n=10)

    # 4. 打印结果
    print("\n" + "="*50)
    print(f"【热门召回测试】Top 10 推荐结果")
    print("="*50)
    for i, (mid, score) in enumerate(results):
        title = movie_dict.get(mid, "Unknown Title")
        print(f"{i+1}. [ID: {mid:6}] Score: {score:8.1f} | Title: {title}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_popularity()
