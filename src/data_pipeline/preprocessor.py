import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

# Enable progress_apply for pandas
tqdm.pandas()

def process_all_data():
    raw_dir = Path("data/raw/ml-32m")
    processed_dir = Path("data/processed")
    tmdb_file = processed_dir / "tmdb_features.parquet"

    print("Loading MovieLens core files (this may take ~40s)...")
    movies = pd.read_csv(raw_dir / "movies.csv")
    links = pd.read_csv(raw_dir / "links.csv")
    # Use float32 for rating to save memory
    ratings = pd.read_csv(raw_dir / "ratings.csv", dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32, 'timestamp': np.int64})
    
    print("Loading TMDB features...")
    tmdb = pd.read_parquet(tmdb_file)

    # 1. Item Profile Engineering
    print("Building Item Profile (Year extraction & Binning)...")
    # 向量化提取年份：从 "(1995)" 提取 1995
    movies['release_year_orig'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
    
    # Release Year Binning (Vectorized)
    def bin_year_vec(years):
        # 11 edges for 10 labels
        bins = [-1, 1, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 9999]
        labels = ["Unknown", "<1950", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", ">2020"]
        return pd.cut(years, bins=bins, labels=labels, right=False)
    
    movies['release_year'] = bin_year_vec(movies['release_year_orig'])
    
    # Merge with links to get tmdbId
    item_profile = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    
    # Merge with TMDB features
    item_profile = item_profile.merge(tmdb, left_on='tmdbId', right_on='tmdb_id', how='left')
    
    # Item Stats from ratings
    item_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    item_stats.columns = ['movieId', 'avg_rating', 'vote_count_ml']
    item_profile = item_profile.merge(item_stats, on='movieId', how='left')
    
    # Log transformation for long-tail features
    for col in ['revenue', 'budget', 'vote_count_ml', 'vote_count']:
        if col in item_profile.columns:
            item_profile[col] = np.log1p(item_profile[col].fillna(0))
    
    item_profile.to_parquet(processed_dir / "item_profile.parquet", index=False)
    print(f"Item Profile saved. Shape: {item_profile.shape}")

    # 2. Data Splitting (Leave-One-Out by User)
    print("Splitting datasets (Leave-One-Out)...")
    
    # Filter users with < 5 interactions
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    ratings = ratings[ratings['userId'].isin(valid_users)].copy()
    
    # Sort by timestamp
    ratings = ratings.sort_values(['userId', 'timestamp'], ascending=[True, True])
    
    # Positive interactions for target (rating >= 3.0)
    pos_ratings = ratings[ratings['rating'] >= 3.0].copy()
    
    # For each user, take the last as test, second last as val
    pos_ratings['rank'] = pos_ratings.groupby('userId')['timestamp'].rank(method='first', ascending=False)
    
    test_data = pos_ratings[pos_ratings['rank'] == 1].drop(columns=['rank'])
    val_data = pos_ratings[pos_ratings['rank'] == 2].drop(columns=['rank'])
    
    # Training data is everything else (including low ratings)
    # We remove the exact records used in val/test from the original ratings
    val_test_indices = pd.concat([test_data, val_data]).index
    train_data = ratings.drop(val_test_indices)
    
    train_data.to_parquet(processed_dir / "train_data.parquet", index=False)
    val_data.to_parquet(processed_dir / "val_data.parquet", index=False)
    test_data.to_parquet(processed_dir / "test_data.parquet", index=False)
    print(f"Data split complete. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 3. User Profile Engineering (Strictly using Train Data to avoid leakage)
    print("Building User Profile from Train Data...")
    
    # User history and stats from train_data
    user_stats = train_data.groupby('userId').agg(
        avg_rating=('rating', 'mean'),
        activity_orig=('movieId', 'count')
    ).reset_index()
    user_stats['activity'] = np.log1p(user_stats['activity_orig'])
    
    # User Top-3 Genres
    # Explode movies genres
    train_with_genres = train_data.merge(movies[['movieId', 'genres']], on='movieId')
    train_with_genres['genre_list'] = train_with_genres['genres'].str.split('|')
    exploded_genres = train_with_genres.explode('genre_list')
    
    def get_top_genres(series):
        return series.value_counts().head(3).index.tolist()
    
    user_genres = exploded_genres.groupby('userId')['genre_list'].apply(get_top_genres).reset_index()
    user_genres.columns = ['userId', 'top_genres']
    
    # User History Sequence (Latest 50 items)
    print("Building User History Sequence (Time-based)...")
    def get_history_with_time(group):
        group = group.sort_values('timestamp')
        latest_ts = group['timestamp'].max()
        # 将秒转换为小时
        ts_diffs = (latest_ts - group['timestamp']) / 3600.0
        return pd.Series({
            'history': group['movieId'].tolist()[-50:],
            'history_ts_diff': ts_diffs.tolist()[-50:]
        })

    user_history = train_data.groupby('userId').progress_apply(get_history_with_time).reset_index()
    
    # Merge all user features
    user_profile = user_stats.merge(user_genres, on='userId', how='left')
    user_profile = user_profile.merge(user_history, on='userId', how='left')
    
    user_profile.to_parquet(processed_dir / "user_profile.parquet", index=False)
    print(f"User Profile saved. Shape: {user_profile.shape}")

    # 4. Building Inverted Index (Genre -> Items) and Popularity List for Hard Negatives
    print("Building Inverted Index and Popularity List...")
    # Genre-to-Items Inverted Index
    item_genres = movies[['movieId', 'genres']].copy()
    item_genres['genre_list'] = item_genres['genres'].str.split('|')
    exploded_item_genres = item_genres.explode('genre_list')
    
    genre_to_items = exploded_item_genres.groupby('genre_list')['movieId'].apply(list).to_dict()
    
    # Global Popularity List (Sorted by vote_count_ml)
    # Using item_profile because it already has merged stats
    item_profile = pd.read_parquet(processed_dir / "item_profile.parquet")
    popularity_list = item_profile.sort_values('vote_count_ml', ascending=False)['movieId'].tolist()
    
    # Save these as artifacts in feature_store
    import json
    feature_store_dir = Path("data/feature_store")
    with open(feature_store_dir / "genre_to_items.json", "w") as f:
        json.dump(genre_to_items, f)
    
    with open(feature_store_dir / "popularity_list.json", "w") as f:
        json.dump(popularity_list, f)
        
    print("Hard Negative Sampling artifacts saved to feature_store.")

if __name__ == "__main__":
    process_all_data()
