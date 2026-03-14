import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import json

from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_STORE_DIR

def process_all_data():
    raw_dir = RAW_DATA_DIR / "ml-32m"
    processed_dir = PROCESSED_DATA_DIR
    feature_store_dir = FEATURE_STORE_DIR
    tmdb_file = processed_dir / "tmdb_features.parquet"

    processed_dir.mkdir(parents=True, exist_ok=True)
    feature_store_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Stage 1/5] Loading Raw CSVs (32M rows)...")
    movies = pd.read_csv(raw_dir / "movies.csv")
    links = pd.read_csv(raw_dir / "links.csv")
    ratings = pd.read_csv(raw_dir / "ratings.csv", 
                         dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32, 'timestamp': np.int64})
    tmdb = pd.read_parquet(tmdb_file)

    # 1. Item Profile Base
    print("\n[Stage 2/5] Building Item Profile Base (Vectorized)...")
    movies['release_year_orig'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
    
    def bin_year_vec(years):
        bins = [-1, 1, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 9999]
        labels = ["Unknown", "<1950", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", ">2020"]
        return pd.cut(years, bins=bins, labels=labels, right=False)
    
    movies['release_year'] = bin_year_vec(movies['release_year_orig'])
    item_profile = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    item_profile = item_profile.merge(tmdb, left_on='tmdbId', right_on='tmdb_id', how='left')
    
    # Genre Alignment (Critical Fix #8)
    def align_genres(genre_list):
        if not isinstance(genre_list, (list, np.ndarray)): return []
        mapping = {"Science Fiction": "Sci-Fi", "Action & Adventure": "Action"}
        return [mapping.get(g, g) for g in genre_list]
    item_profile['tmdb_genres'] = item_profile['tmdb_genres'].apply(align_genres)

    # 2. Data Splitting (Chronological Anti-Leakage)
    print("\n[Stage 3/5] Filtering and Splitting Data (User Timeline)...")
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    ratings = ratings[ratings['userId'].isin(valid_users)].copy()
    ratings = ratings.sort_values(['userId', 'timestamp'], ascending=[True, True])
    
    # Leave-One-Out for Test and Val from Positive Interactions
    pos_ratings = ratings[ratings['rating'] >= 3.0].copy()
    print("Ranking positive interactions...")
    pos_ratings['rank'] = pos_ratings.groupby('userId')['timestamp'].rank(method='first', ascending=False)
    
    test_data = pos_ratings[pos_ratings['rank'] == 1].drop(columns=['rank'])
    val_data = pos_ratings[pos_ratings['rank'] == 2].drop(columns=['rank'])
    
    # Training Data: Remove specific val/test records AND any interactions after val_ts (Critical Fix #6)
    val_ts_map = val_data.set_index('userId')['timestamp'].to_dict()
    
    def filter_train(row):
        uid = row['userId']
        if uid in val_ts_map and row['timestamp'] >= val_ts_map[uid]:
            return False
        return True
    
    print("Applying chronological filter to train set (Anti-Leakage)...")
    # To speed up, we avoid per-row apply for 32M. We use vectorized comparison.
    ratings = ratings.merge(val_data[['userId', 'timestamp']].rename(columns={'timestamp': 'val_ts'}), on='userId', how='left')
    train_data = ratings[pd.isna(ratings['val_ts']) | (ratings['timestamp'] < ratings['val_ts'])].copy()
    train_data = train_data.drop(columns=['val_ts'])
    
    train_data.to_parquet(processed_dir / "train_data.parquet", index=False)
    val_data.to_parquet(processed_dir / "val_data.parquet", index=False)
    test_data.to_parquet(processed_dir / "test_data.parquet", index=False)
    print(f"Split complete. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 3. Item Statistics (Strictly from Train Only - Critical Fix #5)
    print("\n[Stage 4/5] Calculating Item Statistics (Train Only)...")
    item_stats = train_data.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    item_stats.columns = ['movieId', 'avg_rating', 'vote_count_ml']
    item_profile = item_profile.merge(item_stats, on='movieId', how='left')
    
    for col in ['revenue', 'budget', 'vote_count_ml', 'vote_count']:
        if col in item_profile.columns:
            item_profile[col] = np.log1p(item_profile[col].fillna(0))
    
    item_profile.to_parquet(processed_dir / "item_profile.parquet", index=False)
    print(f"Item Profile saved.")
    
    del pos_ratings, item_stats
    gc.collect()

    # 4. User Profile (From Train Data)
    print("\n[Stage 5/5] Building User Profile from Train Data...")
    user_stats = train_data.groupby('userId').agg(
        avg_rating=('rating', 'mean'),
        activity_orig=('movieId', 'count')
    ).reset_index()
    user_stats['activity'] = np.log1p(user_stats['activity_orig'])
    
    print("Aggregating User Top Genres (Vectorized)...")
    train_with_genres = train_data.merge(movies[['movieId', 'genres']], on='movieId')
    train_with_genres['genre_list'] = train_with_genres['genres'].str.split('|')
    exploded_genres = train_with_genres.explode('genre_list')

    genre_counts = exploded_genres.groupby(['userId', 'genre_list']).size().reset_index(name='cnt')
    genre_counts = genre_counts.sort_values(['userId', 'cnt'], ascending=[True, False])
    user_genres = genre_counts.groupby('userId').head(3).groupby('userId')['genre_list'].apply(list).reset_index()
    user_genres.columns = ['userId', 'top_genres']

    del train_with_genres, exploded_genres, genre_counts
    gc.collect()

    print("Building User History Sequence (Vectorized)...")
    # train_data is already sorted by userId, timestamp
    last_50 = train_data.groupby('userId').tail(50)
    max_ts = last_50.groupby('userId')['timestamp'].transform('max')
    last_50 = last_50.copy()
    last_50['ts_diff'] = (max_ts - last_50['timestamp']) / 3600.0

    user_history = last_50.groupby('userId').agg(
        history=('movieId', list),
        history_ts_diff=('ts_diff', list)
    ).reset_index()
    
    user_profile = user_stats.merge(user_genres, on='userId', how='left')
    user_profile = user_profile.merge(user_history, on='userId', how='left')
    user_profile.to_parquet(processed_dir / "user_profile.parquet", index=False)
    
    # 5. Inverted Index Artifacts
    print("Saving Hard Negative artifacts...")
    genre_to_items = item_profile.explode('tmdb_genres').dropna(subset=['tmdb_genres']).groupby('tmdb_genres')['movieId'].apply(list).to_dict()
    popularity_list = item_profile.sort_values('vote_count_ml', ascending=False)['movieId'].tolist()
    
    with open(feature_store_dir / "genre_to_items.json", "w") as f:
        json.dump(genre_to_items, f)
    with open(feature_store_dir / "popularity_list.json", "w") as f:
        json.dump(popularity_list, f)
    print("Done.")

if __name__ == "__main__":
    process_all_data()
