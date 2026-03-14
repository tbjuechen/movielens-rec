import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

def process_all_data():
    raw_dir = Path("data/raw/ml-32m")
    processed_dir = Path("data/processed")
    tmdb_file = processed_dir / "tmdb_features.parquet"

    print("Loading MovieLens core files...")
    movies = pd.read_csv(raw_dir / "movies.csv")
    links = pd.read_csv(raw_dir / "links.csv")
    # Use float32 for rating to save memory
    ratings = pd.read_csv(raw_dir / "ratings.csv", dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32, 'timestamp': np.int64})
    
    print("Loading TMDB features...")
    tmdb = pd.read_parquet(tmdb_file)

    # 1. Item Profile Engineering
    print("Building Item Profile...")
    # Extract year from title: "Toy Story (1995)" -> 1995
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    
    movies['release_year'] = movies['title'].apply(extract_year)
    
    # Merge with links to get tmdbId
    item_profile = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    
    # Merge with TMDB features
    item_profile = item_profile.merge(tmdb, left_on='tmdbId', right_on='tmdb_id', how='left')
    
    # Calculate item average rating from ALL ratings (global item popularity/quality)
    item_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    item_stats.columns = ['movieId', 'avg_rating', 'vote_count_ml']
    item_profile = item_profile.merge(item_stats, on='movieId', how='left')
    
    # Finalize item_profile
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
        activity=('movieId', 'count')
    ).reset_index()
    
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
    user_history = train_data.groupby('userId')['movieId'].apply(lambda x: x.tolist()[-50:]).reset_index()
    user_history.columns = ['userId', 'history']
    
    # Merge all user features
    user_profile = user_stats.merge(user_genres, on='userId', how='left')
    user_profile = user_profile.merge(user_history, on='userId', how='left')
    
    user_profile.to_parquet(processed_dir / "user_profile.parquet", index=False)
    print(f"User Profile saved. Shape: {user_profile.shape}")

if __name__ == "__main__":
    process_all_data()
