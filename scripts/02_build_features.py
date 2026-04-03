import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import PROCESSED_DATA_DIR, FEATURE_STORE_DIR
from src.config.alignment import get_alignment_paths
from src.features.encoder import FeatureEncoder

def main():
    parser = argparse.ArgumentParser(description="Step 2: Build and save feature encoders (vocabs, scalers).")
    parser.add_argument("--alignment", type=str, default=None, help="Alignment mode, e.g. strict_minonicc")
    args = parser.parse_args()
    paths = get_alignment_paths(args.alignment)
    processed_dir = paths.processed_dir
    feature_store_dir = paths.feature_store_dir

    print("=== Starting Step 2: Building Feature Library ===")
    
    # Load processed data
    user_profile = pd.read_parquet(Path(processed_dir) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(processed_dir) / "item_profile.parquet")
    
    encoder = FeatureEncoder(feature_store_dir)
    
    # 1. Fit Categorical Vocabularies
    print("Fitting vocabularies for IDs and Genres...")
    encoder.fit_categorical(user_profile['userId'], 'userId')
    encoder.fit_categorical(item_profile['movieId'], 'movieId')
    
    # Shared Genre Vocabulary
    all_genres = pd.concat([
        user_profile['top_genres'].explode(), 
        item_profile['tmdb_genres'].explode()
    ]).dropna()
    encoder.fit_categorical(all_genres, 'genres')
    
    # 2. Fit Continuous Scalers (Critical Fix #4: Prefix Isolation)
    print("Fitting scalers for numerical features...")
    encoder.fit_continuous(user_profile, ['avg_rating', 'activity'], prefix="user")
    encoder.fit_continuous(item_profile, ['release_year_orig', 'avg_rating', 'revenue', 'budget', 'vote_count_ml'], prefix="item")
    
    # 3. Save to feature_store
    encoder.save()
    print(f"=== Step 2 Complete: Encoders saved to {feature_store_dir} ===")

if __name__ == "__main__":
    main()
