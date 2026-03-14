import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import PROCESSED_DATA_DIR, FEATURE_STORE_DIR
from src.features.encoder import FeatureEncoder

def main():
    parser = argparse.ArgumentParser(description="Step 2: Build and save feature encoders (vocabs, scalers).")
    args = parser.parse_args()

    print("=== Starting Step 2: Building Feature Library ===")
    
    # Load processed data
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    
    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    
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
    
    # 2. Fit Continuous Scalers
    print("Fitting scalers for numerical features...")
    encoder.fit_continuous(user_profile, ['avg_rating', 'activity'])
    encoder.fit_continuous(item_profile, ['release_year_orig', 'avg_rating', 'revenue', 'budget', 'vote_count_ml'])
    
    # 3. Save to feature_store
    encoder.save()
    print(f"=== Step 2 Complete: Encoders saved to {FEATURE_STORE_DIR} ===")

if __name__ == "__main__":
    main()
