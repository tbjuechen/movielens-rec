import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import PROCESSED_DATA_DIR, FEATURE_STORE_DIR
from src.features.feature_builder import FeatureBuilder

def main():
    parser = argparse.ArgumentParser(description="Build and store static features.")
    args = parser.parse_args()

    print("Building features...")
    builder = FeatureBuilder(feature_store_dir=str(FEATURE_STORE_DIR))
    
    # Example logic:
    # ratings_df, movies_df = load_data(...)
    # builder.extract_user_features(ratings_df)
    # builder.extract_item_features(movies_df)
    builder.save_features()
    
    print("Feature building complete.")

if __name__ == "__main__":
    main()
