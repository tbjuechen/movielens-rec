import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

def merge_tmdb_data():
    tmdb_dir = Path(RAW_DATA_DIR) / "tmdb_cache"
    output_file = Path(PROCESSED_DATA_DIR) / "tmdb_features.parquet"
    
    if not tmdb_dir.exists():
        print(f"Error: TMDB cache directory not found at {tmdb_dir}")
        return

    print(f"Reading JSON files from {tmdb_dir}...")
    files = [f for f in os.listdir(tmdb_dir) if f.endswith(".json")]
    print(f"Found {len(files)} files.")

    data_list = []
    for f in tqdm(files, desc="Merging TMDB JSONs"):
        filepath = tmdb_dir / f
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                d = json.load(file)
                
            # Extract core features defined in design spec
            extracted = {
                "tmdb_id": d.get("id"),
                "imdb_id": d.get("imdb_id"),
                "original_language": d.get("original_language"),
                "budget": d.get("budget", 0),
                "revenue": d.get("revenue", 0),
                "runtime": d.get("runtime", 0),
                "vote_average": d.get("vote_average", 0.0),
                "vote_count": d.get("vote_count", 0),
                "tmdb_genres": [g["name"] for g in d.get("genres", [])]
            }
            data_list.append(extracted)
        except Exception as e:
            print(f"Warning: Failed to parse {f}: {e}")

    print("Converting to DataFrame...")
    df = pd.DataFrame(data_list)
    
    print(f"Saving to {output_file}...")
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"Success! Merged {len(df)} movies.")

if __name__ == "__main__":
    merge_tmdb_data()
