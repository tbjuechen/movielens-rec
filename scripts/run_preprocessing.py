from src.data_loader import convert_csv_to_parquet
from pathlib import Path

def preprocess_all():
    raw_dir = Path("data/ml-32m")
    processed_dir = Path("data/processed")
    
    # Files to convert as defined in README.txt
    csv_files = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]
    
    for filename in csv_files:
        csv_path = raw_dir / filename
        parquet_path = processed_dir / f"{csv_path.stem}.parquet"
        
        if csv_path.exists():
            convert_csv_to_parquet(str(csv_path), str(parquet_path))
        else:
            print(f"Warning: {csv_path} not found.")

if __name__ == "__main__":
    preprocess_all()
