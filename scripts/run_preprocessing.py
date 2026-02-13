from src.data_loader import convert_csv_to_parquet
from src.preprocessing import preprocess_movies
import pandas as pd
from pathlib import Path
from loguru import logger

def preprocess_all():
    raw_dir = Path("data/ml-32m")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process Movies with specific logic
    movies_csv = raw_dir / "movies.csv"
    if movies_csv.exists():
        logger.info("Processing and cleaning movies.csv...")
        df_movies = pd.read_csv(movies_csv)
        df_movies = preprocess_movies(df_movies)
        df_movies.to_parquet(processed_dir / "movies.parquet", index=False)
    
    # Other files to convert directly
    other_files = ["ratings.csv", "tags.csv", "links.csv"]
    
    for filename in other_files:
        csv_path = raw_dir / filename
        parquet_path = processed_dir / f"{csv_path.stem}.parquet"
        
        if csv_path.exists():
            convert_csv_to_parquet(str(csv_path), str(parquet_path))
        else:
            logger.warning(f"{csv_path} not found.")

if __name__ == "__main__":
    preprocess_all()
