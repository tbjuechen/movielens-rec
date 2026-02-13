import pandas as pd
from pathlib import Path
from loguru import logger

def convert_csv_to_parquet(csv_path: str, parquet_path: str) -> None:
    """
    Converts a CSV file to Parquet format for efficient storage and retrieval.
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    logger.info(f"Converting {csv_path} to {parquet_path}...")
    # Use engine='pyarrow' for better performance with large files
    df = pd.read_csv(csv_path)
    
    # Ensure parent directory for parquet exists
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(parquet_path, index=False, engine='pyarrow')
    logger.info(f"Successfully converted to {parquet_path}")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from either CSV or Parquet format based on the file extension.
    """
    path = Path(file_path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix == '.csv':
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
