import pandas as pd
import re
from loguru import logger

def extract_year(title: str) -> int:
    """
    Extracts the release year from the movie title (e.g., 'Toy Story (1995)' -> 1995).
    Returns 0 if no year is found.
    """
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return 0

def preprocess_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for the movies dataframe:
    - Extracts year from title
    - Converts genres string to a list
    """
    logger.info("Preprocessing movie metadata...")
    
    # Extract year
    df['year'] = df['title'].apply(extract_year)
    
    # Convert genres to list
    df['genres'] = df['genres'].apply(lambda x: x.split('|'))
    
    return df
