import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_pipeline.preprocessor import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description="Process raw MovieLens data.")
    args = parser.parse_args()

    print("Starting data processing...")
    preprocessor = DataPreprocessor(
        raw_data_dir=str(RAW_DATA_DIR),
        processed_data_dir=str(PROCESSED_DATA_DIR)
    )
    
    preprocessor.clean_data()
    preprocessor.split_dataset()
    print("Data processing complete.")

if __name__ == "__main__":
    main()
