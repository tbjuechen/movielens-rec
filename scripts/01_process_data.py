import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_pipeline.preprocessor import process_all_data

def main():
    parser = argparse.ArgumentParser(description="Step 1: Process raw MovieLens data and split datasets.")
    args = parser.parse_args()

    print("=== Starting Step 1: Data Preprocessing ===")
    # This calls the robust logic we built in src
    process_all_data()
    print("=== Step 1 Complete: Wide tables and splits generated in data/processed/ ===")

if __name__ == "__main__":
    main()
