import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation.")
    args = parser.parse_args()

    print("Running end-to-end evaluation pipeline...")
    
    # Orchestrate retriever, feature_fetcher, and ranker here.
    # Compute final metrics with evaluator.
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
