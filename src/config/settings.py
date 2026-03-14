import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
MODEL_WEIGHTS_DIR = DATA_DIR / "model_weights"

# Model Hyperparameters
RECALL_K = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
EPOCHS = 10

# Dual-Tower Specific Settings
TAU = 0.1  # Temperature for InfoNCE loss
