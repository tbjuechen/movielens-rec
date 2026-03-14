import os
from pathlib import Path
import yaml

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 1. Load configuration from config.yaml if it exists
config_path = BASE_DIR / "config.yaml"
if not config_path.exists():
    # Use sample if real config is missing
    config_path = BASE_DIR / "config.sample.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 2. Path Settings
RAW_DATA_DIR = BASE_DIR / config['paths']['raw_data_dir']
PROCESSED_DATA_DIR = BASE_DIR / config['paths']['processed_data_dir']
FEATURE_STORE_DIR = BASE_DIR / config['paths']['feature_store_dir']
MODEL_WEIGHTS_DIR = BASE_DIR / config['paths']['model_weights_dir']

# 3. Model Hyperparameters
RECALL_K = config['recall']['top_k']
EMBEDDING_DIM = config['recall']['embedding_dim']
TAU = config['recall']['tau']

# 4. Training Settings
BATCH_SIZE = config['training']['batch_size']
LEARNING_RATE = config['training']['learning_rate']
EPOCHS = config['training']['epochs']

# Negative Sampling Sizes
INBATCH_NEG_SIZE = config['training']['inbatch_neg_size']
GLOBAL_NEG_SIZE = config['training']['global_neg_size']
HARD_NEG_SIZE = config['training']['hard_neg_size']
NUM_WORKERS = config['training'].get('num_workers', 4)

# 5. Feature Settings
USER_HISTORY_MAX_LEN = config['features']['user_history_max_len']
USER_TOP_GENRES_MAX_LEN = config['features']['user_top_genres_max_len']
ITEM_GENRES_MAX_LEN = config['features']['item_genres_max_len']
TIME_DECAY_LAMBDA = config['features']['time_decay_lambda']
