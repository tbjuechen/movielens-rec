import os
from pathlib import Path
import yaml

def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

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
BPR_GAMMA = config['recall'].get('bpr_gamma', 5.0)
BPR_MARGIN = config['recall'].get('bpr_margin', 0.1)
LOGIT_SCALE_MAX = config['recall'].get('logit_scale_max', 100.0)
CONT_BUCKET_SIZE = config['recall'].get('cont_bucket_size', 20)
LOSS_INFONCE_WEIGHT = config['recall'].get('loss_infonce_weight', 1.0)
LOSS_BPR_WEIGHT = config['recall'].get('loss_bpr_weight', 1.0)

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

# 6. CF Settings
CF_TOP_K = config.get('cf', {}).get('top_k', 50)
CF_WORKERS = config.get('cf', {}).get('cf_workers', 32)

# 7. Ranking Model Settings
_ranking = config.get('ranking', {})
RANK_ID_EMBED_DIM = _ranking.get('id_embed_dim', 64)
RANK_GENRE_EMBED_DIM = _ranking.get('genre_embed_dim', 8)
RANK_CONT_EMBED_DIM = _ranking.get('cont_embed_dim', 8)
RANK_CONT_BUCKET_SIZE = _ranking.get('cont_bucket_size', 20)
RANK_HIST_SEQ_MAXLEN = _ranking.get('hist_seq_maxlen', 50)
RANK_DIN_ATTENTION_HIDDEN = _ranking.get('din_attention_hidden', [128, 64])
RANK_CROSS_LAYERS = _ranking.get('cross_layers', 3)
RANK_DROPOUT = _ranking.get('dropout', 0.1)
RANK_NUM_EXPERTS = _ranking.get('num_experts', 4)
RANK_EXPERT_DIM = _ranking.get('expert_dim', 128)
RANK_TOWER_DIMS = _ranking.get('tower_dims', [64, 32])
RANK_BATCH_SIZE = _ranking.get('batch_size', 4096)
RANK_LEARNING_RATE = _as_float(_ranking.get('learning_rate', 0.0005), 0.0005)
RANK_WEIGHT_DECAY = _as_float(_ranking.get('weight_decay', 1e-5), 1e-5)
RANK_WARMUP_EPOCHS = _ranking.get('warmup_epochs', 1)
RANK_EPOCHS = _ranking.get('epochs', 10)
RANK_NEG_SAMPLE_RATIO = _ranking.get('neg_sample_ratio', 3)
RANK_NUM_WORKERS = _ranking.get('num_workers', 8)
RANK_GRADNORM_ALPHA = _ranking.get('gradnorm_alpha', 1.5)
RANK_CTR_ALPHA = _ranking.get('ctr_alpha', 1.0)
RANK_RATING_BETA = _ranking.get('rating_beta', 0.5)
RANK_EVAL_KS = _ranking.get('eval_ks', [10, 20, 50])
RANK_OBJECTIVE = _ranking.get('objective', 'ctr')
RANK_NEGATIVES_PER_POSITIVE = _ranking.get('negatives_per_positive', _ranking.get('neg_sample_ratio', 4))
RANK_HARD_NEGATIVE_TOPK = _ranking.get('hard_negative_topk', 50)
RANK_EXPLICIT_NEGATIVE_THRESHOLD = _ranking.get('explicit_negative_threshold', 3.0)
_hard_negative_mix = _ranking.get('hard_negative_mix', [3, 1])
if not isinstance(_hard_negative_mix, list) or len(_hard_negative_mix) != 2:
    _hard_negative_mix = [3, 1]
RANK_HARD_NEGATIVE_MIX = _hard_negative_mix
RANK_TRAIN_POOL_SIZE = _ranking.get('train_pool_size', 100)
RANK_EVAL_POOL_SIZE = _ranking.get('eval_pool_size', 500)
RANK_FORCE_INSERT_TARGET = _ranking.get('force_insert_target', True)
RANK_BPR_WEIGHT = _as_float(_ranking.get('bpr_weight', 0.1), 0.1)
RANK_EARLY_STOP_METRIC = _ranking.get('early_stop_metric', 'mrr').lower()

# 8. Merger Weights
MERGER_WEIGHTS = config.get('merger_weights', {
    'dual_tower': 1.0,
    'item_cf': 1.0,
    'user_cf': 1.0,
    'popularity': 1.0,
    'genre': 1.0
})

# 9. Alignment Settings
_alignment = config.get('alignment', {})
ALIGNMENT_MODE = _alignment.get('mode', '')
ALIGNMENT_SEED = int(_alignment.get('seed', 42))
ALIGNMENT_STRICT_USER_PACK_SIZE = int(_alignment.get('strict_user_pack_size', 40000))
ALIGNMENT_STRICT_LONG_USER_RATIO = _as_float(_alignment.get('strict_long_user_ratio', 0.25), 0.25)
ALIGNMENT_STRICT_LONG_USER_THRESHOLD = int(_alignment.get('strict_long_user_threshold', 150))
ALIGNMENT_STRICT_POOL_SIZE = int(_alignment.get('strict_pool_size', 350))
ALIGNMENT_STRICT_VAL_SAMPLE_SIZE = int(_alignment.get('strict_val_sample_size', 10000))
ALIGNMENT_STRICT_TEST_SAMPLE_FRAC = _as_float(_alignment.get('strict_test_sample_frac', 0.2), 0.2)
