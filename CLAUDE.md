# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Industrial-grade multi-channel recall recommendation system built on MovieLens-32M + TMDB data. Five recall channels (Dual-Tower, ItemCF, UserCF, Popularity, Genre) are merged via Reciprocal Rank Fusion. Communication language: Chinese preferred per GEMINI.md conventions.

## Running the Pipeline

All commands must use the `movielens-rec` Conda environment:

```bash
# Step 0: Merge 86K TMDB JSON files into Parquet
conda run -n movielens-rec python scripts/00_prepare_tmdb.py

# Step 1: Preprocess raw data, build wide tables, leave-one-out split
conda run -n movielens-rec python scripts/01_process_data.py

# Step 2: Fit and save feature encoders (vocabularies + scalers)
conda run -n movielens-rec python scripts/02_build_features.py

# Step 3: Train models (each independently)
conda run -n movielens-rec python scripts/03_train_models.py --model dual_tower
conda run -n movielens-rec python scripts/03_train_models.py --model item_cf
conda run -n movielens-rec python scripts/03_train_models.py --model user_cf

# Step 4: End-to-end evaluation (builds FAISS index, computes Recall/NDCG)
conda run -n movielens-rec python scripts/04_evaluate_e2e.py
conda run -n movielens-rec python scripts/04_evaluate_e2e.py --test  # quick mode: 1000 users
```

## Configuration

- `config.yaml` (gitignored) is loaded at runtime; falls back to `config.sample.yaml`
- All hyperparameters, paths, and merger weights are centralized in `src/config/settings.py` which reads from YAML
- To customize, copy `config.sample.yaml` to `config.yaml` and edit

## Architecture

### Data Flow

```
Raw CSVs + TMDB JSONs
  → scripts/00 (merge TMDB → Parquet)
  → scripts/01 (preprocessor.py: wide tables + chronological leave-one-out split)
  → scripts/02 (encoder.py: fit vocab & scalers → feature_store/)
  → scripts/03 (train models → model_weights/)
  → scripts/04 (evaluator: FAISS index → multi-channel recall → RRF merge → metrics)
```

### Key Modules

- **`src/config/settings.py`** — Single source of truth. Reads `config.yaml`, exports all constants. Every other module imports from here.
- **`src/data_pipeline/preprocessor.py`** — Builds `user_profile.parquet` and `item_profile.parquet` from training data only (anti-leakage). Handles log-transform for long-tail features.
- **`src/features/encoder.py`** — `FeatureEncoder` class. Fits categorical vocabularies (shared genre vocab across user/item) and MinMaxScaler for continuous features. Uses prefix isolation (`user_`, `item_`) to prevent key collisions.
- **`src/data_pipeline/dataset.py`** — Zero-copy PyTorch Dataset using `torch.from_numpy` with dense lookup arrays indexed by raw userId/movieId.
- **`src/models/recall/dual_tower.py`** — User/Item towers with shared item_emb and genre_emb. Learnable logit scale (1/tau). Mixed loss: InfoNCE + BPR with collision masking and Log-Q correction.
- **`src/models/recall/item_cf.py` / `user_cf.py`** — Parallel sparse matrix computation using multiprocessing with module-level `_shared` dict for fork-based COW.
- **`src/models/recall/merger.py`** — RRF fusion (damping k=60) with configurable per-channel weights.
- **`src/models/recall/simple_recall.py`** — Popularity and Genre-based cold-start recall.

### Data Directories (all gitignored)

- `data/raw/ml-32m/` — Original MovieLens CSVs
- `data/raw/tmdb_cache/` — Individual TMDB JSON files
- `data/processed/` — Parquet wide tables, train/val/test splits
- `data/feature_store/` — Encoder artifacts, genre-to-items index, popularity list
- `data/model_weights/` — Dual-tower `.pth`, CF similarity `.pkl`

## Critical Design Constraints (from GEMINI.md)

- **No data leakage**: User features must be computed from `train_data` only. Val/test are targets only.
- **Vectorized ops**: No slow-apply or row-level loops on 32M data. Use Pandas/NumPy vectorization.
- **Parquet only**: No uncompressed CSV for intermediate wide tables.
- **Dual-Tower invariants**: Shared embedding between user history and item ID. Log-Q correction order: `(Score / Tau) - LogQ`. Mixed loss must include InfoNCE + BPR. Must use time-decay weighting.
- **Continuous features**: Long-tail features (revenue, activity, vote_count) require log-transform. Continuous features use quantile-based bucketization into embeddings.
