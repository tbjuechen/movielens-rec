# MovieLens Recommendation System (movielens-rec)

## Directory Overview
This project is dedicated to building and exploring movie recommendation systems using the MovieLens 32M dataset. It features a complete pipeline from raw data engineering to multi-modal deep learning recall and precision ranking.

The MovieLens 32M dataset contains over 32 million ratings across approximately 87,000 movies, created by about 200,000 users.

## Key Files and Data Structure
The core data is located in the `data/processed/` directory (Parquet format):
- **`movies.parquet`**: Metadata with year extraction and genre splitting.
- **`ratings.parquet`**: Interaction records.
- **`two_tower/`**: Specialized time-partitioned datasets and feature lookup tables for deep learning models.

## Implementation Roadmap

### Phase 1: Data Engineering & Feature Engineering
- [x] **Data Formatting**: Convert raw CSVs to Parquet for performance (Completed).
- [x] **EDA (Exploratory Data Analysis)**: Comprehensive insights recorded in `docs/EDA_REPORT.md` (Completed).
- [x] **Feature Processing**:
    - Movie: Year extraction, genre splitting, and normalization.
    - User: Multi-modal preferences and activity stats (Completed).

### Phase 2: Multi-channel Recall (召回)
- [x] **Base Infrastructure**: Abstract base class `BaseRecall` for consistent interfacing (Completed).
- [x] **Popularity-based**: Baseline with seen-item filtering (Completed).
- [x] **Collaborative Filtering**: Sparse-matrix optimized ItemCF (Completed).
- [x] **Two-Tower V2**: Deep multi-modal model with PyTorch & Faiss (Completed).
- [ ] **Embedding-based**: (Planned: Word2Vec/GraphSAGE).

### Phase 3: Ranking (精排)
- **Goal**: Predict specific scores or CTR for candidates.
- [ ] **Dataset Construction**: Negative sampling for ranking.
- [ ] **Model Implementation**: (Planned: XGBoost or DeepFM).

### Phase 4: Reranking & Inference
- [ ] **Diversity & MMR**: Diversity enhancement.
- [ ] **Pipeline**: End-to-end `InferencePipeline` in `src/inference/pipeline.py`.

## Technical Decisions
- **Logging**: Unified with `loguru` for high-performance terminal logging.
- **Data Split**: 80/10/10 global time-based split to ensure realistic evaluation without data leakage.
- **MPS Compatibility**: Replaced `nn.EmbeddingBag` with manual `mean` pooling to ensure high-performance training on Apple Silicon (M1/M2/M3/M4).
- **Architecture**: Used In-batch negative sampling with InfoNCE loss for the Two-Tower model to handle massive candidate spaces.

## Development Conventions
- **Atomic Commits**: Each commit must be atomic, implementing a single feature or fulfilling a single purpose.
- **Logging**: Use `loguru` for all project-wide logging.
- **Documentation**: Proactively update `README.md` and `GEMINI.md` with relevant CLI commands and progress whenever a core feature is implemented.

## Development Status
This project is currently in the **advanced development phase**. The recall layer is complete with multiple production-ready channels. The focus is shifting towards Phase 3 (Ranking).
