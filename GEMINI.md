# MovieLens Recommendation System (movielens-rec)

## Directory Overview
This project is dedicated to building and exploring movie recommendation systems using the MovieLens 32M dataset. It features a complete pipeline from raw data engineering to multi-modal deep learning recall and precision ranking.

The MovieLens 32M dataset contains over 32 million ratings across approximately 87,000 movies, created by about 200,000 users.

## Key Files and Data Structure
- **`data/processed/`**: Cleaned Parquet files (movies, ratings, links).
- **`data/processed/two_tower/`**: Specialized training sets and feature lookup tables.
- **`data/processed/tmdb/`**: Normalized external metadata (Star Schema).

## Implementation Roadmap

### Phase 1: Data Engineering & Feature Engineering
- [x] **Data Formatting**: CSV -> Parquet conversion (Completed).
- [x] **EDA**: Comprehensive insights recorded in `docs/EDA_REPORT.md` (Completed).
- [x] **Core Feature Processing**: Year extraction, genre splitting (Completed).
- [x] **External Data Collection**: High-concurrency TMDb metadata spider (Completed).

### Phase 2: Multi-channel Recall (召回)
- [x] **Base Infrastructure**: Unified `BaseRecall` interface (Completed).
- [x] **Popularity**: Baseline with seen-item filtering (Completed).
- [x] **ItemCF**: Sparse-matrix optimized personalized recall (Completed).
- [x] **Two-Tower V2**: Deep multi-modal model with PyTorch & Faiss (Completed).

### Phase 3: Ranking (精排)
- **Goal**: Predict specific scores or CTR using rich features.
- [/] **Dataset Construction**: Integration of TMDb metadata (In Progress).
- [ ] **Model Implementation**: XGBoost or DeepFM.

## Technical Decisions
- **Logging**: Unified with `loguru` for high-performance terminal logging.
- **Data Modeling**: Used **Star Schema** for external data (Movies, Persons, Cast, Crew) to minimize redundancy and enable rich feature engineering.
- **Concurrency**: Optimized batch crawler with **50 threads** for M4 hardware.
- **MPS Compatibility**: Replaced `nn.EmbeddingBag` with manual pooling for Apple Silicon compatibility.

## Development Conventions
- **Atomic Commits**: Each commit must implement a single feature or purpose.
- **Documentation**: Proactively update README and Technical Reports.
