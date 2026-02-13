# MovieLens Recommendation System (movielens-rec)

## Directory Overview
This project is dedicated to building and exploring movie recommendation systems using the MovieLens 32M dataset. Currently, the repository serves as a data store for the raw MovieLens files, providing a foundation for future development of recommendation algorithms, data analysis, and model evaluation.

The MovieLens 32M dataset contains over 32 million ratings and 2 million tag applications across approximately 87,000 movies, created by about 200,000 users.

## Key Files and Data Structure
The core data is located in the `data/ml-32m/` directory:

- **`data/ml-32m/movies.csv`**: Contains movie metadata.
  - Columns: `movieId`, `title`, `genres` (pipe-separated).
- **`data/ml-32m/ratings.csv`**: The primary ratings dataset.
  - Columns: `userId`, `movieId`, `rating` (0.5 to 5.0), `timestamp`.
- **`data/ml-32m/tags.csv`**: User-generated tags for movies.
  - Columns: `userId`, `movieId`, `tag`, `timestamp`.
- **`data/ml-32m/links.csv`**: Mapping between MovieLens IDs and other movie databases.
  - Columns: `movieId`, `imdbId`, `tmdbId`.
- **`data/ml-32m/README.txt`**: The official documentation for the MovieLens 32M dataset, including license information and detailed field descriptions.
- **`data/ml-32m/checksums.txt`**: MD5 checksums for verifying file integrity.

## Usage
The contents of this directory are intended to be used as follows:
1. **Data Exploration**: Analyzing the distribution of ratings, popular genres, and tagging patterns.
2. **Preprocessing**: Cleaning and formatting the CSV files for use in machine learning models (e.g., creating user-item matrices).
3. **Model Training**: Implementing collaborative filtering, content-based filtering, or hybrid recommendation models.
4. **Evaluation**: Using the ratings data to test the accuracy and relevance of generated recommendations.

## Implementation Roadmap

### Phase 1: Data Engineering & Feature Engineering
- [x] **Data Formatting**: Convert raw CSVs to Parquet for performance (Completed).
- [ ] **EDA (Exploratory Data Analysis)**: Analyze user activity, movie popularity, and rating distributions.
- [ ] **Feature Processing**:
    - User: Historical interactions, average ratings, activity patterns.
    - Movie: Genres, release year, popularity metrics.
    - Context: Temporal features from timestamps.

### Phase 2: Multi-channel Recall (召回)
- **Goal**: Filter ~87,000 movies down to 500-1000 candidates.
- [ ] **Popularity-based**: Global and genre-specific hot movies.
- [ ] **Collaborative Filtering**: ItemCF, UserCF.
- [ ] **Embedding-based**: Vector search using Item2Vec or Two-Tower models.

### Phase 3: Ranking (精排)
- **Goal**: Predict specific scores or CTR for candidates.
- [ ] **Dataset Construction**: Generate positive/negative samples and train/test splits.
- [ ] **Model Implementation**: XGBoost/LightGBM or Deep Models (e.g., DeepFM, Wide&Deep).
- [ ] **Evaluation**: Metric calculation (NDCG, MRR, Precision@K).

### Phase 4: Reranking & Strategy (重排)
- **Goal**: Optimize final presentation.
- [ ] **Diversity & MMR**: Ensure variety in recommendations.
- [ ] **Business Logic**: Filtering watched movies, blacklists.

### Phase 5: Inference Pipeline
- [ ] **End-to-End Flow**: Integrate all modules into `src/inference/pipeline.py`.

## Development Conventions
- **Atomic Commits**: Each commit must be atomic, implementing a single feature or fulfilling a single purpose. Avoid bundling multiple unrelated changes into one commit.
- **Logging**: Use `loguru` for all project-wide logging instead of the standard library `logging` or `print` statements.

## Development Status
This project is currently in the **initial data setup phase**. No source code or scripts have been implemented yet. Future additions may include Python scripts (using libraries like `pandas`, `scikit-learn`, or `PyTorch`), notebooks, or specialized recommendation engine frameworks.
