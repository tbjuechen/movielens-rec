"""Train the ranking model as a single-objective CTR ranker."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import (
    PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR,
    RANK_HIST_SEQ_MAXLEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN,
    RANK_ID_EMBED_DIM, RANK_GENRE_EMBED_DIM, RANK_CONT_EMBED_DIM,
    RANK_CONT_BUCKET_SIZE,
    RANK_CROSS_LAYERS, RANK_DROPOUT,
    RANK_NUM_EXPERTS, RANK_EXPERT_DIM, RANK_TOWER_DIMS,
    RANK_BATCH_SIZE, RANK_LEARNING_RATE, RANK_EPOCHS,
    RANK_NUM_WORKERS,
)
from src.features.encoder import FeatureEncoder
from src.models.ranking.ranker import RankingModel, _quantile_bounds
from src.data_pipeline.ranking_dataset import RankingDataset

VAL_SUBSET_RATIO = 0.01
VAL_SUBSET_SEED = 42
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
VAL_BATCH_SIZE = 8192


def apply_encoding(user_profile, item_profile, encoder):
    user_profile['userId_encoded'] = encoder.transform_categorical(user_profile['userId'], 'userId')
    user_profile['top_genres_encoded'] = encoder.transform_categorical(user_profile['top_genres'], 'genres', is_list=True, max_len=USER_TOP_GENRES_MAX_LEN)

    item_profile['movieId_encoded'] = encoder.transform_categorical(item_profile['movieId'], 'movieId')
    item_profile['tmdb_genres_encoded'] = encoder.transform_categorical(item_profile['tmdb_genres'], 'genres', is_list=True, max_len=ITEM_GENRES_MAX_LEN)

    user_cont = encoder.transform_continuous(user_profile, ['avg_rating', 'activity'], prefix="user")
    user_profile['avg_rating_norm'] = user_cont['user_avg_rating']
    user_profile['activity_norm'] = user_cont['user_activity']

    item_cont = encoder.transform_continuous(item_profile, ['release_year_orig', 'avg_rating', 'revenue', 'budget', 'vote_count_ml'], prefix="item")
    item_profile['release_year_norm'] = item_cont['item_release_year_orig']
    item_profile['avg_rating_norm'] = item_cont['item_avg_rating']
    item_profile['revenue_norm'] = item_cont['item_revenue']
    item_profile['budget_norm'] = item_cont['item_budget']
    item_profile['vote_count_ml_norm'] = item_cont['item_vote_count_ml']
    return user_profile, item_profile


def _unique_int_list(values):
    seen = set()
    out = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _build_feature_lookup_tables(user_profile, item_profile):
    max_uid = int(user_profile['userId'].max())
    max_iid = int(item_profile['movieId'].max())

    user_encoded_id = np.zeros(max_uid + 1, dtype=np.int64)
    user_top_genres_arr = np.zeros((max_uid + 1, USER_TOP_GENRES_MAX_LEN), dtype=np.int64)
    user_cont_arr = np.zeros((max_uid + 1, 2), dtype=np.float32)

    u_idx = user_profile['userId'].values
    user_encoded_id[u_idx] = user_profile['userId_encoded'].values
    user_top_genres_arr[u_idx] = np.stack(user_profile['top_genres_encoded'].values)
    user_cont_arr[u_idx] = np.stack([
        user_profile['avg_rating_norm'].values,
        user_profile['activity_norm'].values
    ], axis=1)

    item_encoded_id = np.zeros(max_iid + 1, dtype=np.int64)
    item_genres_arr = np.zeros((max_iid + 1, ITEM_GENRES_MAX_LEN), dtype=np.int64)
    item_cont_arr = np.zeros((max_iid + 1, 5), dtype=np.float32)

    i_idx = item_profile['movieId'].values
    item_encoded_id[i_idx] = item_profile['movieId_encoded'].values
    item_genres_arr[i_idx] = np.stack(item_profile['tmdb_genres_encoded'].values)
    item_cont_arr[i_idx] = np.stack([
        item_profile['release_year_norm'].values,
        item_profile['avg_rating_norm'].values,
        item_profile['revenue_norm'].values,
        item_profile['budget_norm'].values,
        item_profile['vote_count_ml_norm'].values
    ], axis=1)

    return {
        'user_encoded_id': user_encoded_id,
        'user_top_genres': user_top_genres_arr,
        'user_cont': user_cont_arr,
        'item_encoded_id': item_encoded_id,
        'item_genres': item_genres_arr,
        'item_cont': item_cont_arr,
    }


def _load_validation_subset(processed_dir):
    ranking_val_path = Path(processed_dir) / "ranking_val_candidates.parquet"
    ranking_val_hist_path = Path(processed_dir) / "ranking_val_hist_seq.npy"
    if not ranking_val_path.exists():
        print(
            f"  {ranking_val_path} not found. Please run 05_build_ranking_data.py first "
            "to generate validation candidate pools."
        )
        sys.exit(1)
    if not ranking_val_hist_path.exists():
        print(f"  {ranking_val_hist_path} not found. Please run 05_build_ranking_data.py first.")
        sys.exit(1)

    val_df = pd.read_parquet(ranking_val_path, columns=['userId', 'actual'])
    val_hist_seq = np.load(ranking_val_hist_path)
    if len(val_hist_seq) != len(val_df):
        raise ValueError(
            f"Validation hist_seq rows ({len(val_hist_seq)}) do not match val parquet rows ({len(val_df)})."
        )
    val_df = val_df.reset_index().rename(columns={'index': 'row_idx'})
    n_total = len(val_df)
    subset_size = max(1, int(np.ceil(n_total * VAL_SUBSET_RATIO)))
    subset_df = val_df.sample(n=subset_size, random_state=VAL_SUBSET_SEED).reset_index(drop=True)
    print(
        f"  Loaded validation pool with {n_total:,} rows; using fixed subset of "
        f"{subset_size:,} rows ({VAL_SUBSET_RATIO:.2%}, seed={VAL_SUBSET_SEED})"
    )
    return subset_df, val_hist_seq


def _evaluate_validation_loss(model, val_subset_df, val_hist_seq, lookup_tables, device, movie_vocab):
    model.eval()
    losses = []

    user_encoded_id = lookup_tables['user_encoded_id']
    user_top_genres = lookup_tables['user_top_genres']
    user_cont = lookup_tables['user_cont']
    item_encoded_id = lookup_tables['item_encoded_id']
    item_genres = lookup_tables['item_genres']
    item_cont = lookup_tables['item_cont']

    uid_buffer = []
    iid_buffer = []
    seq_buffer = []

    def _encode_hist(hist_row):
        return [movie_vocab.get(int(mid), 0) if int(mid) > 0 else 0 for mid in hist_row]

    def _flush_batch():
        if not uid_buffer:
            return

        uids = torch.tensor(uid_buffer, dtype=torch.long, device=device)
        iids = torch.tensor(iid_buffer, dtype=torch.long, device=device)
        seq_features = torch.tensor(seq_buffer, dtype=torch.long, device=device)
        int_features = torch.cat([
            user_encoded_id[uids].unsqueeze(1),
            item_encoded_id[iids].unsqueeze(1),
            user_top_genres[uids],
            item_genres[iids]
        ], dim=1).contiguous()
        float_features = torch.cat([
            user_cont[uids],
            item_cont[iids]
        ], dim=1).contiguous()
        features = {
            'int_features': int_features,
            'float_features': float_features,
            'seq_features': seq_features,
        }
        labels = torch.ones(len(uid_buffer), device=device)

        with torch.no_grad():
            logits = model(features).view(-1)
            losses.append(model.compute_loss(logits, labels).item())

        uid_buffer.clear()
        iid_buffer.clear()
        seq_buffer.clear()

    for row in val_subset_df.itertuples(index=False):
        actual_items = [int(iid) for iid in row.actual if int(iid) > 0]
        if not actual_items:
            continue
        hist_encoded = _encode_hist(val_hist_seq[int(row.row_idx)])

        for item_id in actual_items:
            uid_buffer.append(int(row.userId))
            iid_buffer.append(item_id)
            seq_buffer.append(hist_encoded)
            if len(uid_buffer) >= VAL_BATCH_SIZE:
                _flush_batch()

    _flush_batch()
    model.train()

    if not losses:
        raise ValueError("Validation subset produced no valid positive targets for early stopping.")
    return float(np.mean(losses))


def main():
    print("=== Training Ranking Model (DCNv2 + MMoE) ===")

    # 1. Load profiles first (train_data is only needed if samples don't exist)
    print("[1/6] Loading profiles...")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    
    print("[2/6] Encoding features...")
    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)

    # 3. Load pre-generated ranking samples
    print("[3/6] Loading ranking candidate pool...")
    ranking_samples_path = Path(PROCESSED_DATA_DIR) / "ranking_candidate_pool.parquet"
    ranking_train_hist_path = Path(PROCESSED_DATA_DIR) / "ranking_train_hist_seq.npy"
    if not ranking_samples_path.exists():
        print(f"  {ranking_samples_path} not found. Please run 05_build_ranking_data.py first.")
        sys.exit(1)
    if not ranking_train_hist_path.exists():
        print(f"  {ranking_train_hist_path} not found. Please run 05_build_ranking_data.py first.")
        sys.exit(1)
    else:
        # Optimization: Use pyarrow to load the pool
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(ranking_samples_path)
        available_cols = set(parquet_file.schema.names)
        has_explicit_negatives = 'explicit_negatives' in available_cols
        cols = ['userId', 'movieId', 'ctr_label', 'rating_norm', 'candidate_pool']
        if has_explicit_negatives:
            cols.append('explicit_negatives')
        table = pq.read_table(ranking_samples_path, columns=cols)
        
        # Convert list column to fixed-size matrix based on the saved pool length
        print("  Converting pool to matrix...")
        raw_pool = table['candidate_pool'].to_numpy()
        n_rows = len(raw_pool)
        pool_width = max((len(p) for p in raw_pool), default=0)
        pool_matrix = np.zeros((n_rows, pool_width), dtype=np.int64)
        for i, p in enumerate(tqdm(raw_pool, desc="Padding Pool")):
            p_len = min(len(p), pool_width)
            pool_matrix[i, :p_len] = p[:p_len]

        if has_explicit_negatives:
            raw_explicit = table['explicit_negatives'].to_numpy()
            explicit_width = max((len(p) for p in raw_explicit), default=0)
            explicit_matrix = np.zeros((n_rows, explicit_width), dtype=np.int64)
            for i, p in enumerate(tqdm(raw_explicit, desc="Padding Explicit Negatives")):
                p_len = min(len(p), explicit_width)
                explicit_matrix[i, :p_len] = p[:p_len]
            explicit_source = "ranking_candidate_pool.parquet"
        else:
            print(
                "  explicit_negatives not found in ranking_candidate_pool.parquet. "
                "Falling back to no explicit hard negatives for this run. "
                "Rebuild ranking data with 05_build_ranking_data.py to restore them."
            )
            explicit_width = 0
            explicit_matrix = np.zeros((n_rows, 0), dtype=np.int64)
            explicit_source = "disabled fallback"
        
        samples = {
            'userId': torch.from_numpy(table['userId'].to_numpy().astype(np.int64)),
            'movieId': torch.from_numpy(table['movieId'].to_numpy().astype(np.int64)),
            'ctr_label': torch.from_numpy(table['ctr_label'].to_numpy().astype(np.float32)),
            'rating_norm': torch.from_numpy(table['rating_norm'].to_numpy().astype(np.float32)),
            'candidate_pool': torch.from_numpy(pool_matrix),
            'explicit_negatives': torch.from_numpy(explicit_matrix),
        }
        print(
            f"  Loaded {n_rows:,} interactions with pool width {pool_width} "
            f"and explicit-negative width {explicit_width} ({explicit_source})"
        )
        del table, raw_pool, pool_matrix, explicit_matrix; import gc; gc.collect()
    n_total = len(samples['userId'])
    train_hist_seq = np.load(ranking_train_hist_path)
    if train_hist_seq.shape != (n_total, RANK_HIST_SEQ_MAXLEN):
        raise ValueError(
            f"Train hist_seq shape mismatch: expected {(n_total, RANK_HIST_SEQ_MAXLEN)}, got {train_hist_seq.shape}"
        )

    print(f"  Unique positives: {n_total:,} (Negative sampling will be 1:3 dynamic)")

    print("[3.5/6] Loading validation subset for early stopping...")
    val_subset_df, val_hist_seq = _load_validation_subset(PROCESSED_DATA_DIR)

    # 4. Compute quantile bucket boundaries
    print("[4/6] Computing bucket boundaries...")
    bucket_boundaries = {
        'user_avg_rating': _quantile_bounds(user_profile['avg_rating_norm'].values, RANK_CONT_BUCKET_SIZE),
        'user_activity': _quantile_bounds(user_profile['activity_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_release_year': _quantile_bounds(item_profile['release_year_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_avg_rating': _quantile_bounds(item_profile['avg_rating_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_revenue': _quantile_bounds(item_profile['revenue_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_budget': _quantile_bounds(item_profile['budget_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_vote_count': _quantile_bounds(item_profile['vote_count_ml_norm'].values, RANK_CONT_BUCKET_SIZE),
    }

    # 5. Create dataset & dataloader
    print("[5/6] Creating dataset & dataloader...")
    dataset = RankingDataset(samples, user_profile, item_profile, train_hist_seq, encoder.vocabularies['movieId'])
    lookup_tables = _build_feature_lookup_tables(user_profile, item_profile)
    
    # Crucial: Clean up profiles after dataset lookup table is built
    del user_profile, item_profile, samples
    import gc; gc.collect()

    # Optimized for 1TB RAM Linux Server: 
    # - Increase num_workers to 32+ (depending on your CPU cores)
    # - Enable pin_memory: now useful with merged block tensors
    # - Add prefetch_factor to keep the GPU fed
    dataloader = DataLoader(
        dataset, 
        batch_size=RANK_BATCH_SIZE, # Suggest increasing this in config.yaml to 16384+
        shuffle=True,
        num_workers=RANK_NUM_WORKERS,
        collate_fn=dataset.collate_fn,
        pin_memory=True,            # ENABLED for faster transfers of merged blocks
        persistent_workers=True,
        prefetch_factor=4           
    )

    # 6. Build model
    print("[6/6] Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    model = RankingModel(
        vocab_sizes=encoder.vocab_sizes,
        id_embed_dim=RANK_ID_EMBED_DIM,
        genre_embed_dim=RANK_GENRE_EMBED_DIM,
        cont_embed_dim=RANK_CONT_EMBED_DIM,
        cont_bucket_size=RANK_CONT_BUCKET_SIZE,
        cross_layers=RANK_CROSS_LAYERS,
        dropout=RANK_DROPOUT,
        num_experts=RANK_NUM_EXPERTS,
        expert_dim=RANK_EXPERT_DIM,
        tower_dims=RANK_TOWER_DIMS,
        bucket_boundaries=bucket_boundaries,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    lookup_tables = {
        key: torch.from_numpy(value).to(device)
        for key, value in lookup_tables.items()
    }

    # 7. Training loop with CTR-only objective + AMP
    optimizer = optim.Adam(model.parameters(), lr=RANK_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6)

    # AMP: mixed precision for GPU throughput
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32
    print(f"AMP: {'enabled (fp16)' if use_amp else 'disabled'}")

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, RANK_EPOCHS + 1):
        model.train()
        epoch_losses = {'total': []}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Efficiently move the 3 major blocks to GPU
            int_feat = batch['int_features'].to(device, non_blocking=True)
            float_feat = batch['float_features'].to(device, non_blocking=True)
            seq_feat = batch['seq_features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            features = {
                'int_features': int_feat,
                'float_features': float_feat,
                'seq_features': seq_feat,
            }
            ctr_label = labels[:, 0]

            # Forward (AMP autocast for model; BCE-with-logits remains numerically stable)
            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                ctr_logit = model(features)
                total_loss = model.compute_loss(ctr_logit, ctr_label)
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = total_loss.item()
            if not np.isnan(loss_val):
                epoch_losses['total'].append(loss_val)

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.6f}",
            })

        train_loss = float(np.mean(epoch_losses['total']))
        val_loss = _evaluate_validation_loss(
            model, val_subset_df, val_hist_seq, lookup_tables, device, encoder.vocabularies['movieId']
        )
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch} train_loss={train_loss:.4f} "
            f"val_loss(subset)={val_loss:.4f} "
            f"best_val_loss={best_val_loss if best_val_loss < float('inf') else float('nan'):.4f} "
            f"best_epoch={best_epoch} "
            f"no_improve={no_improve}/{EARLY_STOP_PATIENCE}"
        )

        if val_loss < (best_val_loss - EARLY_STOP_MIN_DELTA):
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "ranking_model.pth")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{EARLY_STOP_PATIENCE})")
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Ranking model training finished.")


if __name__ == "__main__":
    main()
