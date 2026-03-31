"""Train the ranking model (DCNv2 + MMoE) with pCTR + pRating dual objectives."""
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
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN,
    RANK_ID_EMBED_DIM, RANK_GENRE_EMBED_DIM, RANK_CONT_EMBED_DIM,
    RANK_CONT_BUCKET_SIZE,
    RANK_CROSS_LAYERS, RANK_DROPOUT,
    RANK_NUM_EXPERTS, RANK_EXPERT_DIM, RANK_TOWER_DIMS,
    RANK_BATCH_SIZE, RANK_LEARNING_RATE, RANK_EPOCHS,
    RANK_NEG_SAMPLE_RATIO, RANK_NUM_WORKERS,
    RANK_GRADNORM_ALPHA,
)
from src.features.encoder import FeatureEncoder
from src.models.ranking.ranker import RankingModel, _quantile_bounds
from src.data_pipeline.ranking_dataset import RankingDataset, build_ranking_samples


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
    if not ranking_samples_path.exists():
        print(f"  {ranking_samples_path} not found. Please run 05_build_ranking_data.py first.")
        sys.exit(1)
    else:
        # Optimization: Use pyarrow to load the pool
        import pyarrow.parquet as pq
        cols = ['userId', 'movieId', 'ctr_label', 'rating_norm', 'candidate_pool']
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
        
        samples = {
            'userId': torch.from_numpy(table['userId'].to_numpy().astype(np.int64)),
            'movieId': torch.from_numpy(table['movieId'].to_numpy().astype(np.int64)),
            'ctr_label': torch.from_numpy(table['ctr_label'].to_numpy().astype(np.float32)),
            'rating_norm': torch.from_numpy(table['rating_norm'].to_numpy().astype(np.float32)),
            'candidate_pool': torch.from_numpy(pool_matrix),
        }
        print(f"  Loaded {n_rows:,} interactions with pool width {pool_width} (via pyarrow)")
        del table, raw_pool, pool_matrix; import gc; gc.collect()
        
    n_total = len(samples['userId'])
    print(f"  Unique positives: {n_total:,} (Negative sampling will be 1:3 dynamic)")

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
    dataset = RankingDataset(samples, user_profile, item_profile)
    
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

    # 7. Training loop with GradNorm + AMP
    optimizer = optim.Adam(model.parameters(), lr=RANK_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-6)

    # AMP: mixed precision for GPU throughput
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32
    print(f"AMP: {'enabled (fp16)' if use_amp else 'disabled'}")

    # GradNorm: learnable task weights (update every N steps to reduce overhead)
    log_task_weights = torch.zeros(2, requires_grad=True, device=device)
    weight_optimizer = optim.Adam([log_task_weights], lr=0.01)
    initial_losses = None
    shared_layer = model.cross_net.linears[-1].weight  # shared layer for grad norm
    GRADNORM_INTERVAL = 100  # Increased to 100 to reduce main process overhead

    best_loss = float('inf')
    patience = 3
    no_improve = 0
    epoch = 0
    global_step = 0
    while True:
        epoch += 1
        model.train()
        epoch_losses = {'total': [], 'bce': [], 'mse': [], 'w_ctr': [], 'w_mse': []}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Efficiently move the 3 major blocks to GPU
            int_feat = batch['int_features'].to(device, non_blocking=True)
            float_feat = batch['float_features'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            features = {
                'int_features': int_feat,
                'float_features': float_feat
            }
            ctr_label = labels[:, 0]
            rating_label = labels[:, 1]
            has_rating = labels[:, 2].bool()

            # Compute task weights (clamp min 0.1, then renormalize to sum=2)
            task_weights_raw = torch.softmax(log_task_weights, dim=0).clamp(min=0.1)
            task_weights = task_weights_raw / task_weights_raw.sum() * 2

            do_gradnorm = (global_step % GRADNORM_INTERVAL == 0)

            # Forward (AMP autocast for model; loss uses bce_with_logits which is AMP-safe)
            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                ctr_logit, pRating = model(features)
                loss_bce, loss_mse = model.compute_loss(
                    ctr_logit, pRating, ctr_label, rating_label, has_rating
                )

            # --- GradNorm: only every N steps (expensive due to retain_graph) ---
            if do_gradnorm:
                G_ctr = torch.norm(task_weights[0] * torch.autograd.grad(
                    loss_bce, shared_layer, retain_graph=True, create_graph=True)[0])
                G_mse = torch.norm(task_weights[1] * torch.autograd.grad(
                    loss_mse, shared_layer, retain_graph=True, create_graph=True)[0])
                G_avg = (G_ctr + G_mse) / 2

                with torch.no_grad():
                    if initial_losses is None:
                        initial_losses = [loss_bce.item(), loss_mse.item()]
                    r_ctr = loss_bce.item() / max(initial_losses[0], 1e-8)
                    r_mse = loss_mse.item() / max(initial_losses[1], 1e-8)
                    r_avg = (r_ctr + r_mse) / 2
                    target_ctr = G_avg * (r_ctr / max(r_avg, 1e-8)) ** RANK_GRADNORM_ALPHA
                    target_mse = G_avg * (r_mse / max(r_avg, 1e-8)) ** RANK_GRADNORM_ALPHA

                gradnorm_loss = torch.abs(G_ctr - target_ctr) + torch.abs(G_mse - target_mse)
                weight_optimizer.zero_grad()
                gradnorm_loss.backward(retain_graph=True)
                weight_optimizer.step()

            # --- Model update (use detached weights to avoid double grad) ---
            total_loss = task_weights[0].detach() * loss_bce + task_weights[1].detach() * loss_mse
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            w_ctr = task_weights[0].item()
            w_mse = task_weights[1].item()
            loss_val = total_loss.item()
            # Skip NaN batches from AMP overflow in epoch stats
            if not np.isnan(loss_val):
                epoch_losses['total'].append(loss_val)
                epoch_losses['bce'].append(loss_bce.item())
                epoch_losses['mse'].append(loss_mse.item())
                epoch_losses['w_ctr'].append(w_ctr)
                epoch_losses['w_mse'].append(w_mse)

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'BCE': f"{loss_bce.item():.4f}",
                'MSE': f"{loss_mse.item():.4f}",
                'w_ctr': f"{w_ctr:.2f}",
                'w_mse': f"{w_mse:.2f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.6f}",
            })

        avg_bce = np.mean(epoch_losses['bce'])
        avg_mse = np.mean(epoch_losses['mse'])
        # Use unweighted sum for monitoring to avoid GradNorm weight jitter
        monitor_loss = avg_bce + avg_mse
        
        scheduler.step(monitor_loss)
        print(f"Epoch {epoch} avg: BCE={avg_bce:.4f} MSE={avg_mse:.4f} "
              f"monitor_sum={monitor_loss:.4f} "
              f"w_ctr={np.mean(epoch_losses['w_ctr']):.2f} "
              f"w_mse={np.mean(epoch_losses['w_mse']):.2f}")

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            no_improve = 0
            Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "ranking_model.pth")
            print(f"  -> Saved best model (monitor_sum={best_loss:.4f})")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Ranking model training finished.")


if __name__ == "__main__":
    main()
