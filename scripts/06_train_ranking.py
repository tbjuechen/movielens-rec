"""Train the ranking model (DCNv2 + MMoE) with pCTR + pRating dual objectives."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import (
    PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR,
    USER_HISTORY_MAX_LEN, USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN,
    RANK_ID_EMBED_DIM, RANK_GENRE_EMBED_DIM, RANK_CONT_EMBED_DIM,
    RANK_CONT_BUCKET_SIZE, RANK_PRETRAINED_EMB_DIM,
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

    # 1. Load data
    print("[1/7] Loading data...")
    train_data = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "train_data.parquet")
    user_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "item_profile.parquet")
    print(f"  train={len(train_data):,}, users={len(user_profile):,}, items={len(item_profile):,}")

    print("[2/7] Encoding features...")
    encoder = FeatureEncoder(FEATURE_STORE_DIR)
    encoder.load()
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)

    # 2. Load pretrained embeddings
    print("[3/7] Loading pretrained embeddings...")
    pt_user_emb = np.load(Path(FEATURE_STORE_DIR) / "pretrained_user_emb.npy")
    pt_item_emb = np.load(Path(FEATURE_STORE_DIR) / "pretrained_item_emb.npy")
    print(f"  user {pt_user_emb.shape}, item {pt_item_emb.shape}")

    # 3. Build training samples
    print("[4/7] Building training samples...")
    all_item_ids = item_profile['movieId'].values
    samples = build_ranking_samples(train_data, all_item_ids,
                                    neg_sample_ratio=RANK_NEG_SAMPLE_RATIO, seed=42)
    n_pos = int((samples[:, 2] > 0.5).sum())
    n_neg = len(samples) - n_pos
    print(f"  total={len(samples):,} (pos={n_pos:,}, neg={n_neg:,})")

    # 4. Compute quantile bucket boundaries from training profiles
    print("[5/7] Computing bucket boundaries...")
    bucket_boundaries = {
        'user_avg_rating': _quantile_bounds(user_profile['avg_rating_norm'].values, RANK_CONT_BUCKET_SIZE),
        'user_activity': _quantile_bounds(user_profile['activity_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_release_year': _quantile_bounds(item_profile['release_year_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_avg_rating': _quantile_bounds(item_profile['avg_rating_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_revenue': _quantile_bounds(item_profile['revenue_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_budget': _quantile_bounds(item_profile['budget_norm'].values, RANK_CONT_BUCKET_SIZE),
        'item_vote_count': _quantile_bounds(item_profile['vote_count_ml_norm'].values, RANK_CONT_BUCKET_SIZE),
    }
    # recall_sim_score bounds: compute from pretrained embeddings
    valid_users = user_profile['userId'].values
    valid_items = item_profile['movieId'].values
    # Sample random user-item pairs to estimate sim score distribution
    rng = np.random.RandomState(42)
    n_sample = min(100000, len(valid_users) * 10)
    sample_uids = rng.choice(valid_users, n_sample)
    sample_iids = rng.choice(valid_items, n_sample)
    sim_scores = (pt_user_emb[sample_uids] * pt_item_emb[sample_iids]).sum(axis=1) / RANK_PRETRAINED_EMB_DIM
    bucket_boundaries['recall_sim_score'] = _quantile_bounds(sim_scores, RANK_CONT_BUCKET_SIZE)

    # 5. Create dataset & dataloader
    print("[6/7] Creating dataset & dataloader...")
    dataset = RankingDataset(samples, user_profile, item_profile, pt_user_emb, pt_item_emb)
    dataloader = DataLoader(
        dataset, batch_size=RANK_BATCH_SIZE, shuffle=True,
        num_workers=RANK_NUM_WORKERS, persistent_workers=(RANK_NUM_WORKERS > 0),
        pin_memory=True, prefetch_factor=4 if RANK_NUM_WORKERS > 0 else None,
    )

    # 6. Build model
    print("[7/7] Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    model = RankingModel(
        vocab_sizes=encoder.vocab_sizes,
        id_embed_dim=RANK_ID_EMBED_DIM,
        genre_embed_dim=RANK_GENRE_EMBED_DIM,
        cont_embed_dim=RANK_CONT_EMBED_DIM,
        cont_bucket_size=RANK_CONT_BUCKET_SIZE,
        pretrained_emb_dim=RANK_PRETRAINED_EMB_DIM,
        cross_layers=RANK_CROSS_LAYERS,
        dropout=RANK_DROPOUT,
        num_experts=RANK_NUM_EXPERTS,
        expert_dim=RANK_EXPERT_DIM,
        tower_dims=RANK_TOWER_DIMS,
        bucket_boundaries=bucket_boundaries,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 7. Training loop with GradNorm
    optimizer = optim.Adam(model.parameters(), lr=RANK_LEARNING_RATE)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=RANK_LEARNING_RATE,
        total_steps=len(dataloader) * RANK_EPOCHS,
        pct_start=0.1,          # 10% warmup
        anneal_strategy='cos',
        div_factor=10,           # start_lr = max_lr / 10
        final_div_factor=100,    # end_lr = max_lr / 1000
    )

    # GradNorm: learnable task weights
    log_task_weights = torch.zeros(2, requires_grad=True, device=device)
    weight_optimizer = optim.Adam([log_task_weights], lr=0.01)
    initial_losses = None
    shared_layer = model.cross_net.linears[-1].weight  # shared layer for grad norm

    best_loss = float('inf')
    for epoch in range(RANK_EPOCHS):
        model.train()
        epoch_losses = {'total': [], 'bce': [], 'mse': [], 'w_ctr': [], 'w_mse': []}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{RANK_EPOCHS}")

        for batch in pbar:
            features = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                        if k not in ('ctr_label', 'rating_label', 'has_rating')}
            ctr_label = batch['ctr_label'].to(device, non_blocking=True)
            rating_label = batch['rating_label'].to(device, non_blocking=True)
            has_rating = batch['has_rating'].to(device, non_blocking=True)

            # Forward
            pCTR, pRating = model(features)
            loss_bce, loss_mse = model.compute_loss(
                pCTR, pRating, ctr_label, rating_label, has_rating
            )

            # Compute task weights
            task_weights = torch.softmax(log_task_weights, dim=0) * 2

            # --- GradNorm: compute BEFORE model update (graph still intact) ---
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
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            w_ctr = task_weights[0].item()
            w_mse = task_weights[1].item()
            epoch_losses['total'].append(total_loss.item())
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
                'LR': f"{scheduler.get_last_lr()[0]:.6f}",
            })

        avg_loss = np.mean(epoch_losses['total'])
        print(f"Epoch {epoch + 1} avg: loss={avg_loss:.4f} "
              f"BCE={np.mean(epoch_losses['bce']):.4f} "
              f"MSE={np.mean(epoch_losses['mse']):.4f} "
              f"w_ctr={np.mean(epoch_losses['w_ctr']):.2f} "
              f"w_mse={np.mean(epoch_losses['w_mse']):.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            Path(MODEL_WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(MODEL_WEIGHTS_DIR) / "ranking_model.pth")
            print(f"  -> Saved best model (loss={best_loss:.4f})")

    print("Ranking model training finished.")


if __name__ == "__main__":
    main()
