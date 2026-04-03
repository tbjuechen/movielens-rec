"""CTR-only ranking evaluation on prebuilt candidate pools."""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.settings import (
    PROCESSED_DATA_DIR, FEATURE_STORE_DIR, MODEL_WEIGHTS_DIR,
    USER_TOP_GENRES_MAX_LEN, ITEM_GENRES_MAX_LEN, RANK_HIST_SEQ_MAXLEN,
    RANK_ID_EMBED_DIM, RANK_GENRE_EMBED_DIM, RANK_CONT_EMBED_DIM,
    RANK_CONT_BUCKET_SIZE,
    RANK_CROSS_LAYERS, RANK_DROPOUT,
    RANK_NUM_EXPERTS, RANK_EXPERT_DIM, RANK_TOWER_DIMS,
    RANK_EVAL_KS,
    ALIGNMENT_STRICT_POOL_SIZE, ALIGNMENT_STRICT_VAL_SAMPLE_SIZE, ALIGNMENT_STRICT_TEST_SAMPLE_FRAC,
)
from src.config.alignment import STRICT_MINONICC_MODE, get_alignment_paths
from src.features.encoder import FeatureEncoder
from src.models.ranking.ranker import RankingModel
from src.evaluation.metrics import hitrate_at_k, ndcg_at_k, mrr


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


def _encode_hist_matrix(hist_seq, movie_vocab):
    encoded = np.zeros(hist_seq.shape, dtype=np.int64)
    for row_idx in range(hist_seq.shape[0]):
        encoded[row_idx] = [
            movie_vocab.get(int(mid), 0) if int(mid) > 0 else 0
            for mid in hist_seq[row_idx]
        ]
    return encoded


def _score_candidates(
    ranking_model,
    uid,
    candidates,
    hist_row,
    device,
    user_encoded_id,
    user_top_genres_arr,
    user_cont_arr,
    item_encoded_id,
    item_genres_arr,
    item_cont_arr,
):
    cand_arr = torch.tensor(candidates, dtype=torch.long, device=device)
    n_cand = len(cand_arr)
    int_features = torch.cat([
        user_encoded_id[uid].expand(n_cand, 1),
        item_encoded_id[cand_arr].unsqueeze(1),
        user_top_genres_arr[uid].expand(n_cand, -1),
        item_genres_arr[cand_arr]
    ], dim=1).contiguous()
    float_features = torch.cat([
        user_cont_arr[uid].expand(n_cand, -1),
        item_cont_arr[cand_arr]
    ], dim=1).contiguous()
    seq_features = torch.from_numpy(hist_row).to(device).unsqueeze(0).expand(n_cand, -1).contiguous()
    features = {
        "int_features": int_features,
        "float_features": float_features,
        "seq_features": seq_features,
    }
    with torch.no_grad():
        ctr_logit = ranking_model(features)
        final_score = torch.sigmoid(ctr_logit).view(-1)
        order = final_score.argsort(descending=True)
        ranked = [candidates[i] for i in order.cpu().numpy().reshape(-1)]
    return ranked


def evaluate(
    test_mode=False,
    eval_set='val',
    sample_size=None,
    sample_frac=None,
    sample_seed=42,
    pool_size=None,
    alignment=None,
    json_out=None,
):
    print(f"=== Ranking Evaluation ({eval_set.upper()} Set) ===")
    paths = get_alignment_paths(alignment)
    processed_dir = paths.processed_dir
    feature_store_dir = paths.feature_store_dir
    model_weights_dir = paths.model_weights_dir
    strict_mode = paths.mode == STRICT_MINONICC_MODE

    # 1. Load data & encoder
    ranking_val_path = Path(processed_dir) / f"ranking_{eval_set}_candidates.parquet"
    ranking_hist_path = Path(processed_dir) / f"ranking_{eval_set}_hist_seq.npy"
    if not ranking_val_path.exists():
        print(f"ERROR: {ranking_val_path} not found. Run 05_build_ranking_data.py first.")
        return
    if not ranking_hist_path.exists():
        print(f"ERROR: {ranking_hist_path} not found. Run 05_build_ranking_data.py first.")
        return
    
    val_df = pd.read_parquet(ranking_val_path)
    hist_seq = np.load(ranking_hist_path)
    if hist_seq.shape != (len(val_df), RANK_HIST_SEQ_MAXLEN):
        raise ValueError(
            f"hist_seq shape mismatch for {eval_set}: expected {(len(val_df), RANK_HIST_SEQ_MAXLEN)}, got {hist_seq.shape}"
        )
    if sample_size is not None and sample_frac is not None:
        raise ValueError("Only one of sample_size or sample_frac can be set.")
    if strict_mode and pool_size is None:
        pool_size = ALIGNMENT_STRICT_POOL_SIZE
    if strict_mode and sample_size is None and sample_frac is None:
        if eval_set == "val":
            sample_size = ALIGNMENT_STRICT_VAL_SAMPLE_SIZE
        elif eval_set == "test":
            sample_frac = ALIGNMENT_STRICT_TEST_SAMPLE_FRAC

    if sample_size is not None:
        sample_size = min(int(sample_size), len(val_df))
        sampled_idx = val_df.sample(n=sample_size, random_state=sample_seed).index.to_numpy()
        val_df = val_df.loc[sampled_idx].reset_index(drop=True)
        hist_seq = hist_seq[sampled_idx]
    elif sample_frac is not None:
        if not (0.0 < sample_frac <= 1.0):
            raise ValueError("sample_frac must be in (0, 1].")
        sample_size = max(1, int(len(val_df) * float(sample_frac)))
        sampled_idx = val_df.sample(n=sample_size, random_state=sample_seed).index.to_numpy()
        val_df = val_df.loc[sampled_idx].reset_index(drop=True)
        hist_seq = hist_seq[sampled_idx]

    if test_mode:
        val_df = val_df.head(1000)
        hist_seq = hist_seq[:len(val_df)]
    
    user_profile = pd.read_parquet(Path(processed_dir) / "user_profile.parquet")
    item_profile = pd.read_parquet(Path(processed_dir) / "item_profile.parquet")

    encoder = FeatureEncoder(feature_store_dir)
    encoder.load()
    user_profile, item_profile = apply_encoding(user_profile, item_profile, encoder)
    hist_seq_encoded = _encode_hist_matrix(hist_seq, encoder.vocabularies['movieId'])

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    # 2. Load ranking model
    ranking_model = RankingModel(
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
        user_genre_max_len=USER_TOP_GENRES_MAX_LEN,
        item_genre_max_len=ITEM_GENRES_MAX_LEN,
    ).to(device)
    ranking_path = Path(model_weights_dir) / "ranking_model.pth"
    if not ranking_path.exists():
        print(f"ERROR: {ranking_path} not found.")
        return
    ranking_model.load_state_dict(torch.load(ranking_path, map_location=device, weights_only=True))
    ranking_model.eval()

    # 3. Prepare Feature Lookup Matrices (Optimized for evaluation)
    # Item Lookup
    max_iid = int(item_profile['movieId'].max())
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

    item_encoded_id = torch.from_numpy(item_encoded_id).to(device)
    item_genres_arr = torch.from_numpy(item_genres_arr).to(device)
    item_cont_arr = torch.from_numpy(item_cont_arr).to(device)

    # User Lookup
    max_uid = int(user_profile['userId'].max())
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

    user_encoded_id = torch.from_numpy(user_encoded_id).to(device)
    user_top_genres_arr = torch.from_numpy(user_top_genres_arr).to(device)
    user_cont_arr = torch.from_numpy(user_cont_arr).to(device)

    # 4. Evaluation Loop
    metrics_capability = defaultdict(list)
    metrics_e2e = defaultdict(list)
    n_target_in_eval_pool = 0
    n_target_in_raw_pool = 0
    force_inserted_count = 0
    empty_pool_fallback_count = 0
    
    effective_pool_desc = f"{pool_size}" if pool_size else "full"
    print(
        f"Ranking {len(val_df):,} users with batch scoring "
        f"(pool={effective_pool_desc}, sample_seed={sample_seed})..."
    )
    
    for row_idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Ranking"):
        uid = int(row['userId'])
        candidates = list(row['candidates'])
        raw_candidates = list(row.get('raw_candidates', [])) if isinstance(row.get('raw_candidates', []), (list, np.ndarray)) else []
        if pool_size is not None:
            candidates = candidates[:pool_size]
            raw_candidates = raw_candidates[:pool_size]
        actual_items = list(row['actual'])
        
        if not candidates:
            continue
        if set(actual_items) & set(candidates):
            n_target_in_eval_pool += 1
        if raw_candidates and set(actual_items) & set(raw_candidates):
            n_target_in_raw_pool += 1
        force_inserted_count += int(row.get('target_force_inserted', False))
        empty_pool_fallback_count += int(row.get('target_pool_was_empty', False))

        # Candidate-pool baseline metrics (capability pool)
        for ek in RANK_EVAL_KS:
            metrics_capability[f'CandidatePool_HR@{ek}'].append(hitrate_at_k(actual_items, candidates, k=ek))
            metrics_capability[f'CandidatePool_NDCG@{ek}'].append(ndcg_at_k(actual_items, candidates, k=ek))
        metrics_capability['CandidatePool_MRR'].append(mrr(actual_items, candidates))

        ranked = _score_candidates(
            ranking_model=ranking_model,
            uid=uid,
            candidates=candidates,
            hist_row=hist_seq_encoded[row_idx],
            device=device,
            user_encoded_id=user_encoded_id,
            user_top_genres_arr=user_top_genres_arr,
            user_cont_arr=user_cont_arr,
            item_encoded_id=item_encoded_id,
            item_genres_arr=item_genres_arr,
            item_cont_arr=item_cont_arr,
        )

        for ek in RANK_EVAL_KS:
            metrics_capability[f'Ranking_HR@{ek}'].append(hitrate_at_k(actual_items, ranked, k=ek))
            metrics_capability[f'Ranking_NDCG@{ek}'].append(ndcg_at_k(actual_items, ranked, k=ek))
        metrics_capability['Ranking_MRR'].append(mrr(actual_items, ranked))

        if raw_candidates:
            for ek in RANK_EVAL_KS:
                metrics_e2e[f'CandidatePool_HR@{ek}'].append(hitrate_at_k(actual_items, raw_candidates, k=ek))
                metrics_e2e[f'CandidatePool_NDCG@{ek}'].append(ndcg_at_k(actual_items, raw_candidates, k=ek))
            metrics_e2e['CandidatePool_MRR'].append(mrr(actual_items, raw_candidates))

            ranked_raw = _score_candidates(
                ranking_model=ranking_model,
                uid=uid,
                candidates=raw_candidates,
                hist_row=hist_seq_encoded[row_idx],
                device=device,
                user_encoded_id=user_encoded_id,
                user_top_genres_arr=user_top_genres_arr,
                user_cont_arr=user_cont_arr,
                item_encoded_id=item_encoded_id,
                item_genres_arr=item_genres_arr,
                item_cont_arr=item_cont_arr,
            )
            for ek in RANK_EVAL_KS:
                metrics_e2e[f'Ranking_HR@{ek}'].append(hitrate_at_k(actual_items, ranked_raw, k=ek))
                metrics_e2e[f'Ranking_NDCG@{ek}'].append(ndcg_at_k(actual_items, ranked_raw, k=ek))
            metrics_e2e['Ranking_MRR'].append(mrr(actual_items, ranked_raw))

    # 5. Report
    n_total = len(val_df)
    print(f"\n=== Ranking Evaluation Report ({eval_set.upper()}) ===")
    print(
        f"Total users: {n_total}, Target in eval pool: {n_target_in_eval_pool} "
        f"({n_target_in_eval_pool/max(n_total,1)*100:.1f}%)"
    )
    print(
        f"Forced target insertions: {force_inserted_count} "
        f"({force_inserted_count/max(n_total,1)*100:.1f}%), "
        f"Empty-pool fallback: {empty_pool_fallback_count} "
        f"({empty_pool_fallback_count/max(n_total,1)*100:.1f}%)\n"
    )

    print(f"Raw target in pool (pre-force): {n_target_in_raw_pool} ({n_target_in_raw_pool/max(n_total,1)*100:.1f}%)\n")

    print("[Ranking Capability Metrics]")
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    for name in sorted(metrics_capability.keys()):
        vals = metrics_capability[name]
        if vals:
            print(f"{name:<30} {np.mean(vals):>10.4f}")

    if metrics_e2e:
        print("\n[End-to-End Metrics (Raw Pool, No Force Insert)]")
        print(f"{'Metric':<30} {'Value':>10}")
        print("-" * 42)
        for name in sorted(metrics_e2e.keys()):
            vals = metrics_e2e[name]
            if vals:
                print(f"{name:<30} {np.mean(vals):>10.4f}")

    report = {
        "alignment_mode": paths.mode,
        "eval_set": eval_set,
        "sample_seed": sample_seed,
        "pool_size": pool_size,
        "num_users": int(n_total),
        "target_in_eval_pool": int(n_target_in_eval_pool),
        "target_in_raw_pool": int(n_target_in_raw_pool),
        "force_inserted_count": int(force_inserted_count),
        "empty_pool_fallback_count": int(empty_pool_fallback_count),
        "ranking_capability": {
            k: float(np.mean(v)) for k, v in metrics_capability.items() if v
        },
        "end_to_end": {
            k: float(np.mean(v)) for k, v in metrics_e2e.items() if v
        },
    }
    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to: {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Quick eval with 1000 users")
    parser.add_argument("--set", type=str, default="val", choices=['val', 'test'], help="Set to evaluate: val or test")
    parser.add_argument("--alignment", type=str, default=None, help="Alignment mode, e.g. strict_minonicc")
    parser.add_argument("--sample-size", type=int, default=None, help="Randomly sample N users for evaluation")
    parser.add_argument("--sample-frac", type=float, default=None, help="Randomly sample a fraction of users for evaluation")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling users")
    parser.add_argument("--pool-size", type=int, default=None, help="Truncate candidate pool to top-N before ranking")
    parser.add_argument("--json-out", type=str, default=None, help="Write evaluation report JSON to this path")
    args = parser.parse_args()
    evaluate(
        test_mode=args.test,
        eval_set=args.set,
        sample_size=args.sample_size,
        sample_frac=args.sample_frac,
        sample_seed=args.sample_seed,
        pool_size=args.pool_size,
        alignment=args.alignment,
        json_out=args.json_out,
    )
