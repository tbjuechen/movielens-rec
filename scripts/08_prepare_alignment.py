import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config.alignment import STRICT_MINONICC_MODE, get_alignment_paths
from src.config.settings import (
    ALIGNMENT_SEED,
    ALIGNMENT_STRICT_LONG_USER_RATIO,
    ALIGNMENT_STRICT_LONG_USER_THRESHOLD,
    ALIGNMENT_STRICT_USER_PACK_SIZE,
    PROCESSED_DATA_DIR,
)
from src.features.encoder import FeatureEncoder


def _build_user_pack(train_data, val_data, test_data, seed, pack_size, long_ratio, long_threshold):
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    all_data = all_data.sort_values(["userId", "timestamp"])
    seq_lengths = all_data.groupby("userId").size()
    valid_users = seq_lengths[seq_lengths >= 5].index.to_numpy()
    if len(valid_users) < pack_size:
        raise ValueError(f"Not enough valid users: {len(valid_users)} < {pack_size}")

    long_users = seq_lengths[(seq_lengths.index.isin(valid_users)) & (seq_lengths > long_threshold)].index.to_numpy()
    other_users = np.setdiff1d(valid_users, long_users, assume_unique=False)

    rng = np.random.default_rng(seed)
    n_long_target = min(int(pack_size * long_ratio), len(long_users))
    sampled_long = rng.choice(long_users, size=n_long_target, replace=False) if n_long_target > 0 else np.array([], dtype=np.int64)

    n_other_target = pack_size - len(sampled_long)
    if len(other_users) < n_other_target:
        fallback_pool = np.setdiff1d(valid_users, sampled_long, assume_unique=False)
        sampled_other = rng.choice(fallback_pool, size=n_other_target, replace=False)
    else:
        sampled_other = rng.choice(other_users, size=n_other_target, replace=False)

    sampled_users = np.concatenate([sampled_long, sampled_other]).astype(np.int64)
    rng.shuffle(sampled_users)

    pack = pd.DataFrame(
        {
            "userId": sampled_users,
            "is_long_user": np.isin(sampled_users, sampled_long),
            "seq_len_total": seq_lengths.loc[sampled_users].values.astype(np.int64),
        }
    )
    return pack


def _fit_and_save_encoder(user_profile, item_profile, feature_store_dir):
    encoder = FeatureEncoder(feature_store_dir)
    encoder.fit_categorical(user_profile["userId"], "userId")
    encoder.fit_categorical(item_profile["movieId"], "movieId")
    all_genres = pd.concat(
        [
            user_profile["top_genres"].explode(),
            item_profile["tmdb_genres"].explode(),
        ]
    ).dropna()
    encoder.fit_categorical(all_genres, "genres")
    encoder.fit_continuous(user_profile, ["avg_rating", "activity"], prefix="user")
    encoder.fit_continuous(item_profile, ["release_year_orig", "avg_rating", "revenue", "budget", "vote_count_ml"], prefix="item")
    encoder.save()


def main():
    parser = argparse.ArgumentParser(description="Prepare strict_minonicc aligned data package.")
    parser.add_argument("--alignment", type=str, default=STRICT_MINONICC_MODE, choices=[STRICT_MINONICC_MODE])
    parser.add_argument("--seed", type=int, default=ALIGNMENT_SEED)
    parser.add_argument("--user-pack-size", type=int, default=ALIGNMENT_STRICT_USER_PACK_SIZE)
    parser.add_argument("--long-user-ratio", type=float, default=ALIGNMENT_STRICT_LONG_USER_RATIO)
    parser.add_argument("--long-user-threshold", type=int, default=ALIGNMENT_STRICT_LONG_USER_THRESHOLD)
    args = parser.parse_args()

    base_processed = Path(PROCESSED_DATA_DIR)
    train_path = base_processed / "train_data.parquet"
    val_path = base_processed / "val_data.parquet"
    test_path = base_processed / "test_data.parquet"
    user_profile_path = base_processed / "user_profile.parquet"
    item_profile_path = base_processed / "item_profile.parquet"

    for p in (train_path, val_path, test_path, user_profile_path, item_profile_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print("Loading base processed datasets...")
    train_data = pd.read_parquet(train_path)
    val_data = pd.read_parquet(val_path)
    test_data = pd.read_parquet(test_path)
    user_profile = pd.read_parquet(user_profile_path)
    item_profile = pd.read_parquet(item_profile_path)

    print("Sampling aligned user pack...")
    user_pack = _build_user_pack(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        seed=args.seed,
        pack_size=args.user_pack_size,
        long_ratio=args.long_user_ratio,
        long_threshold=args.long_user_threshold,
    )
    selected_users = set(user_pack["userId"].tolist())

    print("Filtering datasets to aligned user pack...")
    train_aligned = train_data[train_data["userId"].isin(selected_users)].copy()
    val_aligned = val_data[val_data["userId"].isin(selected_users)].copy()
    test_aligned = test_data[test_data["userId"].isin(selected_users)].copy()
    user_profile_aligned = user_profile[user_profile["userId"].isin(selected_users)].copy()

    aligned_movie_ids = set(
        pd.concat(
            [
                train_aligned["movieId"],
                val_aligned["movieId"],
                test_aligned["movieId"],
            ],
            ignore_index=True,
        ).dropna().astype(np.int64).tolist()
    )
    item_profile_aligned = item_profile[item_profile["movieId"].isin(aligned_movie_ids)].copy()

    paths = get_alignment_paths(args.alignment)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.feature_store_dir.mkdir(parents=True, exist_ok=True)
    paths.model_weights_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    print("Saving aligned processed data...")
    train_aligned.to_parquet(paths.processed_dir / "train_data.parquet", index=False)
    val_aligned.to_parquet(paths.processed_dir / "val_data.parquet", index=False)
    test_aligned.to_parquet(paths.processed_dir / "test_data.parquet", index=False)
    user_profile_aligned.to_parquet(paths.processed_dir / "user_profile.parquet", index=False)
    item_profile_aligned.to_parquet(paths.processed_dir / "item_profile.parquet", index=False)
    user_pack.to_parquet(paths.processed_dir / "alignment_user_pack.parquet", index=False)

    print("Building aligned feature store artifacts...")
    _fit_and_save_encoder(user_profile_aligned, item_profile_aligned, paths.feature_store_dir)

    genre_to_items = (
        item_profile_aligned.explode("tmdb_genres")
        .dropna(subset=["tmdb_genres"])
        .groupby("tmdb_genres")["movieId"]
        .apply(lambda x: [int(v) for v in x.tolist()])
        .to_dict()
    )
    popularity_list = (
        train_aligned[train_aligned["rating"] >= 3.0]
        .groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .index.astype(np.int64)
        .tolist()
    )

    with open(paths.feature_store_dir / "genre_to_items.json", "w") as f:
        json.dump(genre_to_items, f)
    with open(paths.feature_store_dir / "popularity_list.json", "w") as f:
        json.dump(popularity_list, f)

    metadata = {
        "alignment_mode": args.alignment,
        "seed": args.seed,
        "user_pack_size": int(len(user_pack)),
        "long_user_ratio_target": float(args.long_user_ratio),
        "long_user_threshold": int(args.long_user_threshold),
        "long_user_count": int(user_pack["is_long_user"].sum()),
        "train_rows": int(len(train_aligned)),
        "val_rows": int(len(val_aligned)),
        "test_rows": int(len(test_aligned)),
        "item_count": int(len(item_profile_aligned)),
    }
    with open(paths.processed_dir / "alignment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Alignment data prepared successfully.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

