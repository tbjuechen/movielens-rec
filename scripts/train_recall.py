import argparse
from loguru import logger
import pandas as pd
from src.models.recall.popularity import PopularityRecall
from src.models.recall.itemcf import ItemCFRecall
from src.models.recall.two_tower import TwoTowerRecall
from pathlib import Path

def train_recall():
    parser = argparse.ArgumentParser(description="训练/预计算召回模型")
    parser.add_argument("--model", type=str, required=True, choices=["popularity", "itemcf", "two_tower"], help="选择召回模型")
    parser.add_argument("--input", type=str, help="数据路径 (若不指定则根据模型选择默认路径)")
    parser.add_argument("--output_dir", type=str, default="saved_models", help="模型保存目录")
    
    # 双塔模型超参数
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=64)
    
    args = parser.parse_args()
    
    # 1. 确定默认输入路径
    if args.input:
        input_path = args.input
    else:
        if args.model == "two_tower":
            input_path = "data/processed/two_tower/train.parquet"
        else:
            input_path = "data/processed/ratings.parquet"

    # 2. 加载数据
    logger.info(f"Loading training data from {input_path}...")
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        return
    df_train = pd.read_parquet(input_path)
    
    # 双塔 V2 特有：加载特征查询表和映射表
    user_features, item_features = None, None
    user_map, movie_map = {}, {}
    if args.model == "two_tower":
        feat_dir = Path("data/processed/two_tower")
        logger.info("Loading feature tables and mappings for Two-Tower V2...")
        user_features = pd.read_parquet(feat_dir / "user_features.parquet")
        item_features = pd.read_parquet(feat_dir / "item_features.parquet")
        with open(feat_dir / "user_map.pkl", "rb") as f: user_map = pickle.load(f)
        with open(feat_dir / "movie_map.pkl", "rb") as f: movie_map = pickle.load(f)

    # 3. 实例化模型并确定保存路径
    if args.model == "popularity":
        model = PopularityRecall()
        save_path = Path(args.output_dir) / "popularity_recall.pkl"
        model.train(df_train)
        model.save(str(save_path))
    elif args.model == "itemcf":
        model = ItemCFRecall()
        save_path = Path(args.output_dir) / "itemcf_recall.pkl"
        model.train(df_train)
        model.save(str(save_path))
    elif args.model == "two_tower":
        model = TwoTowerRecall(embed_dim=args.embed_dim)
        save_path = Path(args.output_dir) / "two_tower_v2"
        model.train(
            df_train, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            user_features=user_features,
            item_features=item_features,
            user_map=user_map,
            movie_map=movie_map
        )
        model.save(str(save_path))
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    logger.success(f"Model {args.model} trained and saved to {save_path}")

if __name__ == "__main__":
    train_recall()
