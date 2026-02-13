import argparse
from loguru import logger
import pandas as pd
from src.models.recall.popularity import PopularityRecall
from pathlib import Path

def train_recall():
    parser = argparse.ArgumentParser(description="训练/预计算召回模型")
    parser.add_argument("--model", type=str, required=True, choices=["popularity"], help="选择召回模型")
    parser.add_argument("--input", type=str, default="data/processed/ratings.parquet", help="评分数据路径")
    parser.add_argument("--output_dir", type=str, default="saved_models", help="模型保存目录")
    
    args = parser.parse_args()
    
    # 1. 加载数据
    logger.info(f"Loading training data from {args.input}...")
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return

    df_train = pd.read_parquet(args.input)
    
    # 2. 实例化模型
    if args.model == "popularity":
        model = PopularityRecall()
        save_path = Path(args.output_dir) / "popularity_recall.pkl"
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # 3. 训练并保存
    model.train(df_train)
    model.save(str(save_path))
    
    logger.success(f"Model {args.model} trained and saved to {save_path}")

if __name__ == "__main__":
    train_recall()
