import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def verify_ranking_data():
    ranking_dir = Path("data/processed/ranking")
    item_p_path = ranking_dir / "item_profile_ranking.parquet"
    user_p_path = ranking_dir / "user_profile_ranking.parquet"
    samples_path = ranking_dir / "ranking_samples_prototype.parquet"

    logger.info("开始精排特征数据验证...")

    # 1. 验证 Item Profile
    if item_p_path.exists():
        df_item = pd.read_parquet(item_p_path)
        logger.info(f"Item Profile | 行数: {len(df_item):,}")
        
        # 检查 Embedding 维度
        if 'embedding' in df_item.columns:
            emb_sample = df_item['embedding'].iloc[0]
            emb_dim = len(emb_sample) if isinstance(emb_sample, (list, np.ndarray)) else 0
            logger.info(f"- 语义向量维度: {emb_dim}")
            
            # 统计非空 Embedding
            empty_count = (df_item['embedding'].apply(lambda x: np.sum(np.abs(x)) == 0)).sum()
            logger.info(f"- 空向量占比: {empty_count/len(df_item)*100:.2f}% (TMDb 缺失导致)")
        
        # 检查主创特征
        crew_coverage = df_item['director_ids'].apply(lambda x: len(x) > 0).mean() * 100
        logger.info(f"- 导演特征覆盖率: {crew_coverage:.2f}%")
    else:
        logger.error("Item Profile 不存在！")

    # 2. 验证 User Profile
    if user_p_path.exists():
        df_user = pd.read_parquet(user_p_path)
        logger.info(f"User Profile | 行数: {len(df_user):,}")
        logger.info(f"- 字段: {df_user.columns.tolist()}")
    else:
        logger.error("User Profile 不存在！")

    # 3. 验证 样本集
    if samples_path.exists():
        df_samples = pd.read_parquet(samples_path)
        logger.info(f"样本集 | 总数: {len(df_samples):,}")
        
        # 检查正负比例
        label_counts = df_samples['label'].value_counts(normalize=True).to_dict()
        logger.info(f"- 正负样本比例: {label_counts}")
        
        # 检查序列特征
        seq_coverage = df_samples['seq_history'].apply(lambda x: len(x) > 0).mean() * 100
        logger.info(f"- 历史序列(Last 5)覆盖率: {seq_coverage:.2f}%")
        
        # 预览
        logger.info("样本集预览:")
        print(df_samples.head(3))
    else:
        logger.error("样本集文件不存在！")

    logger.success("验证结束。")

if __name__ == "__main__":
    verify_ranking_data()
