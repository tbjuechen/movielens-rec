import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
import math
from pathlib import Path

class ItemCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.item_sim_matrix = {}

    def fit(self, train_df: pd.DataFrame, top_k=50):
        """
        计算物品相似度矩阵。
        train_df: 必须包含 ['userId', 'movieId'] 字段。
        """
        print("ItemCF: Calculating co-occurrence...")
        user_item_dict = train_df.groupby('userId')['movieId'].apply(list).to_dict()
        
        # 统计物品出现的次数
        item_cnt = defaultdict(int)
        # 统计物品共现次数
        item_sim_matrix = defaultdict(lambda: defaultdict(int))
        
        for user, items in tqdm(user_item_dict.items(), desc="Processing User-Item Co-occurrence"):
            for i in items:
                item_cnt[i] += 1
                for j in items:
                    if i == j:
                        continue
                    # 惩罚活跃用户 (IUF: Inverse User Frequence)
                    item_sim_matrix[i][j] += 1 / math.log(1 + len(items))
        
        print("ItemCF: Normalizing similarity matrix...")
        # 计算余弦相似度并截断
        final_sim_matrix = {}
        for i, related_items in tqdm(item_sim_matrix.items(), desc="Normalizing Matrix"):
            # Sort by similarity and keep top_k
            sorted_items = sorted(
                related_items.items(), 
                key=lambda x: x[1] / math.sqrt(item_cnt[i] * item_cnt[x[0]]), 
                reverse=True
            )[:top_k]
            final_sim_matrix[i] = {k: v / math.sqrt(item_cnt[i] * item_cnt[k]) for k, v in sorted_items}
        
        self.item_sim_matrix = final_sim_matrix
        self.save()

    def retrieve(self, user_history: list, k=50):
        """
        根据用户历史召回相似物品。
        """
        rank = defaultdict(float)
        for item in user_history:
            if item not in self.item_sim_matrix:
                continue
            for related_item, score in self.item_sim_matrix[item].items():
                if related_item in user_history:
                    continue
                rank[related_item] += score
        
        # 返回 Top-K
        sorted_res = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:k]
        return [res[0] for res in sorted_res]

    def save(self):
        self.sim_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sim_save_path, "wb") as f:
            pickle.dump(self.item_sim_matrix, f)
        print(f"ItemCF similarity matrix saved to {self.sim_save_path}")

    def load(self):
        with open(self.sim_save_path, "rb") as f:
            self.item_sim_matrix = pickle.load(f)
        print("ItemCF similarity matrix loaded.")
