import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
import math
from pathlib import Path

class UserCFModel:
    def __init__(self, sim_save_path: str):
        self.sim_save_path = Path(sim_save_path)
        self.user_sim_matrix = {}
        self.user_item_dict = {}

    def fit(self, train_df: pd.DataFrame, top_k=50):
        """
        计算用户相似度矩阵。
        """
        print("UserCF: Calculating user similarity via item-inverted index...")
        # 1. 建立物品到用户的倒排表
        item_user_dict = train_df.groupby('movieId')['userId'].apply(list).to_dict()
        self.user_item_dict = train_df.groupby('userId')['movieId'].apply(list).to_dict()
        
        user_cnt = defaultdict(int)
        user_sim_matrix = defaultdict(lambda: defaultdict(int))
        
        for item, users in tqdm(item_user_dict.items(), desc="Processing Item-User Inverted Index"):
            for u in users:
                user_cnt[u] += 1
                for v in users:
                    if u == v:
                        continue
                    # 惩罚热门物品 (IIF: Inverse Item Frequency)
                    user_sim_matrix[u][v] += 1 / math.log(1 + len(users))
        
        print("UserCF: Normalizing similarity matrix...")
        final_sim_matrix = {}
        for u, related_users in tqdm(user_sim_matrix.items(), desc="Normalizing Matrix"):
            # Sort and keep top_k
            sorted_users = sorted(
                related_users.items(), 
                key=lambda x: x[1] / math.sqrt(user_cnt[u] * user_cnt[x[0]]), 
                reverse=True
            )[:top_k]
            final_sim_matrix[u] = {k: v / math.sqrt(user_cnt[u] * user_cnt[k]) for k, v in sorted_users}
            
        self.user_sim_matrix = final_sim_matrix
        self.save()

    def retrieve(self, user_id: int, k=50):
        """
        根据相似用户召回物品。
        """
        if user_id not in self.user_sim_matrix:
            return []
            
        rank = defaultdict(float)
        interacted_items = set(self.user_item_dict.get(user_id, []))
        
        for v, sim in self.user_sim_matrix[user_id].items():
            for item in self.user_item_dict.get(v, []):
                if item in interacted_items:
                    continue
                rank[item] += sim
                
        sorted_res = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:k]
        return [res[0] for res in sorted_res]

    def save(self):
        self.sim_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sim_save_path, "wb") as f:
            data = {'matrix': self.user_sim_matrix, 'user_item': self.user_item_dict}
            pickle.dump(data, f)
        print(f"UserCF artifacts saved to {self.sim_save_path}")

    def load(self):
        with open(self.sim_save_path, "rb") as f:
            data = pickle.load(f)
            self.user_sim_matrix = data['matrix']
            self.user_item_dict = data['user_item']
        print("UserCF artifacts loaded.")
