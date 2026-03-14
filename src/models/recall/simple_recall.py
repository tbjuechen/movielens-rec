import pandas as pd
import json
from pathlib import Path

class PopularityRecall:
    def __init__(self, item_profile_path: str):
        item_profile = pd.read_parquet(item_profile_path)
        # 预先排好序
        self.top_items = item_profile.sort_values('vote_count_ml', ascending=False)['movieId'].tolist()

    def retrieve(self, k=100):
        return self.top_items[:k]

class GenreRecall:
    def __init__(self, genre_to_items_path: str, item_profile_path: str):
        with open(genre_to_items_path, "r") as f:
            self.genre_to_items = json.load(f)
        
        # 为了在同类型里选最好的，我们需要物品热度表
        item_profile = pd.read_parquet(item_profile_path)
        self.item_score = item_profile.set_index('movieId')['vote_count_ml'].to_dict()

    def retrieve(self, user_top_genres: list, k=100):
        """
        user_top_genres: 用户最喜欢的类型列表 ['Action', 'Sci-Fi']
        """
        candidates = []
        if not user_top_genres:
            return []
            
        # 每个类型均分 K 个名额
        per_genre_k = max(1, k // len(user_top_genres))
        
        for genre in user_top_genres:
            if genre not in self.genre_to_items:
                continue
            # 取出该类型下的所有物品，按热度排序
            genre_items = self.genre_to_items[genre]
            sorted_items = sorted(genre_items, key=lambda x: self.item_score.get(x, 0), reverse=True)
            candidates.extend(sorted_items[:per_genre_k])
            
        return list(dict.fromkeys(candidates))[:k] # 去重
