import pandas as pd
from collections import defaultdict

class RecallMerger:
    def __init__(self, top_k=100):
        self.top_k = top_k

    def merge(self, results_dict: dict, weights: dict = None):
        """
        results_dict: {
            'dual_tower': [item1, item2, ...],
            'item_cf': [item3, item1, ...],
            'user_cf': [...],
            'genre': [...],
            'pop': [...]
        }
        weights: { 'dual_tower': 1.0, 'item_cf': 0.8, ... } 可选权重
        """
        if not results_dict:
            return []

        # 默认权重全为 1.0
        if weights is None:
            weights = {k: 1.0 for k in results_dict.keys()}

        # 1. 加权分数融合逻辑 (Simple Version: Rank-based Weighting)
        # 如果没有原始分数，我们根据排名给分：第一名给 1.0, 第二名 0.99... 乘以通道权重
        final_scores = defaultdict(float)
        
        for channel, items in results_dict.items():
            channel_weight = weights.get(channel, 1.0)
            for i, item_id in enumerate(items):
                # 简单的排名倒数分 (RRR)
                rank_score = (1.0 / (i + 1)) * channel_weight
                final_scores[item_id] += rank_score
        
        # 2. 排序并截断
        sorted_res = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        merged_items = [res[0] for res in sorted_res]
        
        return merged_items[:self.top_k]

    def sequential_merge(self, results_dict: dict, priority_order: list):
        """
        按优先级顺序合并，直到填满 top_k。
        """
        final_list = []
        seen = set()
        
        for channel in priority_order:
            if channel not in results_dict:
                continue
            for item in results_dict[channel]:
                if item not in seen:
                    final_list.append(item)
                    seen.add(item)
                if len(final_list) >= self.top_k:
                    break
            if len(final_list) >= self.top_k:
                break
                
        return final_list
