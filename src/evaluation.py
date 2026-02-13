import pandas as pd
from typing import List, Set, Dict
from tqdm import tqdm
from loguru import logger

def calculate_hit_rate(preds: List[List[int]], ground_truth: List[Set[int]]) -> float:
    """
    计算 Hit Rate。
    preds: 召回的列表的列表 [[mid1, mid2, ...], [user2_mids], ...]
    ground_truth: 用户实际看过的电影集合列表 [{mid_a, mid_b}, {mid_c}, ...]
    """
    hits = 0
    total = len(ground_truth)
    
    for p, gt in zip(preds, ground_truth):
        # 只要召回列表中有一个电影在用户实际看过的集合里，就算命中
        if any(mid in gt for mid in p):
            hits += 1
            
    return hits / total if total > 0 else 0.0
