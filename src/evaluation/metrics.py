import numpy as np

def recall_at_k(actual, predicted, k=50):
    """
    actual: 真实点击的 item_id (list or single id)
    predicted: 推荐的 item_id 列表
    """
    if not isinstance(actual, list):
        actual = [actual]
    
    predicted_k = predicted[:k]
    if not predicted_k:
        return 0.0
        
    hits = set(actual) & set(predicted_k)
    return len(hits) / len(actual)

def ndcg_at_k(actual, predicted, k=50):
    """
    actual: 真实点击的 item_id (single id)
    predicted: 推荐的 item_id 列表
    """
    if not isinstance(actual, (int, np.integer)):
        # For multi-item actual, simplified to first hit
        actual_list = actual if isinstance(actual, list) else [actual]
    else:
        actual_list = [actual]
        
    predicted_k = predicted[:k]
    for i, p in enumerate(predicted_k):
        if p in actual_list:
            return 1.0 / np.log2(i + 2)
    return 0.0

def hitrate_at_k(actual, predicted, k=10):
    """
    HitRate@K: 1 if any actual item appears in top-K, else 0.
    For leave-one-out evaluation (single target item).
    """
    if not isinstance(actual, list):
        actual = [actual]
    predicted_k = predicted[:k]
    return 1.0 if set(actual) & set(predicted_k) else 0.0

def mrr(actual, predicted, k=None):
    """
    Mean Reciprocal Rank: 1/rank of the first relevant item.
    If k is given, only consider top-K predictions.
    """
    if not isinstance(actual, list):
        actual = [actual]
    actual_set = set(actual)
    candidates = predicted[:k] if k else predicted
    for i, p in enumerate(candidates):
        if p in actual_set:
            return 1.0 / (i + 1)
    return 0.0

def auc_score(actual, scores):
    """
    Simplified AUC for ranking.
    """
    pass
