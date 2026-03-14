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

def auc_score(actual, scores):
    """
    Simplified AUC for ranking.
    """
    pass
