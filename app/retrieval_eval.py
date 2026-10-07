"""
Retrieval evaluation metrics module for deterministic measurement of:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)

Completely independent of LLM generation; runs locally without API keys.
"""

from typing import List, Dict, Any, Set


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Compute Recall@K: fraction of relevant documents that appear in the top K retrieved results.
    
    Args:
        retrieved_ids: Ordered list of retrieved chunk/document IDs
        relevant_ids: List of ground-truth relevant chunk/document IDs
        k: Cutoff rank
        
    Returns:
        Recall score in range [0.0, 1.0]
    """
    if not relevant_ids:
        return 1.0
    if k <= 0 or not retrieved_ids:
        return 0.0
    
    top_k_set: Set[str] = set(retrieved_ids[:k])
    relevant_set: Set[str] = set(relevant_ids)
    
    hits = len(top_k_set & relevant_set)
    return hits / len(relevant_set)


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Compute Precision@K: fraction of top K retrieved documents that are relevant.
    
    Args:
        retrieved_ids: Ordered list of retrieved chunk/document IDs
        relevant_ids: List of ground-truth relevant chunk/document IDs
        k: Cutoff rank
        
    Returns:
        Precision score in range [0.0, 1.0]
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    if not relevant_ids:
        return 0.0
    
    evaluated_ids = retrieved_ids[:k]
    relevant_set: Set[str] = set(relevant_ids)
    
    hits = sum(1 for cid in evaluated_ids if cid in relevant_set)
    return hits / len(evaluated_ids)


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Compute Reciprocal Rank: 1 / rank of the first relevant document in retrieved results.
    
    Args:
        retrieved_ids: Ordered list of retrieved chunk/document IDs
        relevant_ids: List of ground-truth relevant chunk/document IDs
        
    Returns:
        Reciprocal rank score in range [0.0, 1.0]
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    
    relevant_set: Set[str] = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_set:
            return 1.0 / rank
    
    return 0.0


def evaluate_retrieval_batch(eval_cases: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    """
    Evaluate a batch of retrieval cases and compute aggregate metrics.
    
    Each case in eval_cases must have:
      - 'retrieved_ids': List[str]
      - 'relevant_ids': List[str]
      
    Returns:
      Dict with 'mean_recall_at_k', 'mean_precision_at_k', 'mrr', 'cases_count'.
    """
    if not eval_cases:
        return {
            f"mean_recall@{k}": 0.0,
            f"mean_precision@{k}": 0.0,
            "mrr": 0.0,
            "cases_count": 0
        }
    
    recalls = [compute_recall_at_k(c["retrieved_ids"], c["relevant_ids"], k) for c in eval_cases]
    precisions = [compute_precision_at_k(c["retrieved_ids"], c["relevant_ids"], k) for c in eval_cases]
    mrrs = [compute_mrr(c["retrieved_ids"], c["relevant_ids"]) for c in eval_cases]
    mean_mrr = sum(mrrs) / len(mrrs)
    count = len(eval_cases)
    return {
        f"mean_recall@{k}": sum(recalls) / count,
        f"mean_precision@{k}": sum(precisions) / count,
        "mrr": mean_mrr,
        "mean_mrr": mean_mrr,
        "cases_count": count,
        "sample_count": count,
    }
