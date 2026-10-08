"""
Comprehensive evaluation metrics module for Aura AI Large-Scale Benchmark.
Provides:
1. Retrieval Metrics: Recall@K, Precision@K, MRR, Hit Rate, nDCG@K
2. Answer Metrics: Exact Match (EM), Normalized EM, Token F1, Numeric Accuracy, Date Accuracy
3. Citation Metrics: Citation Validity, Citation Correctness, Citation Completeness
4. Missing Information Metrics: Missing Info Detection Rate, Hallucination Rate
5. Latency Metrics: Min, Mean, Median (p50), p90, p95, p99
6. LLM Judge Interface: Correctness, Completeness, Faithfulness, Relevance (0-4 scale)
"""

import re
import math
import string
from typing import List, Dict, Any, Set, Optional, Tuple


# =============================================================================
# 1. Retrieval Metrics
# =============================================================================

def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant items retrieved in top-K."""
    if not relevant_ids:
        return 1.0
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k_set & relevant_set)
    return hits / len(relevant_set)


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    if not relevant_ids:
        return 0.0
    evaluated = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for item in evaluated if item in relevant_set)
    return hits / len(evaluated)


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first relevant retrieved item."""
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, 1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """1.0 if at least one relevant item is in top-K, else 0.0."""
    if not relevant_ids:
        return 1.0
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    return 1.0 if (top_k_set & set(relevant_ids)) else 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance, deduplicated by entity)."""
    if not relevant_ids:
        return 1.0
    if k <= 0 or not retrieved_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    seen_relevant = set()
    dcg = 0.0
    for rank, item in enumerate(retrieved_ids[:k], 1):
        if item in relevant_set and item not in seen_relevant:
            dcg += 1.0 / math.log2(rank + 1)
            seen_relevant.add(item)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))
    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


# =============================================================================
# 2. Deterministic Answer Metrics
# =============================================================================

def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    if not s:
        return ""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Strict exact match."""
    return 1.0 if prediction.strip() == ground_truth.strip() else 0.0


def compute_normalized_exact_match(prediction: str, ground_truth: str) -> float:
    """Normalized exact match after stripping punctuation and whitespace."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def extract_numbers(text: str) -> List[float]:
    """Extract numeric values from text (including decimals and percentages)."""
    # Remove commas in numbers
    clean = text.replace(",", "")
    matches = re.findall(r'[-+]?\d*\.?\d+', clean)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers


def compute_numeric_accuracy(prediction: str, ground_truth: str, tolerance: float = 0.02) -> float:
    """
    Check if all ground truth numbers appear in prediction within a percentage tolerance.
    """
    gt_nums = extract_numbers(ground_truth)
    if not gt_nums:
        return 1.0  # Not a numeric question

    pred_nums = extract_numbers(prediction)
    if not pred_nums:
        return 0.0

    matched = 0
    for gn in gt_nums:
        found = False
        for pn in pred_nums:
            diff = abs(pn - gn)
            if gn == 0:
                if diff < 1e-5:
                    found = True
                    break
            elif (diff / abs(gn)) <= tolerance:
                found = True
                break
        if found:
            matched += 1

    return matched / len(gt_nums)


# =============================================================================
# 3. Citation Metrics
# =============================================================================

def extract_citations(text: str) -> List[str]:
    """Extract all citation markers like [S1], [S2] or [doc_1.txt, p. 2]."""
    return re.findall(r'\[(?:S\d+|[^\]]+)\]', text)


def compute_citation_validity(answer: str, valid_citation_markers: Set[str]) -> float:
    """
    Measures what fraction of cited markers in the answer correspond to actual retrieved sources.
    """
    cited = extract_citations(answer)
    if not cited:
        return 1.0 if not valid_citation_markers else 0.0
    valid_count = sum(1 for c in cited if c in valid_citation_markers)
    return valid_count / len(cited)


def compute_citation_completeness(answer: str, key_facts: List[str]) -> float:
    """
    Measures whether statements containing key facts are accompanied by at least one citation.
    """
    if not key_facts:
        return 1.0
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    facts_with_citation = 0
    for fact in key_facts:
        fact_found = False
        fact_cited = False
        for s in sentences:
            if fact.lower() in s.lower():
                fact_found = True
                if extract_citations(s):
                    fact_cited = True
                    break
        if fact_found and fact_cited:
            facts_with_citation += 1
        elif not fact_found:
            # Fact was not even generated
            pass

    return facts_with_citation / len(key_facts)


# =============================================================================
# 4. Missing Information Metrics
# =============================================================================

MISSING_INFO_INDICATORS = [
    "not mentioned", "not found", "insufficient information",
    "insufficient evidence", "does not contain", "cannot be determined",
    "no information", "not provided", "not available", "no data"
]

def evaluate_missing_information(answer: str, answerable: bool, prohibited_facts: List[str] = None) -> Dict[str, float]:
    """
    Evaluates unanswerable / missing-information queries.
    Returns detection score (1.0 if properly refused/flagged missing) and hallucination score.
    """
    ans_lower = answer.lower()
    indicates_missing = any(ind in ans_lower for ind in MISSING_INFO_INDICATORS)

    # Check for hallucination of prohibited facts
    hallucinated = False
    if prohibited_facts:
        for pf in prohibited_facts:
            if pf.lower() in ans_lower:
                hallucinated = True
                break

    if not answerable:
        # Question was unanswerable:
        # Success = detected missing info AND did not hallucinate
        detected = 1.0 if indicates_missing else 0.0
        hallucination_rate = 1.0 if (hallucinated or not indicates_missing) else 0.0
    else:
        # Question was answerable:
        # Refusing is a false negative (missed answer)
        detected = 0.0 if indicates_missing else 1.0
        hallucination_rate = 1.0 if hallucinated else 0.0

    return {
        "missing_info_detected": detected,
        "hallucination_rate": hallucination_rate
    }


# =============================================================================
# 5. Latency Percentiles
# =============================================================================

def compute_latency_summary(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculate min, mean, median, p90, p95, p99 from a list of latencies in ms."""
    if not latencies_ms:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_lat[int(k)]
        d0 = sorted_lat[int(f)] * (c - k)
        d1 = sorted_lat[int(c)] * (k - f)
        return d0 + d1

    return {
        "min": round(sorted_lat[0], 2),
        "mean": round(sum(sorted_lat) / n, 2),
        "median": round(percentile(0.50), 2),
        "p90": round(percentile(0.90), 2),
        "p95": round(percentile(0.95), 2),
        "p99": round(percentile(0.99), 2)
    }


# =============================================================================
# 6. LLM Judge Interface
# =============================================================================

class LLMJudge:
    """
    Interface for LLM-as-a-judge evaluation scoring on a 0-4 scale:
    - Correctness: Is the answer factually aligned with ground truth?
    - Completeness: Does the answer address all parts of the question?
    - Faithfulness: Is every claim strictly supported by retrieved context?
    - Relevance: Is the answer directly responsive to the query?
    """

    JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator assessing the quality of a RAG assistant's response.
Evaluate the candidate answer based on the question, retrieved context, and reference ground truth.

Question: {question}
Retrieved Context: {context}
Reference Answer: {reference_answer}
Candidate Answer: {candidate_answer}

Rate each of the following dimensions on a strict scale from 0 to 4:
- Correctness (0=completely wrong, 4=completely accurate according to ground truth)
- Completeness (0=omits all key info, 4=fully comprehensive answer)
- Faithfulness (0=entirely fabricated/unsupported by context, 4=strictly grounded in context)
- Relevance (0=unrelated to question, 4=perfectly addresses the question)

Return ONLY valid JSON in the format:
{{
  "correctness": <0-4>,
  "completeness": <0-4>,
  "faithfulness": <0-4>,
  "relevance": <0-4>,
  "rationale": "<brief explanation>"
}}
"""

    @classmethod
    def format_prompt(cls, question: str, context: str, reference_answer: str, candidate_answer: str) -> str:
        return cls.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            context=context or "(No context retrieved)",
            reference_answer=reference_answer or "(Unanswerable question)",
            candidate_answer=candidate_answer
        )

    @classmethod
    def evaluate_rule_based_fallback(
        cls,
        question: str,
        candidate_answer: str,
        reference_answer: Optional[str],
        key_facts: List[str]
    ) -> Dict[str, Any]:
        """
        Deterministic, offline scoring proxy when an LLM API key is not available.
        Provides continuous 0-4 scores based on token overlap, key fact presence, and answerability.
        """
        if reference_answer is None:
            # Unanswerable case
            ans_lower = candidate_answer.lower()
            refused = any(ind in ans_lower for ind in MISSING_INFO_INDICATORS)
            score = 4.0 if refused else 0.5
            return {
                "correctness": score,
                "completeness": score,
                "faithfulness": score,
                "relevance": 4.0 if refused else 2.0,
                "rationale": "Rule-based fallback evaluation for unanswerable query."
            }

        # Answerable case
        f1 = compute_token_f1(candidate_answer, reference_answer)
        fact_coverage = 0.0
        if key_facts:
            hits = sum(1 for kf in key_facts if kf.lower() in candidate_answer.lower())
            fact_coverage = hits / len(key_facts)
        else:
            fact_coverage = f1

        correctness = round(4.0 * ((f1 * 0.4) + (fact_coverage * 0.6)), 2)
        completeness = round(4.0 * fact_coverage, 2)
        faithfulness = round(4.0 * min(1.0, f1 + 0.2), 2)
        relevance = 4.0 if len(candidate_answer.strip()) > 10 else 1.0

        return {
            "correctness": correctness,
            "completeness": completeness,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "rationale": f"Deterministic rule-based fallback: F1={f1:.2f}, FactCoverage={fact_coverage:.2f}."
        }
