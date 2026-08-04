"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

from __future__ import annotations

import os
from typing import Any, Optional
import numpy as np


def _cosine_sim(v1: list[float] | np.ndarray, v2: list[float] | np.ndarray) -> float:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker API hoặc local model).

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates or top_k <= 0:
        return []

    jina_api_key = os.getenv("JINA_API_KEY")
    if jina_api_key:
        try:
            import requests

            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {jina_api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c.get("content", "") for c in candidates],
                    "top_n": top_k,
                },
                timeout=10,
            )
            response.raise_for_status()
            reranked = response.json().get("results", [])
            results = []
            for r in reranked:
                idx = r["index"]
                item = candidates[idx].copy()
                item["score"] = float(r["relevance_score"])
                results.append(item)
            return results
        except Exception as e:
            print(f"Warning: Jina Reranker API failed ({e}). Falling back to score sorting.")

    # Fallback if no Jina API key or request failed: sort candidates by score descending
    sorted_candidates = sorted(
        candidates, key=lambda x: x.get("score", 0.0), reverse=True
    )
    return [c.copy() for c in sorted_candidates[:top_k]]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates or top_k <= 0:
        return []

    selected_indices: list[int] = []
    remaining_indices: list[int] = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_mmr_score = float("-inf")

        for idx in remaining_indices:
            cand = candidates[idx]
            cand_emb = cand.get("embedding")

            if query_embedding and cand_emb:
                relevance = _cosine_sim(query_embedding, cand_emb)
            else:
                relevance = float(cand.get("score", 0.0))

            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sel_cand = candidates[sel_idx]
                sel_emb = sel_cand.get("embedding")

                if cand_emb and sel_emb:
                    sim = _cosine_sim(cand_emb, sel_emb)
                else:
                    tokens1 = set(cand.get("content", "").lower().split())
                    tokens2 = set(sel_cand.get("content", "").lower().split())
                    union = tokens1 | tokens2
                    sim = len(tokens1 & tokens2) / len(union) if union else 0.0

                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    results = []
    for idx in selected_indices:
        results.append(candidates[idx].copy())

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict] | list[list[dict]],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval (hoặc list of lists từ nhiều searcher)
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking ("cross_encoder" | "mmr" | "rrf")

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates or top_k <= 0:
        return []

    if method == "cross_encoder":
        if isinstance(candidates[0], list):
            flat = [item for sublist in candidates for item in sublist]  # type: ignore
            return rerank_cross_encoder(query, flat, top_k)
        return rerank_cross_encoder(query, candidates, top_k)  # type: ignore
    elif method == "mmr":
        try:
            from .task4_chunking_indexing import get_embedding_model

            model = get_embedding_model()
            query_emb = model.encode(
                [query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
            )[0].tolist()
        except Exception:
            query_emb = []

        if isinstance(candidates[0], list):
            flat = [item for sublist in candidates for item in sublist]  # type: ignore
            return rerank_mmr(query_emb, flat, top_k)
        return rerank_mmr(query_emb, candidates, top_k)  # type: ignore
    elif method == "rrf":
        if not candidates:
            return []
        if isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k)
        else:
            return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")

