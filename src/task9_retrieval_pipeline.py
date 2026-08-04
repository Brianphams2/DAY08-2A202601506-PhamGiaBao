"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from concurrent.futures import ThreadPoolExecutor

from .task5_semantic_search import hyde_search, semantic_search
from .task6_lexical_search import lexical_search
from .task6_tfidf_search import tfidf_search
from .task7_reranking import rerank
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Corpus calibration (BGE-M3, 404 chunks): known in-domain questions generally
# score above 0.48, while nonsense probes score around 0.43-0.45.  The threshold
# is compared with the original cosine score, never the RRF score.
SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    retrieval_mode: str = "hybrid",
    lexical_method: str = "bm25",
    use_hyde: bool = False,
    use_pageindex: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
        ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
        ├→ Lexical Search  → sparse_results
        │
        ├→ Merge (RRF) → merged_results
        ├→ Rerank → reranked_results
        │
        └→ If dense_results[0]["score"] < threshold:
              └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng RRF reranking hay không
        retrieval_mode: ``hybrid`` (dense + lexical) hoặc ``dense``.
        lexical_method: ``bm25`` mặc định hoặc bonus strategy ``tfidf``.
        use_hyde: Dùng HyDE cho nhánh dense; tự fallback semantic khi LLM lỗi.
        use_pageindex: Cho phép PageIndex fallback. Evaluation có thể tắt để
            so sánh retrieval configs mà không phát sinh API call ngoài ý muốn.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    query = query.strip()
    if not query or top_k <= 0:
        return []
    if retrieval_mode not in {"hybrid", "dense"}:
        raise ValueError("retrieval_mode must be 'hybrid' or 'dense'")
    if lexical_method not in {"bm25", "tfidf"}:
        raise ValueError("lexical_method must be 'bm25' or 'tfidf'")

    dense_search = hyde_search if use_hyde else semantic_search
    lexical_searcher = lexical_search if lexical_method == "bm25" else tfidf_search

    # Step 1: Dense and BM25 are independent, so run them concurrently.
    if retrieval_mode == "hybrid":
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval") as pool:
            dense_future = pool.submit(dense_search, query, top_k * 2)
            sparse_future = pool.submit(lexical_searcher, query, top_k * 2)
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()
    else:
        dense_results = dense_search(query, top_k=top_k * 2)
        sparse_results = []

    # Preserve method-specific scores before RRF replaces the generic ``score``
    # field with its fusion score.  These diagnostics are useful to the UI:
    # cosine similarity is interpretable against SCORE_THRESHOLD, while RRF is
    # only a ranking signal and must not be presented as confidence.
    dense_diagnostics = {
        item["content"]: {"dense_score": float(item["score"]), "dense_rank": rank}
        for rank, item in enumerate(dense_results, 1)
        if item.get("content") and item.get("score") is not None
    }
    sparse_diagnostics = {}
    for rank, item in enumerate(sparse_results, 1):
        content = item.get("content")
        score = item.get("score")
        if not content or score is None:
            continue
        prefix = "bm25" if lexical_method == "bm25" else "tfidf"
        sparse_diagnostics[content] = {
            "lexical_method": lexical_method,
            "lexical_score": float(score),
            "lexical_rank": rank,
            f"{prefix}_score": float(score),
            f"{prefix}_rank": rank,
        }
    best_score = dense_results[0]["score"] if dense_results else 0.0

    # Steps 2-3: RRF is both the merge and reranking operation.  Calling RRF a
    # second time on its own output would only rewrite scores without adding
    # ranking evidence, so it is deliberately performed once here.
    if retrieval_mode == "hybrid" and use_reranking:
        final_results = rerank(
            query, [dense_results, sparse_results], top_k=top_k, method=RERANK_METHOD
        )
    elif retrieval_mode == "hybrid":
        final_results = []
        seen: set[str] = set()
        for item in [*dense_results, *sparse_results]:
            content = item.get("content", "")
            if content and content not in seen:
                seen.add(content)
                final_results.append(item.copy())
            if len(final_results) >= top_k:
                break
    else:
        final_results = [item.copy() for item in dense_results[:top_k]]

    for final_rank, item in enumerate(final_results, 1):
        content = item.get("content", "")
        if retrieval_mode == "hybrid" and use_reranking:
            item["rrf_score"] = float(item.get("score", 0.0))
            item["score_type"] = "rrf"
        elif content in dense_diagnostics:
            item["score_type"] = "cosine"
        else:
            item["score_type"] = lexical_method
        item.update(dense_diagnostics.get(content, {}))
        item.update(sparse_diagnostics.get(content, {}))
        item["lexical_method"] = lexical_method
        item["query_dense_score"] = float(best_score)
        item["final_rank"] = final_rank
        item["source"] = "hybrid"
        item.setdefault("metadata", {})["retrieval_mode"] = retrieval_mode
        item["metadata"]["hyde"] = bool(use_hyde)
        item["metadata"]["lexical_method"] = lexical_method

    # Step 4: Check threshold DÙNG ĐIỂM COSINE GỐC (dense_results), KHÔNG PHẢI RRF
    if use_pageindex and best_score < score_threshold:
        print(f"  [Fallback] Semantic score {best_score:.3f} < {score_threshold:.3f}")
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                for final_rank, item in enumerate(fallback, 1):
                    item["score_type"] = "pageindex_rank"
                    item["query_dense_score"] = float(best_score)
                    item["final_rank"] = final_rank
                return fallback
        except Exception as e:
            print(f"  [Fallback] PageIndex unavailable: {e}")

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
