"""Task 5 - Semantic search tren BGE-M3 va ChromaDB cua Task 4."""

from __future__ import annotations

import math

from .task4_chunking_indexing import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    get_collection,
    get_embedding_model,
)


def _validate_collection() -> tuple[object, int]:
    """Mo collection va xac minh no tuong thich voi embedding model hien tai."""
    collection = get_collection()
    metadata = collection.metadata or {}

    indexed_model = metadata.get("embedding_model")
    indexed_dimension = metadata.get("embedding_dimension")
    distance_space = metadata.get("hnsw:space")
    if indexed_model not in (None, EMBEDDING_MODEL):
        raise RuntimeError(
            f"Chroma collection uses {indexed_model}, expected {EMBEDDING_MODEL}. Re-run Task 4."
        )
    if indexed_dimension not in (None, EMBEDDING_DIM):
        raise RuntimeError(
            f"Chroma collection dimension is {indexed_dimension}, expected {EMBEDDING_DIM}. "
            "Re-run Task 4."
        )
    if distance_space not in (None, "cosine"):
        raise RuntimeError(
            f"Chroma collection uses '{distance_space}' distance, expected cosine. Re-run Task 4."
        )
    return collection, collection.count()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tim kiem ngu nghia bang cosine similarity.

    Args:
        query: Cau truy van tieng Viet hoac tieng Anh.
        top_k: So ket qua toi da.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sap xep
        theo cosine similarity giam dan. Score goc nay duoc giu rieng de Task 9
        dung khi quyet dinh fallback, khong thay bang diem RRF.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")

    query = query.strip()
    if not query or top_k <= 0:
        return []

    collection, collection_count = _validate_collection()
    if collection_count == 0:
        return []

    model = get_embedding_model()
    encoded = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if encoded.ndim != 2 or encoded.shape != (1, EMBEDDING_DIM):
        raise RuntimeError(
            f"Unexpected query embedding shape: expected (1, {EMBEDDING_DIM}), got {encoded.shape}"
        )

    n_results = min(top_k, collection_count)
    response = collection.query(
        query_embeddings=encoded.tolist(),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    if not (len(documents) == len(metadatas) == len(distances)):
        raise RuntimeError("Chroma returned misaligned documents, metadatas and distances")

    output: list[dict] = []
    for content, metadata, distance in zip(documents, metadatas, distances, strict=True):
        distance_value = float(distance)
        if not math.isfinite(distance_value):
            continue

        # Chroma cosine distance = 1 - cosine similarity. Corpus va query deu
        # L2-normalized; clamp ve [0, 1] de threshold Task 9 de dien giai.
        score = max(0.0, min(1.0, 1.0 - distance_value))
        output.append(
            {
                "content": str(content),
                "score": float(score),
                "metadata": dict(metadata or {}),
            }
        )

    output.sort(
        key=lambda item: (
            -item["score"],
            str(item["metadata"].get("chunk_id", "")),
        )
    )
    return output[:top_k]


if __name__ == "__main__":
    for result in semantic_search("quy dinh tra hang hoan tien Shopee", top_k=5):
        source = result["metadata"].get("source", "unknown")
        print(f"[{result['score']:.4f}] {source}: {result['content'][:100]}...")
