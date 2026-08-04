"""
Task 6 - Lexical search bang BM25.

BM25 ket hop term frequency, inverse document frequency va document-length
normalization. Cau hinh k1=1.5 dat muc term saturation can bang; b=0.75 la muc
normalization do dai pho bien. Corpus dung dung cac chunk cua Task 4 de ket qua
co the merge truc tiep voi semantic search bang chunk_id o Task 7.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .task4_chunking_indexing import chunk_documents, get_collection, load_documents


BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")

# Khoi tao lazy de import module nhanh va khong mo ChromaDB cho den query dau tien.
CORPUS: list[dict] = []
_BM25_INDEX: Any | None = None


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode va fold dau de query tieng Viet co/khong dau deu match."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    without_marks = without_marks.replace("đ", "d")
    return _TOKEN_PATTERN.findall(without_marks)


def _searchable_text(document: dict) -> str:
    """Index content cung metadata ngan co ich cho keyword retrieval."""
    content = str(document.get("content", ""))
    metadata = document.get("metadata") or {}
    metadata_text = " ".join(
        str(metadata.get(key, ""))
        for key in (
            "source",
            "filename",
            "title",
            "category",
            "source_type",
            "type",
            "platform",
        )
    )
    return f"{content}\n{metadata_text}"


def load_corpus() -> list[dict]:
    """Doc dung chunks da index trong Chroma; fallback chunk truc tiep neu rong."""
    collection = get_collection()
    if collection.count() > 0:
        response = collection.get(include=["documents", "metadatas"])
        ids = response.get("ids") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        if not (len(ids) == len(documents) == len(metadatas)):
            raise RuntimeError("Chroma returned misaligned corpus fields")

        rows = sorted(
            zip(ids, documents, metadatas, strict=True),
            key=lambda row: str(row[0]),
        )
        return [
            {"content": str(content), "metadata": dict(metadata or {})}
            for _, content, metadata in rows
            if str(content).strip()
        ]

    # Cho phep unit-test/chay Task 6 truoc khi embedding, nhung van dung chinh
    # strategy va chunk IDs cua Task 4.
    return chunk_documents(load_documents())


def build_bm25_index(corpus: list[dict]) -> Any:
    """
    Xay BM25Okapi index tu list {'content': str, 'metadata': dict}.

    Metadata source/title/category duoc dua vao text index de query bang ten file
    hoac nhan danh muc van tim dung chunk, nhung output van tra noi dung goc.
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Cannot build BM25 index from an empty corpus")

    tokenized_corpus: list[list[str]] = []
    for document in corpus:
        content = document.get("content")
        metadata = document.get("metadata")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each corpus document needs non-empty string content")
        if not isinstance(metadata, dict):
            raise ValueError("Each corpus document needs a metadata dictionary")

        tokens = tokenize(_searchable_text(document))
        if not tokens:
            raise ValueError("A corpus document produced no BM25 tokens")
        tokenized_corpus.append(tokens)

    return BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)


def refresh_bm25_index() -> int:
    """Nap lai corpus va BM25 index sau khi Task 4 duoc reindex."""
    global _BM25_INDEX

    corpus = load_corpus()
    CORPUS.clear()
    CORPUS.extend(corpus)
    _BM25_INDEX = build_bm25_index(CORPUS) if CORPUS else None
    return len(CORPUS)


def _get_bm25_index() -> Any | None:
    global _BM25_INDEX
    if _BM25_INDEX is None:
        refresh_bm25_index()
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tim kiem tu khoa bang BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted theo
        BM25 score giam dan. Ket qua diem <= 0 bi loai de query lac de co the
        tra danh sach rong va kich hoat fallback trong pipeline sau nay.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if not query.strip() or top_k <= 0:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    bm25 = _get_bm25_index()
    if bm25 is None or not CORPUS:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        (
            (float(score), index)
            for index, score in enumerate(scores)
            if float(score) > 0.0
        ),
        key=lambda item: (
            -item[0],
            str(CORPUS[item[1]]["metadata"].get("chunk_id", "")),
        ),
    )[:top_k]

    return [
        {
            "content": CORPUS[index]["content"],
            "score": score,
            "metadata": dict(CORPUS[index]["metadata"]),
        }
        for score, index in ranked
    ]


if __name__ == "__main__":
    for result in lexical_search("phuong thuc thanh toan Shopee", top_k=5):
        source = result["metadata"].get("source", "unknown")
        print(f"[{result['score']:.4f}] {source}: {result['content'][:100]}...")
