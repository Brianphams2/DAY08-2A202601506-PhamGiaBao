"""Bonus lexical retrieval with TF-IDF over the exact Task 4 chunk corpus.

TF-IDF represents each chunk with term weights instead of BM25's probabilistic
term-saturation formula. The shared Vietnamese tokenizer folds accents, drops
common stopwords, and emits phrase bigrams, making the comparison with BM25
about the ranking formula rather than different text preprocessing.
"""

from __future__ import annotations

from typing import Any

from .task6_lexical_search import load_corpus, tokenize


CORPUS: list[dict] = []
_VECTORIZER: Any | None = None
_TFIDF_MATRIX: Any | None = None


def _searchable_text(document: dict) -> str:
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


def build_tfidf_index(corpus: list[dict]) -> tuple[Any, Any]:
    """Build an L2-normalized TF-IDF matrix for a validated chunk corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not corpus:
        raise ValueError("Cannot build TF-IDF index from an empty corpus")
    texts: list[str] = []
    for document in corpus:
        content = document.get("content")
        metadata = document.get("metadata")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each corpus document needs non-empty string content")
        if not isinstance(metadata, dict):
            raise ValueError("Each corpus document needs a metadata dictionary")
        texts.append(_searchable_text(document))

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        norm="l2",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def refresh_tfidf_index() -> int:
    """Reload the shared Chroma chunk corpus and rebuild the TF-IDF matrix."""
    global _VECTORIZER, _TFIDF_MATRIX

    corpus = load_corpus()
    CORPUS.clear()
    CORPUS.extend(corpus)
    if CORPUS:
        _VECTORIZER, _TFIDF_MATRIX = build_tfidf_index(CORPUS)
    else:
        _VECTORIZER, _TFIDF_MATRIX = None, None
    return len(CORPUS)


def _get_tfidf_index() -> tuple[Any | None, Any | None]:
    if _VECTORIZER is None or _TFIDF_MATRIX is None:
        refresh_tfidf_index()
    return _VECTORIZER, _TFIDF_MATRIX


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top chunks ranked by cosine similarity in TF-IDF space."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    query = query.strip()
    if not query or top_k <= 0 or not tokenize(query):
        return []

    vectorizer, matrix = _get_tfidf_index()
    if vectorizer is None or matrix is None or not CORPUS:
        return []

    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
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
    for result in tfidf_search("thời hạn hoàn tiền", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
