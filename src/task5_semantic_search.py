"""Task 5 - Semantic search tren BGE-M3 va ChromaDB cua Task 4.

Module nay co hai duong tim kiem dung chung mot collection:
    - semantic_search(): embed thang cau hoi cua nguoi dung.
    - hyde_search():     HyDE - cho LLM sinh mot doan tra loi gia dinh truoc,
                         roi embed doan do de truy van (xem ghi chu HyDE ben duoi).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

from dotenv import load_dotenv

from .task4_chunking_indexing import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    get_collection,
    get_embedding_model,
)


load_dotenv()


# =============================================================================
# HyDE CONFIGURATION (Hypothetical Document Embeddings)
# =============================================================================

# Cau hoi nguoi dung thuong ngan va dung tu ngu hoi thoai ("cho doi hang duoc
# may ngay?"), trong khi chunk trong corpus la van phong chinh sach ("Nguoi mua
# co quyen gui yeu cau Tra hang/Hoan tien trong vong 15 ngay..."). Khoang cach
# van phong do keo cosine similarity xuong thap hon do lien quan thuc te.
# HyDE cho LLM sinh mot doan tra loi GIA DINH theo dung van phong tai lieu roi
# embed doan do, nen vector truy van roi vao dung vung khong gian cua corpus.
HYDE_MODEL = os.getenv("HYDE_MODEL", "openai/gpt-4o-mini")
HYDE_BASE_URL = "https://openrouter.ai/api/v1"
HYDE_MAX_TOKENS = 220
HYDE_TEMPERATURE = 0.3
HYDE_TIMEOUT = 30.0
HYDE_CACHE_SIZE = 128

# Ghep query goc vao truoc doan gia dinh thay vi embed rieng doan gia dinh.
# Doi lai: neu LLM sinh sai chi tiet (vd sai so ngay, sai ten chinh sach), query
# goc van neo vector ve dung y dinh ban dau. Embed rieng doan gia dinh la HyDE
# nguyen ban nhung troi theo loi cua LLM khi model sinh bay.
HYDE_INCLUDE_QUERY = True

HYDE_PROMPT = (
    "Ban la chuyen vien soan thao chinh sach thuong mai dien tu.\n"
    "Viet MOT doan van ngan (3-4 cau) bang tieng Viet, theo van phong tai lieu "
    "chinh sach chinh thuc, tra loi cau hoi ben duoi nhu the doan van do duoc "
    "trich tu trang chinh sach that.\n"
    "Chi viet noi dung doan van: khong mo dau, khong giai thich, khong trich dan nguon.\n"
    "Neu khong chac chan ve chi tiet, cu dien dat theo cach mot tai lieu chinh sach "
    "thong thuong se viet.\n\n"
    "Cau hoi: {query}"
)


# =============================================================================
# COLLECTION VALIDATION
# =============================================================================

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


# =============================================================================
# CORE RETRIEVAL
# =============================================================================

def _search_by_text(text: str, top_k: int) -> list[dict]:
    """
    Embed `text` va tra ve top_k chunk gan nhat theo cosine similarity.

    Dung chung cho semantic_search (embed query goc) va hyde_search (embed doan
    gia dinh), nen ca hai duong deu cho ra cung thang diem va cung cach sort.
    """
    collection, collection_count = _validate_collection()
    if collection_count == 0:
        return []

    model = get_embedding_model()
    encoded = model.encode(
        [text],
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

    return _search_by_text(query, top_k)


# =============================================================================
# HyDE
# =============================================================================

def _get_llm_api_key() -> str:
    """Lay API key that; tra chuoi rong neu .env con la placeholder."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    api_key = api_key.strip()
    if "..." in api_key:  # vd placeholder 'sk-or-v1-...' trong .env.example
        return ""
    return api_key


@lru_cache(maxsize=HYDE_CACHE_SIZE)
def generate_hypothetical_document(query: str) -> str:
    """
    Goi LLM sinh mot doan tra loi gia dinh cho query.

    Returns:
        Doan van gia dinh, hoac chuoi rong neu khong co API key hop le / LLM loi.
        Chuoi rong la tin hieu cho hyde_search fallback ve semantic search thuong,
        de pipeline khong bao gio chet chi vi LLM khong san sang.

    Ket qua duoc cache theo query: RAGAS eval chay lai nhieu lan tren cung bo
    golden dataset se khong dot them quota OpenRouter.
    """
    api_key = _get_llm_api_key()
    if not api_key:
        return ""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=HYDE_BASE_URL, timeout=HYDE_TIMEOUT)
        response = client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            temperature=HYDE_TEMPERATURE,
            max_tokens=HYDE_MAX_TOKENS,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # rate limit, mat mang, model id sai, ...
        print(f"  [HyDE] LLM khong dung duoc, dung semantic search thuong: {exc}")
        return ""


def hyde_search(
    query: str,
    top_k: int = 10,
    include_query: bool = HYDE_INCLUDE_QUERY,
) -> list[dict]:
    """
    Semantic search co HyDE (Hypothetical Document Embeddings).

    Luong:
        1. LLM sinh doan tra loi gia dinh theo van phong tai lieu chinh sach.
        2. Embed (query + doan gia dinh) thay vi chi embed query.
        3. Truy van ChromaDB nhu semantic_search binh thuong.

    Neu khong co API key hoac LLM loi -> tu dong fallback ve semantic_search,
    nen ham nay luon an toan de goi tu Task 9.

    Args:
        query: Cau truy van.
        top_k: So ket qua toi da.
        include_query: True thi ghep query goc vao truoc doan gia dinh (mac dinh),
            False thi embed rieng doan gia dinh dung nhu HyDE nguyen ban.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict,
                 'retrieval_method': 'hyde' | 'semantic'}.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")

    query = query.strip()
    if not query or top_k <= 0:
        return []

    hypothetical = generate_hypothetical_document(query)
    if not hypothetical:
        results = _search_by_text(query, top_k)
        for item in results:
            item["retrieval_method"] = "semantic"
        return results

    search_text = f"{query}\n{hypothetical}" if include_query else hypothetical
    results = _search_by_text(search_text, top_k)
    for item in results:
        item["retrieval_method"] = "hyde"
    return results


# =============================================================================
# DEMO - so sanh semantic search thuong vs HyDE
# =============================================================================

def _print_results(results: list[dict]) -> None:
    if not results:
        print("  (khong co ket qua)")
        return
    for result in results:
        source = result["metadata"].get("source", "unknown")
        preview = result["content"][:100].replace("\n", " ")
        print(f"  [{result['score']:.4f}] {source}: {preview}...")


if __name__ == "__main__":
    demo_query = "cho doi hang bao lau thi duoc tra lai"

    print("=" * 70)
    print(f"Query: {demo_query}")
    print("=" * 70)

    print("\n--- Semantic Search (embed query goc) ---")
    _print_results(semantic_search(demo_query, top_k=5))

    print("\n--- HyDE Search (embed query + doan gia dinh) ---")
    hypothetical = generate_hypothetical_document(demo_query)
    if hypothetical:
        print(f"\nDoan gia dinh do LLM sinh:\n{hypothetical}\n")
    else:
        print("\nKhong co API key hop le -> fallback semantic search thuong.\n")
    _print_results(hyde_search(demo_query, top_k=5))
