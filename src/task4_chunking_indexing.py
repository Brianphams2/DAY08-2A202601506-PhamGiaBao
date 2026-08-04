"""
Task 4 - Chunking va indexing vao ChromaDB.

Pipeline:
    1. Doc toan bo Markdown trong data/standardized/.
    2. Chia doan bang RecursiveCharacterTextSplitter.
    3. Tao embedding chuan hoa bang BAAI/bge-m3.
    4. Luu document, embedding va metadata vao ChromaDB local.

Khi reindex, code chi thay the collection COLLECTION_NAME. Cach nay ngan chunk
cu cua corpus truoc bi tron voi corpus hien tai ma khong xoa ca thu muc ChromaDB.

Implementation nay chon provider local sentence-transformers voi BAAI/bge-m3.
Neu doi sang Google/OpenAI, Task 4 va Task 5 phai dung chung ham embedding va
phai reindex collection vi dimension 1024/768/1536 khong tuong thich voi nhau.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Recursive splitting duoc chon vi corpus gom ca policy PDF chuyen doi (heading
# khong dong deu) va bai support Markdown. Thu tu separator uu tien heading,
# paragraph va cau truoc khi buoc phai tach theo khoang trang/ky tu.
# 800 ky tu giu du ngu canh cho mot muc policy ngan; overlap 120 (~15%) giup cac
# khai niem nam o bien chunk khong bi mat ma khong tao qua nhieu noi dung lap.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"
CHUNK_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", "; ", ", ", " ", ""]

# BGE-M3 duoc chon vi ho tro da ngon ngu, dac biet phu hop corpus tieng Viet,
# chay local khong can API key va tra ve vector 1024 chieu. Embedding duoc L2
# normalize de cosine similarity o Task 5 co thang do on dinh.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 8

# ChromaDB la vector store local mac dinh cua lab, persistence truc tiep va
# khong can Docker. Collection su dung cosine distance.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"
CHROMA_WRITE_BATCH_SIZE = 128


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

_METADATA_LINE = re.compile(r"^\*\*(?P<label>[^*]+):\*\*\s*(?P<value>.+?)\s*$")
_METADATA_KEYS = {
    "Source": "source_url",
    "Platform": "platform",
    "Category": "category",
    "Source type": "source_type",
    "Customer role": "customer_role",
    "Language": "language",
    "Effective date": "effective_date",
    "Source updated": "source_last_updated",
    "Crawled": "crawled_at",
    "Captured": "captured_at",
}


def _read_markdown_metadata(content: str) -> dict[str, str]:
    """Doc title va metadata header do Task 3 tao ra."""
    parsed: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and "title" not in parsed:
            parsed["title"] = stripped[2:].strip()
            continue

        match = _METADATA_LINE.match(stripped)
        if match:
            key = _METADATA_KEYS.get(match.group("label"))
            if key:
                parsed[key] = match.group("value").strip()
        elif stripped == "---":
            break
    return parsed


def _document_year(metadata: dict[str, str]) -> int | None:
    """Lay nam tot nhat co san de Task 10 tao citation [Nguon, Nam]."""
    for key in ("effective_date", "source_last_updated", "crawled_at", "captured_at"):
        value = metadata.get(key, "")
        if len(value) >= 4 and value[:4].isdigit():
            return int(value[:4])
    return None


def load_documents() -> list[dict]:
    """
    Doc toan bo Markdown trong data/standardized/ theo thu tu on dinh.

    Returns:
        List of {'content': str, 'metadata': dict}. Metadata chua source,
        document type, title, URL nguon va cac truong truy vet neu co.
    """
    if not STANDARDIZED_DIR.is_dir():
        raise FileNotFoundError(f"Standardized directory not found: {STANDARDIZED_DIR}")

    documents: list[dict] = []
    for markdown_path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not markdown_path.is_file():
            continue

        content = markdown_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Standardized document is empty: {markdown_path}")

        relative_path = markdown_path.relative_to(STANDARDIZED_DIR)
        relative_source = relative_path.as_posix()
        parsed = _read_markdown_metadata(content)
        document_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"

        metadata: dict[str, str | int | bool | float] = {
            "source": relative_source,
            "filename": markdown_path.name,
            "type": document_type,
            "title": parsed.get("title", markdown_path.stem.replace("-", " ").title()),
            "document_id": hashlib.sha256(relative_source.encode("utf-8")).hexdigest()[:16],
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        for key, value in parsed.items():
            if key != "title" and value:
                metadata[key] = value

        year = _document_year(parsed)
        if year is not None:
            metadata["year"] = year

        documents.append({"content": content, "metadata": metadata})

    return documents


# =============================================================================
# CHUNKING AND EMBEDDING
# =============================================================================

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bang RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}; moi metadata co chunk_index,
        chunk_count va chunk_id on dinh.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        keep_separator=True,
        strip_whitespace=True,
    )

    chunks: list[dict] = []
    for document in documents:
        content = document.get("content")
        metadata = document.get("metadata")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each document must contain non-empty string content")
        if not isinstance(metadata, dict):
            raise ValueError("Each document must contain a metadata dictionary")

        splits = [text for text in splitter.split_text(content) if text.strip()]
        source = str(metadata.get("source", "unknown"))
        for chunk_index, chunk_text in enumerate(splits):
            chunk_digest = hashlib.sha256(
                f"{source}\0{chunk_index}\0{chunk_text}".encode("utf-8")
            ).hexdigest()
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "chunk_count": len(splits),
                "chunk_id": chunk_digest,
            }
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})

    return chunks


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load BGE-M3 mot lan va xac minh dimension dung cau hinh."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    if hasattr(model, "get_embedding_dimension"):
        dimension = model.get_embedding_dimension()
    else:  # Compatibility with sentence-transformers 2.x.
        dimension = model.get_sentence_embedding_dimension()
    if dimension != EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected embedding dimension for {EMBEDDING_MODEL}: "
            f"expected {EMBEDDING_DIM}, got {dimension}"
        )
    return model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toan bo chunks bang BAAI/bge-m3 va L2-normalize vector.

    Returns:
        Ban sao moi chunk co them key 'embedding': list[float].
    """
    if not chunks:
        return []

    texts: list[str] = []
    for chunk in chunks:
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each chunk must contain non-empty string content")
        texts.append(content)

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if embeddings.ndim != 2 or embeddings.shape != (len(chunks), EMBEDDING_DIM):
        raise RuntimeError(
            f"Unexpected embedding matrix shape: expected "
            f"({len(chunks)}, {EMBEDDING_DIM}), got {embeddings.shape}"
        )

    embedded_chunks: list[dict] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        embedded_chunks.append({**chunk, "embedding": embedding.tolist()})
    return embedded_chunks


# =============================================================================
# CHROMADB INDEX
# =============================================================================

def _get_chroma_client() -> Any:
    """Tao local persistent Chroma client, tat anonymous telemetry."""
    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_metadata() -> dict[str, str | int | bool | float]:
    return {
        "hnsw:space": "cosine",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "chunking_method": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def get_collection() -> Any:
    """Lay collection cho Task 5; tao rong neu Task 4 chua index."""
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=_collection_metadata(),
    )


def index_to_vectorstore(chunks: list[dict]) -> Any:
    """
    Ghi chunks vao ChromaDB va thay the dung collection hien tai.

    Collection cu cung ten duoc xoa truoc khi index de khong con stale chunks
    khi corpus thay doi. Cac collection khac va file ngoai chroma_db khong bi tac dong.
    """
    if not chunks:
        raise ValueError("Cannot index an empty chunk list")

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    for chunk in chunks:
        content = chunk.get("content")
        metadata = chunk.get("metadata")
        embedding = chunk.get("embedding")
        if not isinstance(content, str) or not content:
            raise ValueError("Each indexed chunk needs non-empty content")
        if not isinstance(metadata, dict) or not metadata.get("chunk_id"):
            raise ValueError("Each indexed chunk needs metadata.chunk_id")
        if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM:
            raise ValueError(f"Each embedding must contain {EMBEDDING_DIM} values")

        ids.append(str(metadata["chunk_id"]))
        documents.append(content)
        embeddings.append(embedding)
        metadatas.append(metadata)

    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique before indexing")

    client = _get_chroma_client()
    existing_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=_collection_metadata(),
    )
    for start in range(0, len(chunks), CHROMA_WRITE_BATCH_SIZE):
        end = start + CHROMA_WRITE_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    indexed_count = collection.count()
    if indexed_count != len(chunks):
        raise RuntimeError(
            f"Chroma count mismatch: expected {len(chunks)}, got {indexed_count}"
        )
    return collection


def run_pipeline() -> dict[str, int]:
    """Chay toan bo pipeline load -> chunk -> embed -> index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} ({CHROMA_DIR})")
    print("=" * 60)

    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No Markdown documents found in {STANDARDIZED_DIR}")
    print(f"\nLoaded {len(documents)} documents")

    chunks = chunk_documents(documents)
    if not chunks:
        raise RuntimeError("Chunking produced no chunks")
    print(f"Created {len(chunks)} chunks")

    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")

    collection = index_to_vectorstore(embedded_chunks)
    print(f"Indexed {collection.count()} chunks to collection '{COLLECTION_NAME}'")
    return {"documents": len(documents), "chunks": len(embedded_chunks)}


if __name__ == "__main__":
    run_pipeline()
