"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
TEMP_PDF_DIR = Path(__file__).parent.parent / "data" / "temp_pdf"


def _md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """Chuyển đổi file Markdown sang PDF đơn giản để upload lên PageIndex."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    content = md_path.read_text(encoding="utf-8", errors="ignore")
    # Clean non-latin1 characters for basic fpdf compatibility
    safe_content = content.encode("latin-1", "replace").decode("latin-1")
    for line in safe_content.splitlines():
        pdf.multi_cell(0, 8, text=line)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def upload_documents() -> list[str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Returns:
        Danh sách doc_ids đã được upload thành công.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa được set trong .env")
        return []

    try:
        from pageindex import PageIndexClient
    except ImportError:
        print("⚠ Thư viện 'pageindex' chưa được cài đặt (pip install pageindex).")
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids: list[str] = []

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"⚠ Không tìm thấy file markdown nào trong {STANDARDIZED_DIR}")
        return []

    TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

    for md_file in md_files:
        pdf_path = TEMP_PDF_DIR / f"{md_file.stem}.pdf"
        try:
            _md_to_pdf(md_file, pdf_path)
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id") or resp.get("document_id")
            if doc_id:
                doc_ids.append(str(doc_id))
                print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
            else:
                print(f"  ⚠ Uploaded {md_file.name} nhưng không lấy được doc_id: {resp}")
        except Exception as e:
            print(f"  ❌ Lỗi upload {md_file.name}: {e}")

    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not query or top_k <= 0:
        return []

    if not PAGEINDEX_API_KEY:
        # Khi chưa cấu hình API Key, trả về danh sách rỗng
        return []

    try:
        from pageindex import PageIndexClient
    except ImportError:
        return []

    try:
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        # Lấy danh sách tài liệu đã upload
        doc_list_resp = client.list_documents()
        documents = []
        if isinstance(doc_list_resp, list):
            documents = doc_list_resp
        elif isinstance(doc_list_resp, dict):
            documents = doc_list_resp.get("documents") or doc_list_resp.get("data") or []

        if not documents:
            # Nếu chưa có tài liệu nào trên PageIndex, thử upload
            uploaded = upload_documents()
            if not uploaded:
                return []
            doc_id = uploaded[0]
        else:
            first_doc = documents[0]
            doc_id = first_doc.get("id") or first_doc.get("doc_id") or first_doc.get("document_id")

        if not doc_id:
            return []

        # Gửi query
        query_resp = client.submit_query(doc_id=str(doc_id), query=query)
        retrieval_id = query_resp.get("retrieval_id") or query_resp.get("id")

        if not retrieval_id:
            return []

        # Poll cho đến khi status == "completed"
        max_retries = 30
        retrieval: dict[str, Any] = {}
        for _ in range(max_retries):
            retrieval = client.get_retrieval(str(retrieval_id))
            status = retrieval.get("status")
            if status == "completed":
                break
            elif status == "failed":
                print(f"⚠ PageIndex retrieval failed: {retrieval}")
                return []
            time.sleep(1)

        retrieved_nodes = retrieval.get("retrieved_nodes") or []
        results: list[dict] = []
        score_counter = 1.0

        for node in retrieved_nodes:
            relevant_contents = node.get("relevant_contents") or []
            for group in relevant_contents:
                if isinstance(group, list):
                    for item in group:
                        if isinstance(item, dict):
                            content = item.get("relevant_content") or item.get("text") or ""
                            section = item.get("section_title") or node.get("title") or ""
                            if content:
                                results.append(
                                    {
                                        "content": content,
                                        "score": score_counter,
                                        "metadata": {
                                            "section": section,
                                            "node_id": node.get("id"),
                                        },
                                        "source": "pageindex",
                                    }
                                )
                                score_counter = max(0.1, score_counter - 0.05)
                elif isinstance(group, dict):
                    content = group.get("relevant_content") or group.get("text") or ""
                    section = group.get("section_title") or node.get("title") or ""
                    if content:
                        results.append(
                            {
                                "content": content,
                                "score": score_counter,
                                "metadata": {
                                    "section": section,
                                    "node_id": node.get("id"),
                                },
                                "source": "pageindex",
                            }
                        )
                        score_counter = max(0.1, score_counter - 0.05)

        return results[:top_k]

    except Exception as e:
        print(f"⚠ PageIndex search error: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")

