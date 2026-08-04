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


def to_unsigned_vietnamese(text: str) -> str:
    """Chuyển đổi tiếng Việt có dấu thành không dấu để ghi file PDF an toàn bằng fpdf2."""
    patterns = {
        '[àáảãạăằắẳẵặâầấẩẫậ]': 'a',
        '[ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]': 'A',
        '[èéẻẽẹêềếểễệ]': 'e',
        '[ÈÉẺẼẸÊỀẾỂỄỆ]': 'E',
        '[ìíỉĩị]': 'i',
        '[ÌÍỈĨỊ]': 'I',
        '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
        '[ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ]': 'O',
        '[ùúủũụưừứửữự]': 'u',
        '[ÙÚỦŨỤƯỪỨỬỮỰ]': 'U',
        '[ỳýỷỹỵ]': 'y',
        '[ỲÝỶỸỴ]': 'Y',
        'đ': 'd',
        'Đ': 'D'
    }
    import re
    res = text
    for pattern, repl in patterns.items():
        res = re.sub(pattern, repl, res)
    return res


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Returns:
        Danh sách doc_ids đã được upload thành công.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY not set")

    from pageindex.client import PageIndexClient
    from fpdf import FPDF
    import tempfile

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        # Đọc nội dung markdown
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Tạo PDF từ nội dung
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=10)
        
        content_unsigned = to_unsigned_vietnamese(content)
        for line in content_unsigned.splitlines():
            clean_line = line.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, txt=clean_line)

        # Lưu file tạm
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = tmp.name

        try:
            pdf.output(pdf_path)
            # Submit lên PageIndex
            resp = client.submit_document(pdf_path)
            doc_id = resp.get("doc_id") or resp.get("id")
            print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)


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
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY not set")

    from pageindex.client import PageIndexClient
    import time

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Lấy danh sách tài liệu trong workspace
    docs = client.list_documents()
    results = []

    for doc in docs:
        if isinstance(doc, dict):
            doc_id = doc.get("id") or doc.get("doc_id")
            filename = doc.get("filename") or doc.get("name") or "unknown"
        else:
            doc_id = getattr(doc, "id", None) or getattr(doc, "doc_id", None)
            filename = getattr(doc, "filename", None) or getattr(doc, "name", "unknown")

        if not doc_id:
            continue

        # Đảm bảo document sẵn sàng để query
        if not client.is_retrieval_ready(doc_id):
            continue

        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        if not retrieval_id:
            continue

        # Poll cho đến khi status == "completed" hoặc "failed"
        retrieval = client.get_retrieval(retrieval_id)
        while retrieval.get("status") not in ("completed", "failed"):
            time.sleep(0.5)
            retrieval = client.get_retrieval(retrieval_id)

        if retrieval.get("status") == "failed":
            continue

        # Parse retrieved_nodes
        for node in retrieval.get("retrieved_nodes", [])[:2]:
            for group in node.get("relevant_contents", []):
                for item in group:
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": 0.5,
                        "metadata": {
                            "section": item.get("section_title"),
                            "source": filename
                        },
                        "source": "pageindex",
                    })

    # Lọc trùng và gán lại score giảm dần
    seen_content = set()
    unique_results = []
    for item in results:
        content = item["content"]
        if content not in seen_content:
            seen_content.add(content)
            unique_results.append(item)

    for idx, item in enumerate(unique_results):
        item["score"] = 1.0 - (idx * 0.1)

    return unique_results[:top_k]


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

