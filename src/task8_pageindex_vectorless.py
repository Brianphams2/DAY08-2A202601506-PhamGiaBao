"""Task 8 - PageIndex vectorless retrieval fallback.

The PageIndex cloud API accepts PDF files.  The standardized Markdown corpus is
therefore compiled into two Unicode PDFs (legal and news), uploaded once, and
tracked in an ignored local manifest.  Retrieval uses the PageIndex legacy
``/retrieval`` endpoint because Task 8 needs evidence chunks rather than a
generated chat answer.  The endpoint is still supported by PageIndex, although
their current SDK recommends the Chat API for new conversational products.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
TEMP_PDF_DIR = PROJECT_ROOT / "tmp" / "pdfs" / "pageindex"
MANIFEST_PATH = PROJECT_ROOT / "pageindex_doc_ids.json"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()

PAGEINDEX_POLL_INTERVAL = float(os.getenv("PAGEINDEX_POLL_INTERVAL", "2"))
PAGEINDEX_RETRIEVAL_TIMEOUT = float(os.getenv("PAGEINDEX_RETRIEVAL_TIMEOUT", "120"))
PAGEINDEX_PROCESSING_TIMEOUT = float(os.getenv("PAGEINDEX_PROCESSING_TIMEOUT", "900"))


def _client():
    """Create the current PageIndex SDK client lazily."""
    if not PAGEINDEX_API_KEY or "..." in PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured")

    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _unicode_font_paths() -> tuple[Path, Path]:
    """Return a regular/bold TrueType pair that contains Vietnamese glyphs."""
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            return regular, bold
    raise RuntimeError("No Vietnamese-capable TrueType font was found for PDF conversion")


def _plain_markdown(line: str) -> str:
    """Remove lightweight Markdown syntax while preserving readable text."""
    replacements = {
        "⚠️": "Lưu ý: ",
        "⚠": "Lưu ý: ",
        "✔": "Có",
        "✓": "Có",
        "❌": "Không",
        "→": "->",
        "–": "-",
        "—": "-",
    }
    for symbol, replacement in replacements.items():
        line = line.replace(symbol, replacement)
    line = line.replace("\ufe0f", "").replace("\u200b", "").replace("\u200d", "")
    line = re.sub(r"[\U0001F300-\U0001FAFF]", "", line)
    line = re.sub(r"Lưu ý:\s*Lưu ý:\s*", "Lưu ý: ", line, flags=re.IGNORECASE)
    line = re.sub(r"!\[[^]]*]\([^)]*\)", "", line)
    line = re.sub(r"\[([^]]+)]\(([^)]+)\)", r"\1 (\2)", line)
    line = re.sub(r"[`*~]", "", line)
    return line.strip()


def build_corpus_pdf(markdown_files: Iterable[Path], output_path: Path, title: str) -> Path:
    """Compile Markdown files into a legible, searchable Unicode PDF."""
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    files = sorted(Path(path) for path in markdown_files)
    if not files:
        raise ValueError("At least one Markdown document is required")

    regular_font, bold_font = _unicode_font_paths()
    pdfmetrics.registerFont(TTFont("CorpusSans", str(regular_font)))
    pdfmetrics.registerFont(TTFont("CorpusSans-Bold", str(bold_font)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "CorpusBody", parent=styles["BodyText"], fontName="CorpusSans",
        fontSize=9.5, leading=13, spaceAfter=5,
    )
    h1 = ParagraphStyle(
        "CorpusH1", parent=styles["Heading1"], fontName="CorpusSans-Bold",
        fontSize=17, leading=21, spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "CorpusH2", parent=styles["Heading2"], fontName="CorpusSans-Bold",
        fontSize=13, leading=17, spaceBefore=7, spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "CorpusH3", parent=styles["Heading3"], fontName="CorpusSans-Bold",
        fontSize=11, leading=15, spaceBefore=5, spaceAfter=4,
    )
    cover = ParagraphStyle(
        "CorpusCover", parent=h1, alignment=TA_CENTER, fontSize=22, leading=27,
        spaceAfter=14,
    )

    story: list[Any] = [Paragraph(html.escape(title), cover), Spacer(1, 8 * mm)]
    for file_index, markdown_path in enumerate(files):
        text = markdown_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        if file_index:
            story.append(PageBreak())
        story.append(Paragraph(html.escape(f"Source file: {markdown_path.name}"), h2))
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped == "---":
                story.append(Spacer(1, 2.5 * mm))
                continue
            heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading:
                level = len(heading.group(1))
                style = h1 if level == 1 else h2 if level == 2 else h3
                story.append(Paragraph(html.escape(_plain_markdown(heading.group(2))), style))
                continue
            clean = _plain_markdown(stripped)
            if clean:
                story.append(Paragraph(html.escape(clean), body))

    def add_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("CorpusSans", 8)
        canvas.setFillGray(0.45)
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title=title,
    )
    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return output_path


def _corpus_groups() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for markdown_path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        relative = markdown_path.relative_to(STANDARDIZED_DIR)
        category = relative.parts[0] if len(relative.parts) > 1 else "general"
        groups.setdefault(category, []).append(markdown_path)
    return groups


def _corpus_hash(groups: dict[str, list[Path]]) -> str:
    digest = hashlib.sha256()
    for category, paths in sorted(groups.items()):
        digest.update(category.encode("utf-8"))
        for path in paths:
            digest.update(path.relative_to(STANDARDIZED_DIR).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def upload_documents(
    *,
    force: bool = False,
    wait_for_ready: bool = True,
    timeout_seconds: float = PAGEINDEX_PROCESSING_TIMEOUT,
) -> list[str]:
    """Upload the complete standardized corpus and return PageIndex document IDs.

    The function never deletes remote documents.  If the corpus hash matches the
    ignored local manifest, existing IDs are reused so reruns do not create
    duplicate uploads.
    """
    groups = _corpus_groups()
    if not groups:
        raise FileNotFoundError(f"No Markdown documents found in {STANDARDIZED_DIR}")

    corpus_hash = _corpus_hash(groups)
    previous = _load_manifest()
    previous_docs = previous.get("documents") if isinstance(previous.get("documents"), list) else []
    if not force and previous.get("corpus_hash") == corpus_hash and previous_docs:
        ids = [str(item.get("doc_id")) for item in previous_docs if item.get("doc_id")]
        if ids:
            return ids

    client = _client()
    uploaded: list[dict[str, str]] = []
    TEMP_PDF_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for category, markdown_files in sorted(groups.items()):
            pdf_name = f"ecommerce-support-{category}.pdf"
            pdf_path = build_corpus_pdf(
                markdown_files,
                TEMP_PDF_DIR / pdf_name,
                f"E-commerce Support Corpus - {category.title()}",
            )
            response = client.submit_document(str(pdf_path))
            doc_id = response.get("doc_id") or response.get("id")
            if not doc_id:
                raise RuntimeError(f"PageIndex upload returned no doc_id for {pdf_name}: {response}")
            uploaded.append({"category": category, "name": pdf_name, "doc_id": str(doc_id)})
            print(f"  [PageIndex] Uploaded {pdf_name} -> {doc_id}")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "corpus_hash": corpus_hash,
            "documents": uploaded,
        }
        _write_manifest(manifest)

        if wait_for_ready:
            deadline = time.monotonic() + timeout_seconds
            pending = {item["doc_id"] for item in uploaded}
            while pending and time.monotonic() < deadline:
                for doc_id in list(pending):
                    status = str(client.get_document(doc_id).get("status", "unknown")).lower()
                    if status == "completed":
                        pending.remove(doc_id)
                    elif status == "failed":
                        raise RuntimeError(f"PageIndex processing failed for {doc_id}")
                if pending:
                    time.sleep(PAGEINDEX_POLL_INTERVAL)
            if pending:
                raise TimeoutError(f"PageIndex processing timed out for: {sorted(pending)}")
        return [item["doc_id"] for item in uploaded]
    finally:
        if TEMP_PDF_DIR.exists():
            shutil.rmtree(TEMP_PDF_DIR)


def _configured_documents(client: Any) -> list[dict[str, str]]:
    manifest_docs = _load_manifest().get("documents", [])
    configured = [
        {"doc_id": str(item.get("doc_id")), "name": str(item.get("name", "unknown.pdf"))}
        for item in manifest_docs
        if isinstance(item, dict) and item.get("doc_id")
    ]
    if configured:
        return configured

    response = client.list_documents(limit=100)
    documents = response.get("documents", response if isinstance(response, list) else [])
    return [
        {
            "doc_id": str(doc.get("id") or doc.get("doc_id")),
            "name": str(doc.get("name") or doc.get("filename") or "unknown.pdf"),
        }
        for doc in documents
        if isinstance(doc, dict)
        and (doc.get("id") or doc.get("doc_id"))
        and str(doc.get("name") or doc.get("filename") or "").startswith("ecommerce-support-")
    ]


def _relevant_items(value: Any, inherited_section: str = "") -> Iterable[dict[str, Any]]:
    """Flatten both documented and older nested ``relevant_contents`` schemas."""
    if isinstance(value, list):
        for item in value:
            yield from _relevant_items(item, inherited_section)
    elif isinstance(value, dict):
        section = str(value.get("section_title") or value.get("title") or inherited_section)
        content = value.get("relevant_content")
        if isinstance(content, str) and content.strip():
            yield {**value, "section_title": section, "relevant_content": content.strip()}
        elif "relevant_contents" in value:
            yield from _relevant_items(value.get("relevant_contents"), section)


def _query_document(client: Any, doc: dict[str, str], query: str) -> list[dict]:
    doc_id = doc["doc_id"]
    if not client.is_retrieval_ready(doc_id):
        return []

    submitted = client.submit_query(doc_id=doc_id, query=query, thinking=False)
    retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
    if not retrieval_id:
        raise RuntimeError(f"PageIndex returned no retrieval_id for {doc_id}: {submitted}")

    deadline = time.monotonic() + PAGEINDEX_RETRIEVAL_TIMEOUT
    while time.monotonic() < deadline:
        retrieval = client.get_retrieval(retrieval_id)
        status = str(retrieval.get("status", "")).lower()
        if status == "failed":
            raise RuntimeError(f"PageIndex retrieval failed for {doc_id}")
        if status == "completed":
            rows: list[dict] = []
            for node in retrieval.get("retrieved_nodes", []):
                section = str(node.get("title") or node.get("section_title") or "")
                for item in _relevant_items(node.get("relevant_contents", []), section):
                    rows.append({
                        "content": item["relevant_content"],
                        "score": 0.0,
                        "metadata": {
                            "source": doc["name"],
                            "title": item.get("section_title") or section or doc["name"],
                            "section": item.get("section_title") or section,
                            "page": item.get("page_index"),
                            "year": 2026,
                            "type": "pageindex",
                            "doc_id": doc_id,
                            "pageindex_rank": len(rows) + 1,
                        },
                        "source": "pageindex",
                    })
            return rows
        time.sleep(PAGEINDEX_POLL_INTERVAL)
    raise TimeoutError(f"PageIndex retrieval timed out for {doc_id}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve evidence from PageIndex for Task 9's low-confidence fallback."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    query = query.strip()
    if not query or top_k <= 0:
        return []

    client = _client()
    documents = _configured_documents(client)
    if not documents:
        raise RuntimeError("No uploaded PageIndex corpus found; run upload_documents() first")

    results: list[dict] = []
    errors: list[str] = []
    for document in documents:
        try:
            results.extend(_query_document(client, document, query))
        except Exception as exc:
            errors.append(f"{document['name']}: {exc}")

    unique: list[dict] = []
    seen: set[str] = set()
    for item in results:
        key = re.sub(r"\s+", " ", item["content"]).strip().casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    # The legacy API provides order but no cross-document numeric score.  Rank
    # the returned PageIndex evidence with lightweight multilingual token
    # overlap so results from the alphabetically first corpus cannot crowd out
    # a much more relevant result from the second corpus.
    from .task6_lexical_search import tokenize

    query_tokens = set(tokenize(query))
    for item in unique:
        evidence_tokens = set(tokenize(
            f"{item['metadata'].get('title', '')} {item['content']}"
        ))
        overlap = sum(2.0 if "_" in token else 1.0 for token in query_tokens & evidence_tokens)
        denominator = sum(2.0 if "_" in token else 1.0 for token in query_tokens) or 1.0
        source_rank = max(1, int(item["metadata"].get("pageindex_rank", 1)))
        item["score"] = overlap / denominator + 0.05 / source_rank
    unique.sort(key=lambda item: item["score"], reverse=True)

    if not unique and errors:
        raise RuntimeError("; ".join(errors))
    return unique[:top_k]


if __name__ == "__main__":
    ids = upload_documents()
    print(f"Ready PageIndex documents: {len(ids)}")
    for result in pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
