"""
Task 3 - Convert toan bo file trong data/landing/ thanh Markdown.

PDF/DOC/DOCX duoc convert bang MarkItDown cua Microsoft. Cac bai ho tro JSON
da co noi dung Markdown tu Task 2 se duoc chuan hoa va bo sung metadata nguon.
Output giu nguyen cau truc thu muc con trong data/standardized/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from markitdown import MarkItDown


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_ROOT / "data" / "landing"
OUTPUT_DIR = PROJECT_ROOT / "data" / "standardized"

LEGAL_EXTENSIONS = {".pdf", ".doc", ".docx"}
LEGAL_MANIFEST = LANDING_DIR / "legal" / "source_manifest.json"
SHOPEE_PDF_CHROME_LINES = {
    "Trung tâm trợ giúp Shopee VN",
    "Shopee Policies",
}


def _load_legal_metadata() -> dict[str, dict[str, Any]]:
    """Load metadata cua Task 1, neu manifest ton tai."""
    if not LEGAL_MANIFEST.exists():
        return {}

    records = json.loads(LEGAL_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Legal manifest must contain a list: {LEGAL_MANIFEST}")

    metadata_by_filename: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("filename"), str):
            metadata_by_filename[record["filename"]] = record
    return metadata_by_filename


def _metadata_header(data: dict[str, Any], fallback_title: str) -> str:
    """Tao Markdown header co metadata phuc vu truy vet va citation."""
    title = str(data.get("title") or fallback_title).strip()
    lines = [f"# {title}", ""]

    fields = (
        ("Source", data.get("url") or data.get("canonical_url")),
        ("Platform", data.get("platform")),
        ("Category", data.get("category")),
        ("Source type", data.get("source_type")),
        ("Customer role", data.get("customer_role")),
        ("Language", data.get("language")),
        ("Effective date", data.get("effective_date")),
        ("Source updated", data.get("source_last_updated")),
        ("Crawled", data.get("crawled_at") or data.get("date_crawled")),
        ("Captured", data.get("captured_at")),
    )
    for label, value in fields:
        if value not in (None, ""):
            lines.append(f"**{label}:** {value}")

    lines.extend(("", "---", "", ""))
    return "\n".join(lines)


def _remove_duplicate_title(content: str, title: str) -> str:
    """Bo heading dau tien neu no trung voi title da co trong metadata header."""
    lines = content.strip().splitlines()
    if not lines:
        return ""

    first_line = re.sub(r"^#{1,6}\s+", "", lines[0]).strip()
    if first_line.casefold() == title.strip().casefold():
        lines = lines[1:]
    return "\n".join(lines).strip()


def _clean_legal_content(content: str, title: str | None) -> str:
    """Loai bo navigation/footer va page header do Chromium chen vao PDF."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")

    if title:
        title_index = normalized.find(title)
        if title_index >= 0:
            normalized = normalized[title_index + len(title) :]

    feedback_index = normalized.find("Bạn có hài lòng với bài viết này?")
    if feedback_index >= 0:
        normalized = normalized[:feedback_index]

    clean_lines = [
        line
        for line in normalized.splitlines()
        if line.strip() not in SHOPEE_PDF_CHROME_LINES
    ]
    normalized = "\n".join(clean_lines)
    normalized = re.sub(r"\n[ \t]+\n", "\n\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _write_markdown(output_path: Path, header: str, content: str) -> None:
    """Validate va ghi mot file Markdown UTF-8."""
    content = content.strip()
    if not content:
        raise ValueError(f"Converted Markdown is empty: {output_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{header}{content}\n", encoding="utf-8")


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX trong data/landing/legal/ bang MarkItDown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    if not legal_dir.is_dir():
        raise FileNotFoundError(f"Legal input directory not found: {legal_dir}")

    metadata_by_filename = _load_legal_metadata()
    converter = MarkItDown()
    converted_paths: list[Path] = []

    source_files = sorted(
        path
        for path in legal_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in LEGAL_EXTENSIONS
    )
    for filepath in source_files:
        print(f"Converting: {filepath.relative_to(LANDING_DIR)}")
        result = converter.convert(str(filepath))
        raw_content = result.text_content or ""

        metadata = metadata_by_filename.get(filepath.name, {})
        title = str(metadata.get("title") or filepath.stem.replace("-", " ").title())
        content = _clean_legal_content(raw_content, title)
        header = _metadata_header(metadata, title)

        relative_path = filepath.relative_to(legal_dir).with_suffix(".md")
        output_path = output_dir / relative_path
        _write_markdown(output_path, header, content)
        converted_paths.append(output_path)
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    return converted_paths


def convert_news_articles() -> list[Path]:
    """Chuan hoa cac bai ho tro JSON trong data/landing/news/ thanh Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    if not news_dir.is_dir():
        raise FileNotFoundError(f"News input directory not found: {news_dir}")

    converted_paths: list[Path] = []
    for filepath in sorted(path for path in news_dir.rglob("*.json") if path.is_file()):
        print(f"Converting: {filepath.relative_to(LANDING_DIR)}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"News JSON must contain an object: {filepath}")

        title = str(data.get("title") or filepath.stem.replace("-", " ").title())
        raw_content = data.get("content_markdown")
        if not isinstance(raw_content, str):
            raise ValueError(f"Missing content_markdown in: {filepath}")

        content = _remove_duplicate_title(raw_content, title)
        header = _metadata_header(data, title)
        relative_path = filepath.relative_to(news_dir).with_suffix(".md")
        output_path = output_dir / relative_path
        _write_markdown(output_path, header, content)
        converted_paths.append(output_path)
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    return converted_paths


def convert_all() -> list[Path]:
    """Convert toan bo tai lieu Task 1-2 sang Markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_paths = convert_legal_docs()

    print("\n--- News Articles ---")
    news_paths = convert_news_articles()

    converted_paths = legal_paths + news_paths
    print(f"\nDone: {len(converted_paths)} files saved to {OUTPUT_DIR}")
    return converted_paths


if __name__ == "__main__":
    convert_all()
