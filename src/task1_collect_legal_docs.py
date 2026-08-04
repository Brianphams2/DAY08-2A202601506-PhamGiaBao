"""Task 1 - Capture official Shopee Vietnam policy pages as PDF files.

Shopee publishes these policies as public HTML pages rather than downloadable
PDFs.  This module opens each official page in Chromium, waits for the policy
text to render, and creates a faithful browser-print PDF.  A manifest records
the original URL and capture method so the provenance remains explicit.

Run:
    python -m src.task1_collect_legal_docs
    python -m src.task1_collect_legal_docs --force
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "source_manifest.json"
USER_AGENT = "VinUni-RAG-Lab/1.0 (+educational-use)"
LOCAL_TIMEZONE = timezone(timedelta(hours=7))

POLICY_SOURCES = [
    {
        "filename": "returns-refund-policy-shopee.pdf",
        "url": "https://help.shopee.vn/portal/4/article/77251?seo=1",
        "title": "CHÍNH SÁCH TRẢ HÀNG VÀ HOÀN TIỀN",
        "wait_text": "CHÍNH SÁCH TRẢ HÀNG VÀ HOÀN TIỀN",
        "category": "return_refund",
        "customer_role": "both",
        "effective_date": "2026-03-11",
    },
    {
        "filename": "privacy-policy-shopee.pdf",
        "url": "https://help.shopee.vn/portal/4/article/77244?seo=1",
        "title": "CHÍNH SÁCH BẢO MẬT",
        "wait_text": "CHÍNH SÁCH BẢO MẬT",
        "category": "privacy",
        "customer_role": "both",
        "effective_date": "2026-06-11",
    },
    {
        "filename": "product-listing-regulations-shopee.pdf",
        "url": "https://help.shopee.vn/portal/4/article/77246?seo=1",
        "title": "QUY ĐỊNH VỀ ĐĂNG BÁN SẢN PHẨM TRÊN SHOPEE",
        "wait_text": "QUY ĐỊNH VỀ ĐĂNG BÁN SẢN PHẨM",
        "category": "seller_policy",
        "customer_role": "seller",
        "effective_date": None,
    },
    {
        "filename": "shipping-policy-shopee.pdf",
        "url": "https://help.shopee.vn/portal/4/article/77250?seo=1",
        "title": "CHÍNH SÁCH VẬN CHUYỂN SHOPEE",
        "wait_text": "CHÍNH SÁCH VẬN CHUYỂN SHOPEE",
        "category": "shipping",
        "customer_role": "both",
        "effective_date": None,
    },
    {
        "filename": "terms-of-service-shopee.pdf",
        "url": "https://help.shopee.vn/portal/4/article/77243?seo=1",
        "title": "ĐIỀU KHOẢN DỊCH VỤ",
        "wait_text": "ĐIỀU KHOẢN DỊCH VỤ",
        "category": "terms_of_service",
        "customer_role": "both",
        "effective_date": None,
    },
]


def setup_directory() -> None:
    """Create the landing directory when it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_browser() -> Path:
    candidates = (
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Chrome or Microsoft Edge is required to create policy PDFs")


def _validate_source_page(source: dict) -> None:
    request = Request(source["url"], headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status {response.status}: {source['url']}")
    if source["wait_text"] not in html or "ssr-key-content" not in html:
        raise RuntimeError(f"Official policy content is incomplete: {source['url']}")


def _print_policy_pdf(browser: Path, source: dict, output_path: Path) -> None:
    _validate_source_page(source)
    with tempfile.TemporaryDirectory(prefix="vinuni-rag-browser-") as profile_dir:
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=8000",
            f"--user-data-dir={profile_dir}",
            f"--print-to-pdf={output_path}",
            source["url"],
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 and not output_path.exists():
            error = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Browser PDF capture failed: {error}")


async def _capture_policy(browser: Path, source: dict, force: bool) -> dict:
    output_path = DATA_DIR / source["filename"]
    if output_path.exists() and output_path.stat().st_size > 1024 and not force:
        print(f"Skip existing: {output_path.name}")
    else:
        print(f"Capture: {source['filename']}")
        await asyncio.to_thread(_print_policy_pdf, browser, source, output_path)
        print(f"Saved: {output_path.name} ({output_path.stat().st_size:,} bytes)")

    if output_path.stat().st_size <= 1024:
        raise RuntimeError(f"Generated PDF is too small: {output_path}")

    if output_path.read_bytes()[:4] != b"%PDF":
        raise RuntimeError(f"Generated file is not a valid PDF: {output_path}")

    captured_at = datetime.now(LOCAL_TIMEZONE).isoformat()
    return {
        "filename": source["filename"],
        "url": source["url"],
        "canonical_url": source["url"].split("?", 1)[0],
        "title": source["title"],
        "platform": "Shopee Vietnam",
        "language": "vi",
        "category": source["category"],
        "source_type": "official_policy",
        "customer_role": source["customer_role"],
        "capture_method": "chromium_browser_print_to_pdf",
        "captured_at": captured_at,
        "effective_date": source["effective_date"],
        "file_size": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }


async def collect_legal_docs(force: bool = False) -> list[dict]:
    """Capture every policy and return the provenance manifest."""
    setup_directory()
    browser = _find_browser()

    manifest: list[dict] = []
    for index, source in enumerate(POLICY_SOURCES):
        manifest.append(await _capture_policy(browser, source, force))
        if index < len(POLICY_SOURCES) - 1:
            await asyncio.sleep(2)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved manifest: {MANIFEST_PATH.name}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing PDFs")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    asyncio.run(collect_legal_docs(force=arguments.force))
