"""Task 2 - Crawl official Shopee Vietnam customer-support articles.

Shopee Help Center currently exposes its article body in server-rendered HTML.
This module therefore uses a lightweight HTTP crawler and a site-specific HTML
parser to select only the official article body.  Each article is stored as one
UTF-8 JSON file with provenance, a content hash, and cleaned Markdown.

Run:
    python -m src.task2_crawl_news
    python -m src.task2_crawl_news --force
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "news"
USER_AGENT = "VinUni-RAG-Lab/1.0 (+educational-use)"
LOCAL_TIMEZONE = timezone(timedelta(hours=7))
MIN_CONTENT_CHARS = 350

ARTICLE_SOURCES = [
    {
        "id": "payment-methods",
        "title": "Shopee hiện đang có những phương thức thanh toán nào?",
        "url": "https://help.shopee.vn/portal/4/article/79198?seo=1",
        "category": "payment",
        "customer_role": "buyer",
    },
    {
        "id": "change-payment-method",
        "title": "Tôi có thể thay đổi phương thức thanh toán cho đơn hàng không?",
        "url": "https://help.shopee.vn/portal/4/article/79555?seo=1",
        "category": "payment",
        "customer_role": "buyer",
    },
    {
        "id": "submit-return-refund",
        "title": "Hướng dẫn gửi yêu cầu Trả hàng/Hoàn tiền",
        "url": "https://help.shopee.vn/portal/4/article/79233?seo=1",
        "category": "return_refund",
        "customer_role": "buyer",
    },
    {
        "id": "refund-evidence",
        "title": "Hướng dẫn chuẩn bị bằng chứng khi yêu cầu Trả hàng/Hoàn tiền",
        "url": "https://help.shopee.vn/portal/4/article/79467?seo=1",
        "category": "refund_evidence",
        "customer_role": "buyer",
    },
    {
        "id": "refund-process",
        "title": "Quy trình Shopee xử lý yêu cầu Trả hàng/Hoàn tiền",
        "url": "https://help.shopee.vn/portal/4/article/190242?seo=1",
        "category": "return_refund",
        "customer_role": "buyer",
    },
    {
        "id": "refund-time",
        "title": "Thời gian nhận tiền hoàn và cách kiểm tra tiền hoàn",
        "url": "https://help.shopee.vn/portal/4/article/189473?seo=1",
        "category": "refund_status",
        "customer_role": "buyer",
    },
    {
        "id": "return-shipping-methods",
        "title": "Các phương thức gửi hàng hoàn trả và phí hoàn trả",
        "url": "https://help.shopee.vn/portal/4/article/189477?seo=1",
        "category": "return_refund",
        "customer_role": "buyer",
    },
    {
        "id": "late-delivery",
        "title": "Đã quá thời gian dự kiến giao hàng nhưng tôi chưa nhận được hàng",
        "url": "https://help.shopee.vn/portal/4/article/79530?seo=1",
        "category": "order_tracking",
        "customer_role": "buyer",
    },
    {
        "id": "delivery-estimate",
        "title": "Kiểm tra phí vận chuyển và thời gian giao hàng dự kiến",
        "url": "https://help.shopee.vn/portal/4/article/79573?seo=1",
        "category": "shipping",
        "customer_role": "buyer",
    },
    {
        "id": "international-shipping",
        "title": "Thông tin về kênh vận chuyển đơn hàng quốc tế",
        "url": "https://help.shopee.vn/portal/4/article/79651?seo=1",
        "category": "cross_border",
        "customer_role": "buyer",
    },
]

# Backward-compatible name used in the starter comments and by students.
ARTICLE_URLS = [source["url"] for source in ARTICLE_SOURCES]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


class _ShopeeArticleParser(HTMLParser):
    """Extract only the server-rendered Shopee article title and body."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._article_div_depth = 0
        self._ignored_depth = 0
        self._title_parts: list[str] = []
        self._content_parts: list[str] = []

    @property
    def in_article(self) -> bool:
        return self._article_div_depth > 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "h2" and attributes.get("id") == "hcArticleTitle":
            self._in_title = True

        classes = (attributes.get("class") or "").split()
        if tag == "div" and "ssr-key-content" in classes and not self.in_article:
            self._article_div_depth = 1
            return

        if not self.in_article:
            return

        if tag == "div":
            self._article_div_depth += 1
        if tag in {"script", "style", "svg"}:
            self._ignored_depth += 1
            return

        prefixes = {"h3": "\n\n## ", "h4": "\n\n### ", "h5": "\n\n#### "}
        if tag in prefixes:
            self._content_parts.append(prefixes[tag])
        elif tag == "li":
            self._content_parts.append("\n- ")
        elif tag == "br":
            self._content_parts.append("\n")
        elif tag == "tr":
            self._content_parts.append("\n")
        elif tag in {"th", "td"}:
            self._content_parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self._in_title:
            self._in_title = False

        if not self.in_article:
            return

        if tag in {"script", "style", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        if self._ignored_depth == 0 and tag in {"p", "h3", "h4", "h5", "li", "tr"}:
            self._content_parts.append("\n")
        if tag == "div":
            self._article_div_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        if self.in_article and self._ignored_depth == 0:
            self._content_parts.append(re.sub(r"\s+", " ", data))

    def result(self) -> tuple[str, str]:
        title = re.sub(r"\s+", " ", "".join(self._title_parts)).strip()
        content = "".join(self._content_parts).replace("\xa0", " ")
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r" *\n *", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        return title, content


def _fetch_article(url: str) -> tuple[int, str, str, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        status_code = response.status
        final_url = response.geturl()
        html = response.read().decode("utf-8", errors="replace")
    if status_code != 200:
        raise RuntimeError(f"Unexpected HTTP status {status_code}: {url}")

    parser = _ShopeeArticleParser()
    parser.feed(html)
    title, content = parser.result()
    if not title or not content:
        raise RuntimeError(f"Shopee article SSR content was not found: {url}")
    return status_code, final_url, title, f"# {title}\n\n{content}"


async def crawl_article(url: str, page=None, source: dict | None = None) -> dict:
    """Crawl one public article and return metadata plus cleaned Markdown."""
    source = source or {
        "id": "article",
        "title": "Unknown",
        "url": url,
        "category": "support",
        "customer_role": "buyer",
    }
    del page  # retained only for backward compatibility with the starter signature
    status_code, final_url, parsed_title, content = await asyncio.to_thread(
        _fetch_article, url
    )
    if len(content) < MIN_CONTENT_CHARS:
        raise ValueError(
            f"Article content is too short: {source['id']} ({len(content)} chars)"
        )

    crawled_at = datetime.now(LOCAL_TIMEZONE).isoformat()
    return {
        "schema_version": 1,
        "id": source["id"],
        "url": url,
        "canonical_url": final_url.split("?", 1)[0],
        "title": parsed_title or source["title"],
        "platform": "Shopee Vietnam",
        "category": source["category"],
        "source_type": "support_article",
        "customer_role": source["customer_role"],
        "language": "vi",
        "crawled_at": crawled_at,
        "source_last_updated": None,
        "year": int(crawled_at[:4]),
        "http_status": status_code,
        "crawl_method": "http_ssr_htmlparser",
        "word_count": len(content.split()),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_markdown": content,
    }


async def crawl_all(force: bool = False) -> list[Path]:
    """Crawl the curated source list sequentially with a polite delay."""
    setup_directory()
    saved: list[Path] = []
    errors: list[str] = []

    for index, source in enumerate(ARTICLE_SOURCES, 1):
        output_path = DATA_DIR / f"{source['id']}.json"
        if output_path.exists() and output_path.stat().st_size > 500 and not force:
            print(f"[{index}/{len(ARTICLE_SOURCES)}] Skip existing: {output_path.name}")
            saved.append(output_path)
            continue

        print(f"[{index}/{len(ARTICLE_SOURCES)}] Crawl: {source['url']}")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                article = await crawl_article(source["url"], None, source)
                output_path.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"Saved: {output_path.name} "
                    f"({article['word_count']} words, {output_path.stat().st_size:,} bytes)"
                )
                saved.append(output_path)
                last_error = None
                break
            except Exception as exc:  # retry transient network failures
                last_error = exc
                print(f"Attempt {attempt}/3 failed: {exc}")
                if attempt < 3:
                    await asyncio.sleep(2**attempt)

        if last_error is not None:
            errors.append(f"{source['id']}: {last_error}")

        if index < len(ARTICLE_SOURCES):
            await asyncio.sleep(2)

    if len(saved) < 5:
        raise RuntimeError(
            f"Only {len(saved)} articles were saved; at least 5 are required. "
            f"Errors: {errors}"
        )
    if errors:
        print("Completed with non-fatal errors:")
        for error in errors:
            print(f"- {error}")
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing JSON files")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    asyncio.run(crawl_all(force=arguments.force))
