"""Safe, accent-insensitive evidence highlighting for the Streamlit UI."""

from __future__ import annotations

import html
import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_STOPWORDS = {
    "ai", "ban", "bi", "cac", "cho", "co", "cua", "dang", "de", "duoc",
    "gi", "hien", "ho", "hoac", "la", "mot", "nao", "nguoi", "nhung",
    "o", "shopee", "the", "thi", "tren", "trong", "tro", "va", "viet", "voi",
}


def _normalize_with_index(text: str) -> tuple[str, list[int]]:
    """Fold Vietnamese accents while retaining a map to original character offsets."""
    normalized: list[str] = []
    original_indices: list[int] = []
    for original_index, character in enumerate(text):
        folded = unicodedata.normalize("NFKD", character.casefold())
        folded = "".join(char for char in folded if not unicodedata.combining(char))
        folded = folded.replace("đ", "d")
        for char in folded:
            normalized.append(char)
            original_indices.append(original_index)
    return "".join(normalized), original_indices


def _query_terms(query: str, max_terms: int) -> list[str]:
    normalized, _ = _normalize_with_index(query)
    unique: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(normalized):
        if token in _STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        if token not in seen:
            seen.add(token)
            unique.append(token)
        if len(unique) >= max_terms:
            break
    return unique


def _merge_ranges(text: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        gap = text[previous_end:start]
        if start <= previous_end or re.fullmatch(r"[\s,.;:/()\-–—]*", gap):
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def highlight_evidence(
    text: str,
    query: str,
    *,
    max_terms: int = 12,
    max_occurrences: int = 40,
) -> str:
    """Return escaped HTML with query evidence wrapped in ``<mark>`` tags.

    Matching is case- and accent-insensitive, but the original Vietnamese text
    is preserved. User/document HTML is escaped before the trusted mark tags are
    inserted, so retrieved content cannot inject markup into the Streamlit page.
    """
    if not isinstance(text, str) or not isinstance(query, str):
        raise TypeError("text and query must be strings")
    if max_terms <= 0 or max_occurrences <= 0:
        raise ValueError("max_terms and max_occurrences must be positive")

    terms = _query_terms(query, max_terms)
    normalized_text, original_indices = _normalize_with_index(text)
    if not text or not terms or not normalized_text:
        return html.escape(text).replace("\n", "<br>")

    ranges: list[tuple[int, int]] = []
    for term in terms:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
        for match in pattern.finditer(normalized_text):
            if len(ranges) >= max_occurrences:
                break
            start = original_indices[match.start()]
            end = original_indices[match.end() - 1] + 1
            ranges.append((start, end))
        if len(ranges) >= max_occurrences:
            break

    ranges = _merge_ranges(text, sorted(set(ranges)))
    if not ranges:
        return html.escape(text).replace("\n", "<br>")

    output: list[str] = []
    cursor = 0
    for start, end in ranges:
        output.append(html.escape(text[cursor:start]))
        output.append('<mark class="evidence-highlight">')
        output.append(html.escape(text[start:end]))
        output.append("</mark>")
        cursor = end
    output.append(html.escape(text[cursor:]))
    return "".join(output).replace("\n", "<br>")
