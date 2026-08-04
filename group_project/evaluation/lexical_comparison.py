"""Compare BM25 and TF-IDF retrieval against expected sources in the golden set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from src.task6_lexical_search import lexical_search
from src.task6_tfidf_search import tfidf_search


EVALUATION_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "lexical_results.md"


def _normalize_source(value: object) -> str:
    return str(value or "").replace("\\", "/").strip().casefold()


def _source_rank(results: list[dict], expected_source: str) -> int | None:
    expected = _normalize_source(expected_source)
    for rank, row in enumerate(results, 1):
        source = _normalize_source((row.get("metadata") or {}).get("source"))
        if source == expected:
            return rank
    return None


def _evaluate_method(
    search: Callable[[str, int], list[dict]],
    dataset: list[dict],
    top_k: int,
) -> dict:
    rows: list[dict] = []
    for item in dataset:
        results = search(str(item["question"]), top_k)
        rank = _source_rank(results, str(item.get("expected_source", "")))
        rows.append({
            "id": item.get("id", ""),
            "question": item["question"],
            "expected_source": item.get("expected_source", ""),
            "rank": rank,
        })
    hit_count = sum(row["rank"] is not None for row in rows)
    reciprocal_rank = sum(1.0 / row["rank"] for row in rows if row["rank"] is not None)
    top1_count = sum(row["rank"] == 1 for row in rows)
    total = len(rows)
    return {
        "rows": rows,
        "hit_at_k": hit_count / total if total else 0.0,
        "mrr": reciprocal_rank / total if total else 0.0,
        "top1_accuracy": top1_count / total if total else 0.0,
    }


def compare(top_k: int = 5) -> dict:
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("golden_dataset.json must contain questions")
    return {
        "dataset_size": len(dataset),
        "top_k": top_k,
        "bm25": _evaluate_method(lexical_search, dataset, top_k),
        "tfidf": _evaluate_method(tfidf_search, dataset, top_k),
    }


def export_markdown(comparison: dict) -> Path:
    top_k = comparison["top_k"]
    bm25 = comparison["bm25"]
    tfidf = comparison["tfidf"]
    lines = [
        "# Lexical Retrieval Comparison",
        "",
        f"- Golden dataset: {comparison['dataset_size']} questions",
        f"- Retrieval depth: top-{top_k}",
        "- Both methods use the same accent-folding, stopword removal and bigram tokenizer.",
        "",
        "## Overall",
        "",
        "| Metric | BM25 | TF-IDF |",
        "|---|---:|---:|",
        f"| Expected-source hit@{top_k} | {bm25['hit_at_k']:.3f} | {tfidf['hit_at_k']:.3f} |",
        f"| Mean reciprocal rank | {bm25['mrr']:.3f} | {tfidf['mrr']:.3f} |",
        f"| Expected source at rank 1 | {bm25['top1_accuracy']:.3f} | {tfidf['top1_accuracy']:.3f} |",
        "",
        "## Per-question expected-source rank",
        "",
        "| Question | Expected source | BM25 rank | TF-IDF rank |",
        "|---|---|---:|---:|",
    ]
    for bm25_row, tfidf_row in zip(bm25["rows"], tfidf["rows"], strict=True):
        question = str(bm25_row["question"]).replace("|", "\\|")
        expected = str(bm25_row["expected_source"]).replace("|", "\\|")
        bm25_rank = bm25_row["rank"] if bm25_row["rank"] is not None else "—"
        tfidf_rank = tfidf_row["rank"] if tfidf_row["rank"] is not None else "—"
        lines.append(f"| {question} | `{expected}` | {bm25_rank} | {tfidf_rank} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "BM25 applies term-frequency saturation and document-length normalization. TF-IDF uses normalized term weights without BM25's saturation parameters. This report is a controlled lexical-only comparison; the production pipeline still uses BM25 by default and can select TF-IDF from the Streamlit sidebar.",
        "",
    ])
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    return RESULTS_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare BM25 and TF-IDF lexical retrieval")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    output = export_markdown(compare(args.top_k))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
