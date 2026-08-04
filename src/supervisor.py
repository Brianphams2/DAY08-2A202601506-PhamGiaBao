"""Supervisor for the complete Task 9 -> Task 10 RAG workflow.

The retrieval workers themselves run concurrently inside Task 9.  This module
owns configuration, follow-up context, and the single public interface used by
Streamlit and evaluation so those callers cannot drift into different pipelines.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .task10_generation import generate_with_citation
from .llm_config import llm_is_configured
from .task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int = 5
    score_threshold: float = SCORE_THRESHOLD
    retrieval_mode: str = "hybrid"
    lexical_method: str = "bm25"
    use_reranking: bool = True
    use_hyde: bool = False
    use_pageindex: bool = True
    use_reordering: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.retrieval_mode not in {"hybrid", "dense"}:
            raise ValueError("retrieval_mode must be 'hybrid' or 'dense'")
        if self.lexical_method not in {"bm25", "tfidf"}:
            raise ValueError("lexical_method must be 'bm25' or 'tfidf'")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")


class RAGSupervisor:
    """Coordinate retrieval, generation, and short conversation memory."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    @staticmethod
    def _retrieval_query(query: str, history: list[dict[str, Any]] | None) -> str:
        """Resolve short follow-ups with the most recent user turn."""
        if not history:
            return query
        normalized = query.casefold()
        follow_up_markers = ("còn", "thế còn", "việc đó", "nó", "như thế nào", "bao lâu")
        is_short = len(query.split()) <= 7
        if not is_short and not any(marker in normalized for marker in follow_up_markers):
            return query
        for message in reversed(history):
            if message.get("role") == "user" and str(message.get("content", "")).strip():
                previous = str(message["content"]).strip()
                if previous != query:
                    return f"Câu hỏi trước: {previous}\nCâu hỏi tiếp theo: {query}"
        return query

    def retrieve(self, query: str, history: list[dict[str, Any]] | None = None) -> list[dict]:
        retrieval_query = self._retrieval_query(query, history)
        return retrieve(
            retrieval_query,
            top_k=self.config.top_k,
            score_threshold=self.config.score_threshold,
            use_reranking=self.config.use_reranking,
            retrieval_mode=self.config.retrieval_mode,
            lexical_method=self.config.lexical_method,
            use_hyde=self.config.use_hyde,
            use_pageindex=self.config.use_pageindex,
        )

    def answer(self, query: str, history: list[dict[str, Any]] | None = None) -> dict:
        chunks = self.retrieve(query, history)
        result = generate_with_citation(
            query,
            context_chunks=chunks,
            top_k=self.config.top_k,
            use_reordering=self.config.use_reordering,
            conversation_history=history,
        )
        result["config"] = asdict(self.config)
        return result

    @staticmethod
    def healthcheck() -> dict[str, Any]:
        project_root = Path(__file__).resolve().parent.parent
        chroma_path = project_root / "chroma_db" / "chroma.sqlite3"
        standardized = list((project_root / "data" / "standardized").rglob("*.md"))
        return {
            "standardized_documents": len(standardized),
            "chroma_ready": chroma_path.is_file(),
            "llm_configured": llm_is_configured(),
            "pageindex_configured": bool(os.getenv("PAGEINDEX_API_KEY", "").strip()),
        }


def run_pipeline(
    query: str,
    *,
    history: list[dict[str, Any]] | None = None,
    config: PipelineConfig | None = None,
) -> dict:
    """Convenience entry point for CLI, UI, and integration tests."""
    return RAGSupervisor(config).answer(query, history)


if __name__ == "__main__":
    supervisor = RAGSupervisor()
    print(supervisor.healthcheck())
    print(supervisor.answer("Shopee hỗ trợ những phương thức thanh toán nào?")["answer"])
