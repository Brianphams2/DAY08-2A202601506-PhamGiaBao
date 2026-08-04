"""Task 10 - context reordering and answer generation with citations."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from dotenv import load_dotenv

from .llm_config import extract_chat_content, get_llm_provider
from .task9_retrieval_pipeline import retrieve

load_dotenv()

# Five chunks provide enough independent evidence for the current short-policy
# corpus without flooding the model's middle context.  Low temperature keeps
# factual answers stable; top_p=0.9 leaves enough wording flexibility for clear
# Vietnamese while retrieval, not sampling, determines factual content.
TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.2
MAX_COMPLETION_TOKENS = 900
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))

UNVERIFIABLE_ANSWER = (
    "I cannot verify this information "
    "(Tôi không thể xác minh thông tin này từ nguồn hiện có)."
)

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp chính sách thương mại điện tử.

Chỉ trả lời bằng thông tin được nêu trong CONTEXT. Khi CONTEXT trực tiếp chứa câu
trả lời, hãy trả lời rõ ràng bằng tiếng Việt; sau mỗi câu hoặc mục liệt kê, chèn đúng
Citation label [Tên nguồn, Năm] của nguồn hỗ trợ. Không dùng kiến thức bên ngoài,
không bịa URL, tên nguồn hoặc năm. Chỉ khi CONTEXT thực sự không chứa câu trả lời,
hãy trả lời đúng câu: I cannot verify this information (Tôi không thể xác minh thông
tin này từ nguồn hiện có).
"""

_CITATION_RE = re.compile(r"\[[^\[\]\n]+,\s*(?:19|20)\d{2}\]")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Move high-ranked chunks to the beginning and end of the context.

    Input ``[1, 2, 3, 4, 5]`` becomes ``[1, 3, 5, 4, 2]``.  A new list is
    returned and the retrieval ranking is never mutated.
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return [*front, *reversed(back)]


def _source_label(chunk: dict, index: int) -> tuple[str, str]:
    metadata = chunk.get("metadata") or {}
    title = str(
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("filename")
        or f"Nguồn {index}"
    ).strip()
    # Support-article titles often begin with tags such as ``[Thành viên mới]``.
    # Nested square brackets make an invalid citation label, so remove the tag
    # and any remaining brackets before presenting the exact label to the LLM.
    title = re.sub(r"^\[[^]]+\]\s*", "", title)
    title = title.replace("[", "").replace("]", "").strip().rstrip("?.!")
    year = str(metadata.get("year") or "2026").strip()
    if not re.fullmatch(r"(?:19|20)\d{2}", year):
        year = "2026"
    return title, year


def format_context(chunks: list[dict]) -> str:
    """Format evidence with the exact labels the model must cite."""
    context_parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        metadata = chunk.get("metadata") or {}
        title, year = _source_label(chunk, index)
        source_file = metadata.get("source") or metadata.get("filename") or title
        context_parts.append(
            f"SOURCE {index}\n"
            f"Citation label: [{title}, {year}]\n"
            f"File: {source_file}\n"
            f"Content:\n{content}"
        )
    return "\n\n---\n\n".join(context_parts)


def _history_text(conversation_history: list[dict[str, Any]] | None) -> str:
    if not conversation_history:
        return ""
    lines: list[str] = []
    for message in conversation_history[-6:]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    *,
    top_k: int = TOP_K,
    use_reordering: bool = True,
    conversation_history: list[dict[str, Any]] | None = None,
    retrieval_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Run retrieval (when needed), reorder context, and generate a cited answer."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    chunks = (
        retrieve(query, top_k=top_k, **(retrieval_kwargs or {}))
        if context_chunks is None
        else [item.copy() for item in context_chunks[:top_k]]
    )
    if not chunks:
        return {
            "answer": UNVERIFIABLE_ANSWER,
            "sources": [],
            "reordered_sources": [],
            "retrieval_source": "none",
            "model": "none",
        }

    reordered = reorder_for_llm(chunks) if use_reordering else list(chunks)
    context = format_context(reordered)
    if not context:
        return {
            "answer": UNVERIFIABLE_ANSWER,
            "sources": chunks,
            "reordered_sources": reordered,
            "retrieval_source": chunks[0].get("source", "hybrid"),
            "model": "none",
        }

    history = _history_text(conversation_history)
    history_block = f"CONVERSATION HISTORY:\n{history}\n\n" if history else ""
    user_message = (
        f"{history_block}CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        "Hãy trả lời câu hỏi và dùng đúng citation label của nguồn hỗ trợ."
    )

    from openai import OpenAI

    provider = get_llm_provider()
    client_kwargs: dict[str, Any] = {
        "api_key": provider.api_key,
        "timeout": LLM_TIMEOUT_SECONDS,
    }
    if provider.base_url:
        client_kwargs["base_url"] = provider.base_url
    client = OpenAI(**client_kwargs)
    response = None
    last_error: Exception | None = None
    for attempt in range(1, max(1, LLM_MAX_ATTEMPTS) + 1):
        try:
            response = client.chat.completions.create(
                model=provider.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_COMPLETION_TOKENS,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt >= max(1, LLM_MAX_ATTEMPTS):
                raise
            time.sleep(min(2 ** (attempt - 1), 4))
    if response is None:
        raise RuntimeError("LLM returned no response") from last_error
    answer = extract_chat_content(response)
    if not answer:
        answer = UNVERIFIABLE_ANSWER
    elif answer != UNVERIFIABLE_ANSWER and not _CITATION_RE.search(answer):
        # Surface a contract violation instead of presenting an uncited factual
        # answer as successful output.
        answer = UNVERIFIABLE_ANSWER

    return {
        "answer": answer,
        "sources": chunks,
        "reordered_sources": reordered,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "model": provider.model,
    }


if __name__ == "__main__":
    for question in (
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để yêu cầu trả hàng hoặc hoàn tiền?",
    ):
        result = generate_with_citation(question)
        print(f"\nQ: {question}\nA: {result['answer']}")
