"""Reproducible RAGAS evaluation and A/B comparison for the group project.

Run the full required benchmark with::

    python -m group_project.evaluation.eval_pipeline

Progress is cached per configuration in ``evaluation_cache/`` so an interrupted
OpenRouter run can resume without regenerating completed answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.llm_config import get_llm_provider
from src.supervisor import PipelineConfig, RAGSupervisor

load_dotenv()

EVALUATION_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "results.md"
CACHE_DIR = EVALUATION_DIR / "evaluation_cache"
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


@dataclass(frozen=True)
class EvaluationConfig:
    name: str
    description: str
    retrieval_mode: str
    use_reranking: bool

    def pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(
            top_k=5,
            retrieval_mode=self.retrieval_mode,
            use_reranking=self.use_reranking,
            use_hyde=False,
            # Keep A/B comparable and prevent one low-confidence row from
            # creating remote PageIndex retrieval calls during the benchmark.
            use_pageindex=False,
            use_reordering=True,
        )


CONFIGS = (
    EvaluationConfig(
        name="hybrid_rrf",
        description="BGE-M3 dense + BM25, merged/reranked once with RRF",
        retrieval_mode="hybrid",
        use_reranking=True,
    ),
    EvaluationConfig(
        name="dense_only",
        description="BGE-M3 semantic search only, without sparse retrieval or RRF",
        retrieval_mode="dense",
        use_reranking=False,
    ),
)


def load_golden_dataset() -> list[dict]:
    """Load and validate the required 15+ Q&A golden dataset."""
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 15:
        raise ValueError("golden_dataset.json must contain at least 15 Q&A pairs")
    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Golden item {index} is missing one of {sorted(required)}")
        if not all(str(item[key]).strip() for key in required):
            raise ValueError(f"Golden item {index} contains an empty required field")
    return data


def _cache_path(config: EvaluationConfig) -> Path:
    return CACHE_DIR / f"{config.name}.json"


def _load_cache(config: EvaluationConfig) -> dict[str, dict]:
    path = _cache_path(config)
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return {str(row["id"]): row for row in rows if isinstance(row, dict) and row.get("id")}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _save_cache(config: EvaluationConfig, rows: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(config).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def collect_answers(
    config: EvaluationConfig,
    golden_dataset: list[dict],
    *,
    reuse_cache: bool = True,
) -> list[dict]:
    """Run one pipeline config over every golden question, with resumable cache."""
    cached = _load_cache(config) if reuse_cache else {}
    supervisor = RAGSupervisor(config.pipeline_config())
    rows: list[dict] = []
    for index, item in enumerate(golden_dataset, 1):
        item_id = str(item.get("id") or f"question_{index:02d}")
        if item_id in cached:
            rows.append(cached[item_id])
            print(f"[{config.name}] {index}/{len(golden_dataset)} cached: {item_id}")
            continue

        print(f"[{config.name}] {index}/{len(golden_dataset)} generate: {item_id}")
        result = supervisor.answer(item["question"])
        row = {
            "id": item_id,
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [str(chunk.get("content", "")) for chunk in result.get("sources", [])],
            "ground_truth": item["expected_answer"],
            "expected_source": item.get("expected_source", ""),
            "retrieved_sources": [
                str((chunk.get("metadata") or {}).get("source", ""))
                for chunk in result.get("sources", [])
            ],
        }
        rows.append(row)
        _save_cache(config, rows)
    return rows


class LocalBGEEmbeddings:
    """LangChain-compatible adapter around the exact BGE-M3 retriever model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from src.task4_chunking_indexing import get_embedding_model

        vectors = get_embedding_model().encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _judge_models(judge_model: str):
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    judge_max_tokens = int(os.getenv("RAGAS_MAX_TOKENS", "2048"))
    provider = get_llm_provider(model_override=judge_model)
    chat = ChatOpenAI(
        model=provider.model,
        api_key=provider.api_key,
        base_url=provider.base_url,
        temperature=0,
        max_tokens=judge_max_tokens,
        timeout=90,
        max_retries=2,
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(LocalBGEEmbeddings())


def _clean_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(score) or math.isinf(score) else score


def evaluate_with_ragas(
    records: list[dict],
    *,
    judge_model: str,
    max_workers: int = 1,
) -> dict:
    """Evaluate collected answers with all four metrics required by README."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    dataset = Dataset.from_dict({
        "question": [row["question"] for row in records],
        "answer": [row["answer"] for row in records],
        "contexts": [row["contexts"] for row in records],
        "ground_truth": [row["ground_truth"] for row in records],
    })
    evaluator_llm, evaluator_embeddings = _judge_models(judge_model)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(
            timeout=180,
            max_retries=2,
            max_wait=20,
            max_workers=max(1, max_workers),
        ),
        raise_exceptions=True,
    )
    frame = result.to_pandas()
    per_question: list[dict] = []
    for index, record in enumerate(records):
        metric_scores = {metric: _clean_score(frame.iloc[index].get(metric)) for metric in METRICS}
        valid_scores = [score for score in metric_scores.values() if score is not None]
        per_question.append({
            "id": record["id"],
            "question": record["question"],
            **metric_scores,
            "average": sum(valid_scores) / len(valid_scores) if valid_scores else None,
            "retrieved_sources": record.get("retrieved_sources", []),
            "expected_source": record.get("expected_source", ""),
        })

    overall = {}
    for metric in METRICS:
        values = [row[metric] for row in per_question if row[metric] is not None]
        overall[metric] = sum(values) / len(values) if values else None
    valid_overall = [value for value in overall.values() if value is not None]
    overall["average"] = sum(valid_overall) / len(valid_overall) if valid_overall else None
    return {"overall": overall, "per_question": per_question}


def _aggregate_rows(per_question: list[dict]) -> dict:
    """Build overall scores from already-scored rows."""
    overall: dict[str, float | None] = {}
    for metric in METRICS:
        values = [row.get(metric) for row in per_question if row.get(metric) is not None]
        overall[metric] = sum(values) / len(values) if values else None
    valid_overall = [value for value in overall.values() if value is not None]
    overall["average"] = sum(valid_overall) / len(valid_overall) if valid_overall else None
    return {"overall": overall, "per_question": per_question}


def evaluate_resumable(
    records: list[dict],
    *,
    config_name: str,
    judge_model: str,
    max_workers: int,
    reuse_cache: bool,
) -> dict:
    """Score small batches and persist each completed batch for safe resume."""
    provider = get_llm_provider(model_override=judge_model)
    judge_identity = {
        "provider": provider.name,
        "base_url": provider.base_url,
        "model": provider.model,
        "max_tokens": int(os.getenv("RAGAS_MAX_TOKENS", "2048")),
    }
    row_cache_path = CACHE_DIR / f"{config_name}_ragas_rows.json"
    cached: dict[str, dict] = {}
    if reuse_cache and row_cache_path.is_file():
        try:
            payload = json.loads(row_cache_path.read_text(encoding="utf-8"))
            cached_rows = payload.get("rows", {}) if isinstance(payload, dict) else {}
            cached = cached_rows if isinstance(cached_rows, dict) else {}
        except (OSError, json.JSONDecodeError, TypeError):
            cached = {}

    signatures: dict[str, str] = {}
    completed: dict[str, dict] = {}
    pending: list[dict] = []
    for record in records:
        item_id = str(record["id"])
        signature = hashlib.sha256(
            json.dumps(
                {"judge": judge_identity, "record": record},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        signatures[item_id] = signature
        cache_item = cached.get(item_id)
        if (
            isinstance(cache_item, dict)
            and cache_item.get("signature") == signature
            and isinstance(cache_item.get("score"), dict)
        ):
            completed[item_id] = cache_item["score"]
        else:
            pending.append(record)

    if completed:
        print(f"[{config_name}] {len(completed)}/{len(records)} cached RAGAS rows")

    batch_size = max(1, int(os.getenv("RAGAS_BATCH_SIZE", "5")))
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        print(
            f"[{config_name}] score batch "
            f"{start // batch_size + 1}/{math.ceil(len(pending) / batch_size)} "
            f"({len(batch)} rows)"
        )
        batch_scores = evaluate_with_ragas(
            batch,
            judge_model=judge_model,
            max_workers=max_workers,
        )
        for score in batch_scores["per_question"]:
            completed[str(score["id"])] = score

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        row_cache_path.write_text(
            json.dumps(
                {
                    "judge": judge_identity,
                    "rows": {
                        item_id: {
                            "signature": signatures[item_id],
                            "score": score,
                        }
                        for item_id, score in completed.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    ordered = [completed[str(record["id"])] for record in records]
    return _aggregate_rows(ordered)


def compare_configs(
    golden_dataset: list[dict],
    *,
    judge_model: str,
    max_workers: int = 1,
    reuse_cache: bool = True,
) -> dict:
    """Run the required hybrid+RRF versus dense-only A/B benchmark."""
    comparison: dict[str, Any] = {}
    for config in CONFIGS:
        records = collect_answers(config, golden_dataset, reuse_cache=reuse_cache)
        score_cache_path = CACHE_DIR / f"{config.name}_ragas.json"
        provider = get_llm_provider(model_override=judge_model)
        signature = hashlib.sha256(
            json.dumps(
                {
                    "judge_model": judge_model,
                    "judge_provider": provider.name,
                    "judge_base_url": provider.base_url,
                    "ragas_max_tokens": int(os.getenv("RAGAS_MAX_TOKENS", "2048")),
                    "records": records,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        scores = None
        if reuse_cache and score_cache_path.is_file():
            try:
                cached_scores = json.loads(score_cache_path.read_text(encoding="utf-8"))
                if cached_scores.get("signature") == signature:
                    scores = cached_scores.get("scores")
                    print(f"[{config.name}] cached RAGAS scores")
            except (OSError, json.JSONDecodeError, TypeError):
                scores = None
        if not isinstance(scores, dict):
            scores = evaluate_resumable(
                records,
                config_name=config.name,
                judge_model=judge_model,
                max_workers=max_workers,
                reuse_cache=reuse_cache,
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            score_cache_path.write_text(
                json.dumps(
                    {"signature": signature, "scores": scores},
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
        comparison[config.name] = {
            "config": asdict(config),
            "records": records,
            **scores,
        }
    return comparison


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def export_results(comparison: dict, *, judge_model: str, dataset_size: int) -> Path:
    """Write the scored A/B table, worst performers, and actionable analysis."""
    a = comparison["hybrid_rrf"]
    b = comparison["dense_only"]
    judge_provider = get_llm_provider(model_override=judge_model).name
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
        "average": "Average",
    }
    lines = [
        "# RAG Evaluation Results",
        "",
        f"- Framework: RAGAS 0.1.21",
        f"- Judge model: `{judge_model}` via `{judge_provider}` OpenAI-compatible API",
        "- Evaluator embeddings: local `BAAI/bge-m3` (same multilingual model as retrieval)",
        f"- Golden dataset: {dataset_size} questions; both configs evaluated on the full set",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Overall scores",
        "",
        "| Metric | Config A: hybrid + RRF | Config B: dense-only | Delta A-B |",
        "|---|---:|---:|---:|",
    ]
    for metric in (*METRICS, "average"):
        a_score = a["overall"].get(metric)
        b_score = b["overall"].get(metric)
        delta = None if a_score is None or b_score is None else a_score - b_score
        lines.append(f"| {labels[metric]} | {_fmt(a_score)} | {_fmt(b_score)} | {_fmt(delta)} |")

    a_avg = a["overall"].get("average")
    b_avg = b["overall"].get("average")
    if a_avg is None or b_avg is None:
        result_statement = "The aggregate result is unavailable because one configuration has no valid score."
    elif math.isclose(a_avg, b_avg, abs_tol=0.0005):
        result_statement = "The two configurations are tied at three-decimal reporting precision."
    elif a_avg > b_avg:
        result_statement = "**Config A (hybrid + RRF)** has the higher aggregate score on this corpus."
    else:
        result_statement = "**Config B (dense-only)** has the higher aggregate score on this corpus."
    lines.extend([
        "",
        "## A/B analysis",
        "",
        f"- Config A: {a['config']['description']}.",
        f"- Config B: {b['config']['description']}.",
        f"- Result: {result_statement}",
        "- PageIndex fallback and HyDE were disabled for both configs so the comparison isolates dense+sparse fusion and does not add remote retrieval/judge calls asymmetrically.",
        "",
        "## Worst performers for Config A",
        "",
        "| # | Question | Faithfulness | Relevance | Recall | Precision | Likely failure stage |",
        "|---:|---|---:|---:|---:|---:|---|",
    ])
    ranked = sorted(
        a["per_question"],
        key=lambda row: row.get("average") if row.get("average") is not None else -1,
    )[:3]
    for index, row in enumerate(ranked, 1):
        recall = row.get("context_recall")
        precision = row.get("context_precision")
        faith = row.get("faithfulness")
        if recall is not None and recall < 0.6:
            stage = "Retrieval recall"
        elif precision is not None and precision < 0.6:
            stage = "Retrieval precision/ranking"
        elif faith is not None and faith < 0.6:
            stage = "Generation grounding"
        else:
            stage = "Answer relevance"
        question = str(row["question"]).replace("|", "\\|")
        lines.append(
            f"| {index} | {question} | {_fmt(row.get('faithfulness'))} | "
            f"{_fmt(row.get('answer_relevancy'))} | {_fmt(recall)} | {_fmt(precision)} | {stage} |"
        )

    lines.extend([
        "",
        "## Recommendations",
        "",
        "1. Calibrate `SCORE_THRESHOLD` again whenever the corpus or embedding model changes; use the original dense cosine score, never the RRF score.",
        "2. Add source-aware reranking or a multilingual cross-encoder when context precision is the weakest retrieval metric.",
        "3. Expand golden cases for ambiguous follow-ups and out-of-domain questions, then measure PageIndex fallback separately from the controlled A/B test.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m group_project.evaluation.eval_pipeline --no-cache",
        "```",
        "",
        "The ignored `evaluation_cache/` directory stores generated answers and per-question RAGAS rows so interrupted or rate-limited runs can resume; the report above is computed from the complete cached or newly generated record set.",
        "",
    ])
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    return RESULTS_PATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full RAGAS A/B evaluation")
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-4o-mini"),
        help="OpenRouter judge model ID",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="RAGAS judge concurrency")
    parser.add_argument("--sample-size", type=int, default=0, help="Smoke subset; 0 means full dataset")
    parser.add_argument("--no-cache", action="store_true", help="Regenerate answers instead of resuming")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    golden_dataset = load_golden_dataset()
    if args.sample_size:
        if args.sample_size < 1:
            raise ValueError("sample-size must be positive")
        golden_dataset = golden_dataset[: args.sample_size]
        print(f"Smoke mode: {len(golden_dataset)} questions (report will still state this size)")
    comparison = compare_configs(
        golden_dataset,
        judge_model=args.judge_model,
        max_workers=args.max_workers,
        reuse_cache=not args.no_cache,
    )
    output = export_results(
        comparison,
        judge_model=args.judge_model,
        dataset_size=len(golden_dataset),
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
