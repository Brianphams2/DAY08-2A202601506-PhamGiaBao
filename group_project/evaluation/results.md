# RAG Evaluation Results

- Framework: RAGAS 0.1.21
- Judge model: `gemini-3.1-flash-lite` via `vilao` OpenAI-compatible API
- Evaluator embeddings: local `BAAI/bge-m3` (same multilingual model as retrieval)
- Golden dataset: 19 questions; both configs evaluated on the full set
- Generated: 2026-08-04T13:44:10+00:00

## Overall scores

| Metric | Config A: hybrid + RRF | Config B: dense-only | Delta A-B |
|---|---:|---:|---:|
| Faithfulness | 0.962 | 0.892 | 0.071 |
| Answer Relevance | 0.916 | 0.930 | -0.013 |
| Context Recall | 0.768 | 0.832 | -0.063 |
| Context Precision | 0.618 | 0.636 | -0.018 |
| Average | 0.816 | 0.822 | -0.006 |

## A/B analysis

- Config A: BGE-M3 dense + BM25, merged/reranked once with RRF.
- Config B: BGE-M3 semantic search only, without sparse retrieval or RRF.
- Result: **Config B (dense-only)** has the higher aggregate score on this corpus.
- PageIndex fallback and HyDE were disabled for both configs so the comparison isolates dense+sparse fusion and does not add remote retrieval/judge calls asymmetrically.

## Worst performers for Config A

| # | Question | Faithfulness | Relevance | Recall | Precision | Likely failure stage |
|---:|---|---:|---:|---:|---:|---|
| 1 | Người mua cần cung cấp những thông tin gì khi gửi yêu cầu Trả hàng/Hoàn tiền tại trang đơn hàng? | 1.000 | 0.785 | 0.000 | 0.000 | Retrieval recall |
| 2 | Sau khi Shopee chấp nhận phương án Trả hàng & Hoàn tiền, người mua có bao lâu để gửi trả hàng? | 1.000 | 0.833 | 0.000 | 0.000 | Retrieval recall |
| 3 | Tiền hoàn cho đơn thanh toán bằng thẻ tín dụng hoặc ghi nợ mất bao lâu và được hoàn về đâu? | 0.667 | 0.911 | 0.500 | 0.000 | Retrieval recall |

## Recommendations

1. Calibrate `SCORE_THRESHOLD` again whenever the corpus or embedding model changes; use the original dense cosine score, never the RRF score.
2. Add source-aware reranking or a multilingual cross-encoder when context precision is the weakest retrieval metric.
3. Expand golden cases for ambiguous follow-ups and out-of-domain questions, then measure PageIndex fallback separately from the controlled A/B test.

## Reproduce

```bash
python -m group_project.evaluation.eval_pipeline --no-cache
```

The ignored `evaluation_cache/` directory stores generated answers and per-question RAGAS rows so interrupted or rate-limited runs can resume; the report above is computed from the complete cached or newly generated record set.
