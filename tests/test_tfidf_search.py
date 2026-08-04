"""Tests for the bonus TF-IDF lexical retrieval strategy."""

import unittest

from src.task6_tfidf_search import build_tfidf_index, tfidf_search
from src.supervisor import PipelineConfig
from src.task9_retrieval_pipeline import retrieve


class TestTFIDFSearch(unittest.TestCase):
    def test_build_rejects_empty_corpus(self):
        with self.assertRaises(ValueError):
            build_tfidf_index([])

    def test_returns_ranked_results(self):
        results = tfidf_search("payment methods", top_k=3)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        self.assertTrue(all({"content", "score", "metadata"}.issubset(row) for row in results))
        scores = [row["score"] for row in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_accent_insensitive_query(self):
        accented = tfidf_search("hoàn tiền", top_k=1)
        plain = tfidf_search("hoan tien", top_k=1)
        self.assertTrue(accented and plain)
        self.assertEqual(
            accented[0]["metadata"].get("chunk_id"),
            plain[0]["metadata"].get("chunk_id"),
        )

    def test_hybrid_pipeline_accepts_tfidf(self):
        results = retrieve(
            "payment methods",
            top_k=3,
            lexical_method="tfidf",
            use_pageindex=False,
        )
        self.assertTrue(results)
        self.assertTrue(any(row.get("tfidf_rank") is not None for row in results))
        self.assertTrue(all(row.get("lexical_method") == "tfidf" for row in results))

    def test_pipeline_config_rejects_unknown_lexical_method(self):
        with self.assertRaises(ValueError):
            PipelineConfig(lexical_method="unknown")


if __name__ == "__main__":
    unittest.main()
