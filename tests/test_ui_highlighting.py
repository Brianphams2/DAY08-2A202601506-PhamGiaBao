"""Tests for safe evidence highlighting used by the Streamlit source viewer."""

import unittest

from src.ui_highlighting import highlight_evidence


class TestEvidenceHighlighting(unittest.TestCase):
    def test_matches_vietnamese_with_or_without_accents(self):
        rendered = highlight_evidence(
            "Thời hạn hoàn tiền là 24 giờ.",
            "thoi han hoan tien 24 gio",
        )
        self.assertIn('<mark class="evidence-highlight">Thời hạn hoàn tiền</mark>', rendered)
        self.assertIn('<mark class="evidence-highlight">24 giờ</mark>', rendered)

    def test_escapes_retrieved_html(self):
        rendered = highlight_evidence(
            '<script>alert("x")</script> hoàn tiền',
            "hoàn tiền",
        )
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn('<mark class="evidence-highlight">hoàn tiền</mark>', rendered)

    def test_returns_escaped_text_when_no_match(self):
        rendered = highlight_evidence("A < B\nDòng hai", "hoàn tiền")
        self.assertEqual(rendered, "A &lt; B<br>Dòng hai")

    def test_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            highlight_evidence("text", "query", max_terms=0)


if __name__ == "__main__":
    unittest.main()
