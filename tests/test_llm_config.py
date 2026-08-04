"""Tests for shared OpenAI-compatible provider resolution."""

import os
import unittest
from unittest.mock import patch

from src.llm_config import extract_chat_content, get_llm_provider, llm_is_configured


PROVIDER_KEYS = {
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
}


def clean_environment(**values):
    environment = {key: value for key, value in os.environ.items() if key not in PROVIDER_KEYS}
    environment.update(values)
    return patch.dict(os.environ, environment, clear=True)


class TestLLMConfig(unittest.TestCase):
    def test_generic_provider_has_priority_and_normalizes_url(self):
        with clean_environment(
            LLM_PROVIDER="vilao",
            LLM_API_KEY="secret",
            LLM_BASE_URL="https://provider.example/v1/",
            LLM_MODEL="provider/model",
            OPENROUTER_API_KEY="legacy",
        ):
            provider = get_llm_provider()
        self.assertEqual(provider.name, "vilao")
        self.assertEqual(provider.base_url, "https://provider.example/v1")
        self.assertEqual(provider.model, "provider/model")

    def test_model_override_is_used_for_judge_or_hyde(self):
        with clean_environment(
            LLM_API_KEY="secret",
            LLM_BASE_URL="https://provider.example/v1",
            LLM_MODEL="generation-model",
        ):
            provider = get_llm_provider(model_override="judge-model")
        self.assertEqual(provider.model, "judge-model")

    def test_openrouter_legacy_fallback_remains_supported(self):
        with clean_environment(OPENROUTER_API_KEY="legacy-secret"):
            provider = get_llm_provider()
        self.assertEqual(provider.name, "openrouter")
        self.assertEqual(provider.base_url, "https://openrouter.ai/api/v1")

    def test_generic_provider_requires_base_url(self):
        with clean_environment(LLM_API_KEY="secret", LLM_MODEL="model"):
            with self.assertRaisesRegex(RuntimeError, "LLM_BASE_URL"):
                get_llm_provider()

    def test_placeholders_are_not_treated_as_configured(self):
        with clean_environment(LLM_API_KEY="sk-...", OPENAI_API_KEY="sk-proj-..."):
            self.assertFalse(llm_is_configured())
            with self.assertRaises(RuntimeError):
                get_llm_provider()

    def test_extracts_typed_and_direct_proxy_responses(self):
        typed = {"choices": [{"message": {"content": "  Câu trả lời  "}}]}
        self.assertEqual(extract_chat_content(typed), "Câu trả lời")
        self.assertEqual(extract_chat_content("  Trả lời trực tiếp  "), "Trả lời trực tiếp")

    def test_extracts_json_encoded_proxy_response(self):
        encoded = '{"choices":[{"message":{"content":"Nội dung JSON"}}]}'
        self.assertEqual(extract_chat_content(encoded), "Nội dung JSON")


if __name__ == "__main__":
    unittest.main()
