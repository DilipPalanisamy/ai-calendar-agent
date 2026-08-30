import sys
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from gemini_resilience import (
    call_gemini_with_retry,
    acall_gemini_with_retry,
    is_rate_limit_error,
)

class MockResourceExhausted(Exception):
    def __init__(self, message="429 RESOURCE_EXHAUSTED: Quota exceeded"):
        super().__init__(message)


class TestGeminiResilience(unittest.TestCase):
    def test_is_rate_limit_error(self):
        self.assertTrue(is_rate_limit_error(MockResourceExhausted()))
        self.assertTrue(is_rate_limit_error(Exception("429 Too Many Requests")))
        self.assertTrue(is_rate_limit_error(Exception("RESOURCE_EXHAUSTED")))
        self.assertTrue(is_rate_limit_error(Exception("503 Service Unavailable")))
        self.assertFalse(is_rate_limit_error(ValueError("Invalid argument")))

    def test_call_gemini_with_retry_success(self):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = "Mocked Success Response"

        result = call_gemini_with_retry(mock_model, "Test Prompt", max_retries=3, initial_delay=0.01)
        self.assertEqual(result, "Mocked Success Response")
        self.assertEqual(mock_model.generate_content.call_count, 1)

    def test_call_gemini_with_retry_recovers_after_rate_limit(self):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            MockResourceExhausted("429 Resource Exhausted"),
            MockResourceExhausted("429 Resource Exhausted"),
            "Success after 2 retries",
        ]

        result = call_gemini_with_retry(mock_model, "Test Prompt", max_retries=5, initial_delay=0.01)
        self.assertEqual(result, "Success after 2 retries")
        self.assertEqual(mock_model.generate_content.call_count, 3)

    def test_call_gemini_with_retry_exceeds_max_retries(self):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = MockResourceExhausted("429 Resource Exhausted")

        with self.assertRaises(MockResourceExhausted):
            call_gemini_with_retry(mock_model, "Test Prompt", max_retries=3, initial_delay=0.01)

        self.assertEqual(mock_model.generate_content.call_count, 3)

    def test_call_gemini_with_retry_reraises_non_rate_limit_immediately(self):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = ValueError("Invalid prompt format")

        with self.assertRaises(ValueError):
            call_gemini_with_retry(mock_model, "Test Prompt", max_retries=5, initial_delay=0.01)

        self.assertEqual(mock_model.generate_content.call_count, 1)

    def test_async_acall_gemini_with_retry(self):
        mock_coro = AsyncMock()
        mock_coro.side_effect = [
            MockResourceExhausted("429 Quota Exceeded"),
            "Async Success",
        ]

        result = asyncio.run(acall_gemini_with_retry(mock_coro, "Async Prompt", max_retries=3, initial_delay=0.01))
        self.assertEqual(result, "Async Success")
        self.assertEqual(mock_coro.call_count, 2)


if __name__ == "__main__":
    unittest.main()
