"""
Test Google AI Provider

This module contains tests for the Google AI provider implementation.
"""

import os
import unittest
from unittest import mock

from ..src.models.base_model import Message, Role
from ..src.models.google_provider import GoogleAIProvider


class TestGoogleAIProvider(unittest.TestCase):
    """Test cases for the Google AI provider."""

    def setUp(self):
        """Set up the test environment."""
        # Skip tests if dependencies are not available
        try:
            import google.generativeai  # noqa
        except ImportError:
            self.skipTest("Google Generative AI package not available")

        # Mock API key for testing
        self.api_key = "test_api_key"
        self.model = "gemini-pro"

        # Create a provider with mocked credentials
        with mock.patch("google.generativeai.configure"):
            self.provider = GoogleAIProvider(
                api_key=self.api_key,
                model=self.model,
            )

    def test_initialization(self):
        """Test provider initialization."""
        self.assertEqual(self.provider.model, self.model)
        self.assertEqual(self.provider.max_retries, 3)  # Default value

    def test_convert_messages(self):
        """Test message conversion."""
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
            Message(role=Role.FUNCTION, content='{"result": "data"}', name="get_data"),
        ]

        google_messages = self.provider._convert_messages(messages)

        # Check that we have the right number of messages
        self.assertEqual(len(google_messages), 4)

        # Check that the system message is converted to a user message with special formatting
        self.assertEqual(google_messages[0]["role"], "user")
        self.assertTrue(google_messages[0]["parts"][0]["text"].startswith("System:"))

        # Check that user message is preserved
        self.assertEqual(google_messages[1]["role"], "user")
        self.assertEqual(google_messages[1]["parts"][0]["text"], "Hello!")

        # Check that assistant message is converted to model role
        self.assertEqual(google_messages[2]["role"], "model")
        self.assertEqual(google_messages[2]["parts"][0]["text"], "Hi there!")

        # Check that function message is properly formatted
        self.assertEqual(google_messages[3]["role"], "user")
        self.assertTrue("Function result from 'get_data'" in google_messages[3]["parts"][0]["text"])
        self.assertTrue("```json" in google_messages[3]["parts"][0]["text"])

    def test_count_tokens(self):
        """Test token counting."""
        text = "Hello, world! This is a test."
        # The method is currently an approximation so just check it returns a positive number
        self.assertGreater(self.provider.count_tokens(text), 0)

    def test_max_context_length(self):
        """Test max context length retrieval."""
        # Check known model
        self.assertEqual(self.provider.get_max_context_length(), 32768)  # gemini-pro

        # Check unknown model with fallback
        with mock.patch.object(self.provider, "model", "unknown-model"):
            self.assertEqual(self.provider.get_max_context_length(), 32768)  # default fallback

    def test_supports_functions(self):
        """Test function calling support check."""
        # Check known model
        self.assertTrue(self.provider.supports_functions())  # gemini-pro

        # Check vision model (doesn't support functions)
        with mock.patch.object(self.provider, "model", "gemini-pro-vision"):
            self.assertFalse(self.provider.supports_functions())

        # Check unknown model with fallback
        with mock.patch.object(self.provider, "model", "unknown-model"):
            self.assertFalse(self.provider.supports_functions())  # default fallback

    @mock.patch("google.generativeai.GenerativeModel")
    def test_generate_method_calls(self, mock_generative_model):
        """Test the generate method calls the API correctly."""
        # Setup mocks
        mock_model_instance = mock.MagicMock()
        mock_generative_model.return_value = mock_model_instance
        mock_response = mock.MagicMock()
        mock_response.text = "This is a test response."
        mock_model_instance.generate_content.return_value = mock_response

        # Call generate
        messages = [Message(role=Role.USER, content="Test query")]
        self.provider.generate(messages, temperature=0.5, max_tokens=100)

        # Verify GenerativeModel was instantiated with the right model
        mock_generative_model.assert_called_once_with(model_name=self.model)

        # Verify generate_content was called with converted messages and config
        mock_model_instance.generate_content.assert_called_once()
        args, kwargs = mock_model_instance.generate_content.call_args
        self.assertEqual(len(args), 1)  # First arg should be the messages

        # Check generation config
        self.assertEqual(kwargs["generation_config"]["temperature"], 0.5)
        self.assertEqual(kwargs["generation_config"]["max_output_tokens"], 100)


if __name__ == "__main__":
    unittest.main() 