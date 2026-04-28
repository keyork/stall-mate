# SPDX-License-Identifier: Apache-2.0
"""LLMClient 单元测试 / LLMClient unit tests.

所有测试均使用 mock，不发起真实 API 调用。
All tests use mocks — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import instructor
import pytest

from stall_mate.client import LLMClient
from stall_mate.schema import StallChoice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stall_choice(stall: int = 3) -> StallChoice:
    return StallChoice(
        chosen_stall=stall,
        chain_of_thought="I chose this stall because it is the middle one and feels safest.",
        confidence=0.8,
    )


def _mock_plain_response(text: str = "OK", tokens: int = 10) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = text
    mock_resp.usage.total_tokens = tokens
    return mock_resp


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_params(self):
        client = LLMClient(
            endpoint="http://localhost:3000/v1",
            model="glm-5.1",
            api_key="test-key",
            timeout=30,
        )
        assert client.endpoint == "http://localhost:3000/v1"
        assert client.model == "glm-5.1"
        assert client.api_key == "test-key"
        assert client.timeout == 30
        assert client._openai_client is None
        assert client._mode is None


# ---------------------------------------------------------------------------
# _get_openai_client
# ---------------------------------------------------------------------------


class TestGetOpenAIClient:
    @patch("stall_mate.client.llm_client.OpenAI")
    def test_lazy_init_called_once(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance

        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        result1 = client._get_openai_client()
        result2 = client._get_openai_client()

        assert result1 is mock_instance
        assert result2 is mock_instance
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:3000/v1",
            api_key="unused",
            timeout=60,
        )

    @patch("stall_mate.client.llm_client.OpenAI")
    def test_uses_api_key_when_provided(self, mock_openai_cls):
        client = LLMClient(
            endpoint="http://localhost:3000/v1",
            model="glm-5.1",
            api_key="my-key",
        )
        client._get_openai_client()
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:3000/v1",
            api_key="my-key",
            timeout=60,
        )


# ---------------------------------------------------------------------------
# probe_api
# ---------------------------------------------------------------------------


class TestProbeApi:
    @patch("stall_mate.client.llm_client.instructor.from_openai")
    @patch("stall_mate.client.llm_client.OpenAI")
    def test_tools_mode_succeeds(self, mock_openai_cls, mock_from_openai):
        mock_from_openai.return_value.chat.completions.create.return_value = "OK"
        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        mode = client.probe_api()
        assert mode == "TOOLS"
        assert client._mode == "TOOLS"

    @patch("stall_mate.client.llm_client.instructor.from_openai")
    @patch("stall_mate.client.llm_client.OpenAI")
    def test_fallback_to_json_schema(self, mock_openai_cls, mock_from_openai):
        call_count = {"n": 0}

        def side_effect(client, mode=None):
            mock_inst = MagicMock()
            if mode == instructor.Mode.TOOLS:
                mock_inst.chat.completions.create.side_effect = Exception("TOOLS failed")
            else:
                mock_inst.chat.completions.create.return_value = "OK"
            return mock_inst

        mock_from_openai.side_effect = side_effect
        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        mode = client.probe_api()
        assert mode == "JSON_SCHEMA"
        assert client._mode == "JSON_SCHEMA"

    @patch("stall_mate.client.llm_client.instructor.from_openai")
    @patch("stall_mate.client.llm_client.OpenAI")
    def test_fallback_to_plain_text(self, mock_openai_cls, mock_from_openai):
        mock_inst = MagicMock()
        mock_inst.chat.completions.create.side_effect = Exception("both fail")
        mock_from_openai.return_value = mock_inst

        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        mode = client.probe_api()
        assert mode == "PLAIN_TEXT"
        assert client._mode == "PLAIN_TEXT"


# ---------------------------------------------------------------------------
# query_structured
# ---------------------------------------------------------------------------


class TestQueryStructured:
    @patch("stall_mate.client.llm_client.instructor.from_openai")
    @patch("stall_mate.client.llm_client.OpenAI")
    def test_tools_mode_returns_tuple(self, mock_openai_cls, mock_from_openai):
        expected = _make_stall_choice()
        mock_inst = MagicMock()
        mock_inst.chat.completions.create.return_value = expected
        mock_from_openai.return_value = mock_inst

        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        client._mode = "TOOLS"
        result, raw, tokens, latency = client.query_structured(
            prompt="Pick a stall", system_message="You are helpful"
        )

        assert isinstance(result, StallChoice)
        assert result.chosen_stall == 3
        assert isinstance(raw, str)
        assert isinstance(tokens, int)
        assert isinstance(latency, int)

    @patch("stall_mate.client.llm_client.instructor.from_openai")
    @patch("stall_mate.client.llm_client.OpenAI")
    def test_exception_returns_none(self, mock_openai_cls, mock_from_openai):
        mock_inst = MagicMock()
        mock_inst.chat.completions.create.side_effect = Exception("API error")
        mock_from_openai.return_value = mock_inst

        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        client._mode = "TOOLS"
        result, raw, tokens, latency = client.query_structured(
            prompt="Pick a stall", system_message="You are helpful"
        )

        assert result is None
        assert "API error" in raw
        assert isinstance(latency, int)

    @patch("stall_mate.client.llm_client.OpenAI")
    def test_plain_text_mode_calls_query_plain(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_plain_response(
            text="I pick stall 3", tokens=15
        )
        mock_openai_cls.return_value = mock_client

        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        client._mode = "PLAIN_TEXT"
        result, raw, tokens, latency = client.query_structured(
            prompt="Pick a stall", system_message="You are helpful"
        )

        assert result is None
        assert raw == "I pick stall 3"
        assert tokens == 15
        assert isinstance(latency, int)


# ---------------------------------------------------------------------------
# query_plain
# ---------------------------------------------------------------------------


class TestQueryPlain:
    @patch("stall_mate.client.llm_client.OpenAI")
    def test_returns_tuple(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_plain_response(
            text="Hello", tokens=5
        )
        mock_openai_cls.return_value = mock_client

        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        raw, tokens, latency = client.query_plain(
            prompt="Hi", system_message="You are helpful"
        )

        assert raw == "Hello"
        assert tokens == 5
        assert isinstance(latency, int)
        assert latency >= 0


# ---------------------------------------------------------------------------
# Connection error handling
# ---------------------------------------------------------------------------


class TestConnectionError:
    @patch("stall_mate.client.llm_client.OpenAI")
    def test_connection_error_returns_none(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError(
            "Connection refused"
        )
        mock_openai_cls.return_value = mock_client

        client = LLMClient(endpoint="http://localhost:3000/v1", model="glm-5.1")
        client._mode = "PLAIN_TEXT"
        result, raw, tokens, latency = client.query_structured(
            prompt="Hi", system_message="Be helpful"
        )

        assert result is None
        assert "ConnectionError" in raw
        assert isinstance(latency, int)
