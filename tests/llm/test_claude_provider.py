from __future__ import annotations

import pytest

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMMessage, LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.providers.claude_provider import ClaudeProvider


class MockMessages:

    def __init__(self, response_text: str, owner):
        self.response_text = response_text
        self.owner = owner

    def create(self, **kwargs):
        self.owner.last_kwargs = kwargs

        class Content:
            def __init__(self, text):
                self.text = text

        class Response:
            def __init__(self, text):
                self.content = [Content(text)]
                self.model = "claude-test"
                self.stop_reason = "end_turn"

        return Response(self.response_text)


class MockAnthropicClient:

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_kwargs = None
        self.messages = MockMessages(
            response_text,
            self,
        )


def test_claude_provider_implements_llm_client():

    client = MockAnthropicClient(
        response_text="Hello from Claude."
    )

    provider = ClaudeProvider(client=client)

    assert isinstance(provider, LLMClient)


def test_claude_provider_returns_normalized_response():

    client = MockAnthropicClient(
        response_text="Hello from Claude."
    )

    provider = ClaudeProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="Say hello.",
            ),
        ),
        model="claude-test",
        temperature=0.2,
        max_tokens=100,
    )

    response = provider.generate(request)

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello from Claude."
    assert response.model == "claude-test"
    # Normalized from Anthropic's raw "end_turn" into the shared
    # cross-provider vocabulary (see ClaudeProvider._FINISH_REASON_MAP).
    assert response.finish_reason == "stop"


def test_claude_provider_builds_request_correctly():

    client = MockAnthropicClient(
        response_text="OK"
    )

    provider = ClaudeProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="system",
                content="You are an assistant.",
            ),
            LLMMessage(
                role="user",
                content="Hello.",
            ),
        ),
        model="claude-test",
        temperature=0.5,
        max_tokens=200,
    )

    provider.generate(request)

    kwargs = client.last_kwargs

    assert kwargs is not None
    assert kwargs["model"] == "claude-test"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 200

    assert kwargs["system"] == "You are an assistant."

    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": "Hello.",
        }
    ]


def test_claude_provider_rejects_invalid_request():

    client = MockAnthropicClient(
        response_text="OK"
    )

    provider = ClaudeProvider(client=client)

    with pytest.raises(TypeError):
        provider.generate("INVALID_REQUEST")
