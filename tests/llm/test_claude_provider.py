from __future__ import annotations

import pytest

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMMessage, LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.providers.claude_provider import ClaudeProvider
from core.llm.token_usage import TokenUsage


class MockMessages:

    def __init__(self, response_text: str, owner, usage=None):
        self.response_text = response_text
        self.owner = owner
        self.usage = usage

    def create(self, **kwargs):
        self.owner.last_kwargs = kwargs

        class Content:
            def __init__(self, text):
                self.text = text

        class Response:
            def __init__(self, text, usage):
                self.content = [Content(text)]
                self.model = "claude-test"
                self.stop_reason = "end_turn"
                if usage is not None:
                    self.usage = usage

        return Response(self.response_text, self.usage)


class MockAnthropicClient:

    def __init__(self, response_text: str, usage=None):
        self.response_text = response_text
        self.last_kwargs = None
        self.messages = MockMessages(
            response_text,
            self,
            usage,
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


def test_claude_provider_normalizes_usage_from_real_response_shape():
    # Anthropic's own response.usage carries input_tokens/output_tokens
    # (no total of its own) -- a plain object with just those two
    # attributes, matching that real shape without needing the actual
    # anthropic package installed.
    class _AnthropicUsage:
        def __init__(self, input_tokens, output_tokens):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    client = MockAnthropicClient(
        response_text="Hello from Claude.",
        usage=_AnthropicUsage(input_tokens=12, output_tokens=7),
    )

    provider = ClaudeProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(role="user", content="Say hello."),
        ),
        model="claude-test",
    )

    response = provider.generate(request)

    assert response.usage == TokenUsage(
        prompt_tokens=12,
        completion_tokens=7,
        total_tokens=19,
    )


def test_claude_provider_usage_is_none_when_response_has_no_usage():
    client = MockAnthropicClient(
        response_text="Hello from Claude."
    )

    provider = ClaudeProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(role="user", content="Say hello."),
        ),
        model="claude-test",
    )

    response = provider.generate(request)

    assert response.usage is None


def test_claude_provider_usage_is_none_when_usage_shape_is_incomplete():
    class _IncompleteUsage:
        def __init__(self, input_tokens):
            self.input_tokens = input_tokens
            # output_tokens deliberately missing.

    client = MockAnthropicClient(
        response_text="Hello from Claude.",
        usage=_IncompleteUsage(input_tokens=12),
    )

    provider = ClaudeProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(role="user", content="Say hello."),
        ),
        model="claude-test",
    )

    response = provider.generate(request)

    assert response.usage is None
