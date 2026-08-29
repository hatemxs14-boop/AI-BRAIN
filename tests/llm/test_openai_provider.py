from __future__ import annotations

import pytest

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMMessage, LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.providers.openai_provider import OpenAIProvider
from core.llm.token_usage import TokenUsage


class MockCompletions:

    def __init__(self, response_text: str, owner, usage=None):
        self.response_text = response_text
        self.owner = owner
        self.usage = usage

    def create(self, **kwargs):
        self.owner.last_kwargs = kwargs

        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)
                self.finish_reason = "stop"

        class Response:
            def __init__(self, content, usage):
                self.choices = [Choice(content)]
                self.model = "gpt-test"
                if usage is not None:
                    self.usage = usage

        return Response(self.response_text, self.usage)


class MockChat:

    def __init__(self, response_text: str, owner, usage=None):
        self.completions = MockCompletions(
            response_text,
            owner,
            usage,
        )


class MockOpenAIClient:

    def __init__(self, response_text: str, usage=None):
        self.response_text = response_text
        self.last_kwargs = None
        self.chat = MockChat(
            response_text,
            self,
            usage,
        )


def test_openai_provider_implements_llm_client():

    client = MockOpenAIClient(
        response_text="Hello from OpenAI."
    )

    provider = OpenAIProvider(client=client)

    assert isinstance(provider, LLMClient)


def test_openai_provider_returns_normalized_response():

    client = MockOpenAIClient(
        response_text="Hello from OpenAI."
    )

    provider = OpenAIProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(
                role="user",
                content="Say hello.",
            ),
        ),
        model="gpt-test",
        temperature=0.2,
        max_tokens=100,
    )

    response = provider.generate(request)

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello from OpenAI."
    assert response.model == "gpt-test"
    assert response.finish_reason == "stop"


def test_openai_provider_builds_request_correctly():

    client = MockOpenAIClient(
        response_text="OK"
    )

    provider = OpenAIProvider(client=client)

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
        model="gpt-test",
        temperature=0.5,
        max_tokens=200,
    )

    provider.generate(request)

    kwargs = client.last_kwargs

    assert kwargs is not None
    assert kwargs["model"] == "gpt-test"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 200

    assert kwargs["messages"] == [
        {
            "role": "system",
            "content": "You are an assistant.",
        },
        {
            "role": "user",
            "content": "Hello.",
        },
    ]


def test_openai_provider_rejects_invalid_request():

    client = MockOpenAIClient(
        response_text="OK"
    )

    provider = OpenAIProvider(client=client)

    with pytest.raises(TypeError):
        provider.generate("INVALID_REQUEST")


def test_openai_provider_normalizes_usage_from_real_response_shape():
    # OpenAI's own response.usage already computes total_tokens itself
    # -- a plain object with all three attributes, matching that real
    # shape without needing the actual openai package installed.
    class _OpenAIUsage:
        def __init__(self, prompt_tokens, completion_tokens, total_tokens):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens

    client = MockOpenAIClient(
        response_text="Hello from OpenAI.",
        usage=_OpenAIUsage(
            prompt_tokens=9,
            completion_tokens=4,
            total_tokens=13,
        ),
    )

    provider = OpenAIProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(role="user", content="Say hello."),
        ),
        model="gpt-test",
    )

    response = provider.generate(request)

    assert response.usage == TokenUsage(
        prompt_tokens=9,
        completion_tokens=4,
        total_tokens=13,
    )


def test_openai_provider_usage_is_none_when_response_has_no_usage():
    client = MockOpenAIClient(
        response_text="Hello from OpenAI."
    )

    provider = OpenAIProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(role="user", content="Say hello."),
        ),
        model="gpt-test",
    )

    response = provider.generate(request)

    assert response.usage is None


def test_openai_provider_usage_is_none_when_usage_shape_is_incomplete():
    class _IncompleteUsage:
        def __init__(self, prompt_tokens, completion_tokens):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            # total_tokens deliberately missing.

    client = MockOpenAIClient(
        response_text="Hello from OpenAI.",
        usage=_IncompleteUsage(prompt_tokens=9, completion_tokens=4),
    )

    provider = OpenAIProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(role="user", content="Say hello."),
        ),
        model="gpt-test",
    )

    response = provider.generate(request)

    assert response.usage is None
