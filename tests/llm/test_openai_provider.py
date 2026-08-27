from __future__ import annotations

import pytest

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMMessage, LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.providers.openai_provider import OpenAIProvider


class MockCompletions:

    def __init__(self, response_text: str, owner):
        self.response_text = response_text
        self.owner = owner

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
            def __init__(self, content):
                self.choices = [Choice(content)]
                self.model = "gpt-test"

        return Response(self.response_text)


class MockChat:

    def __init__(self, response_text: str, owner):
        self.completions = MockCompletions(
            response_text,
            owner,
        )


class MockOpenAIClient:

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_kwargs = None
        self.chat = MockChat(
            response_text,
            self,
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
