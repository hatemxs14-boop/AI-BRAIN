"""
Regression tests for LLM provider normalization gaps between
ClaudeProvider and OpenAIProvider (deferred since Pass 2, addressed
here): differing max_tokens defaults, asymmetric empty-response
handling, an unread OpenAI `message.refusal` field, asymmetric
system-message validation, and unnormalized finish_reason values that
made "the model was cut off" indistinguishable, across providers, from
"the model returned malformed output".
"""
from __future__ import annotations

import pytest

from core.agents.agent_context import AgentContext
from core.agents.llm_decision_engine import LLMDecisionEngine
from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMMessage, LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.providers.claude_provider import ClaudeProvider
from core.llm.providers.openai_provider import OpenAIProvider


# ---------------------------------------------------------------------
# Claude mocks
# ---------------------------------------------------------------------

class _ClaudeBlock:
    def __init__(self, text):
        self.text = text


class _ClaudeResponse:
    def __init__(self, content, model, stop_reason):
        self.content = content
        self.model = model
        self.stop_reason = stop_reason


class _MockClaudeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _MockClaudeClient:
    def __init__(self, response):
        self.messages = _MockClaudeMessages(response)


def _claude_request(**overrides):
    defaults = dict(
        messages=(LLMMessage(role="user", content="Hello."),),
        model="claude-test",
    )
    defaults.update(overrides)
    return LLMRequest(**defaults)


# ---------------------------------------------------------------------
# OpenAI mocks
# ---------------------------------------------------------------------

class _OpenAIMessage:
    def __init__(self, content=None, refusal=None):
        self.content = content
        self.refusal = refusal


class _OpenAIChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _OpenAIResponse:
    def __init__(self, choices, model):
        self.choices = choices
        self.model = model


class _MockOpenAICompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _MockOpenAIChat:
    def __init__(self, response):
        self.completions = _MockOpenAICompletions(response)


class _MockOpenAIClient:
    def __init__(self, response):
        self.chat = _MockOpenAIChat(response)


def _openai_request(**overrides):
    defaults = dict(
        messages=(LLMMessage(role="user", content="Hello."),),
        model="gpt-test",
    )
    defaults.update(overrides)
    return LLMRequest(**defaults)


# ---------------------------------------------------------------------
# Gap 1: max_tokens default parity
# ---------------------------------------------------------------------

def test_claude_provider_defaults_max_tokens_when_omitted():
    response = _ClaudeResponse(
        content=[_ClaudeBlock("ok")],
        model="claude-test",
        stop_reason="end_turn",
    )
    client = _MockClaudeClient(response)
    provider = ClaudeProvider(client=client)

    provider.generate(_claude_request(max_tokens=None))

    assert client.messages.last_kwargs["max_tokens"] == 1024


def test_openai_provider_defaults_max_tokens_when_omitted():
    """
    Previously the OpenAI provider omitted `max_tokens` entirely when
    None, letting the vendor's own per-model default apply --
    inconsistent with Claude, which has always defaulted to 1024.
    """
    response = _OpenAIResponse(
        choices=[_OpenAIChoice(_OpenAIMessage(content="ok"), "stop")],
        model="gpt-test",
    )
    client = _MockOpenAIClient(response)
    provider = OpenAIProvider(client=client)

    provider.generate(_openai_request(max_tokens=None))

    assert client.chat.completions.last_kwargs["max_tokens"] == 1024


def test_both_providers_use_the_same_default_max_tokens():
    assert (
        ClaudeProvider._DEFAULT_MAX_TOKENS
        == OpenAIProvider._DEFAULT_MAX_TOKENS
    )


# ---------------------------------------------------------------------
# Gap 2: consistent, loud failure on genuinely empty responses
# ---------------------------------------------------------------------

def test_claude_provider_raises_on_no_text_content():
    """
    Previously silently returned content="" -- the caller only found
    out several layers later, from a confusing empty-response error
    or JSON-parse failure raised by something else entirely.
    """
    response = _ClaudeResponse(
        content=[],  # no text blocks at all
        model="claude-test",
        stop_reason="end_turn",
    )
    client = _MockClaudeClient(response)
    provider = ClaudeProvider(client=client)

    with pytest.raises(ValueError, match="no text content"):
        provider.generate(_claude_request())


def test_openai_provider_raises_on_no_text_content_without_refusal():
    response = _OpenAIResponse(
        choices=[_OpenAIChoice(_OpenAIMessage(content=None), "stop")],
        model="gpt-test",
    )
    client = _MockOpenAIClient(response)
    provider = OpenAIProvider(client=client)

    with pytest.raises(ValueError, match="no text content"):
        provider.generate(_openai_request())


# ---------------------------------------------------------------------
# Gap 3: OpenAI's `message.refusal` was never read
# ---------------------------------------------------------------------

def test_openai_provider_surfaces_refusal_text():
    response = _OpenAIResponse(
        choices=[
            _OpenAIChoice(
                _OpenAIMessage(
                    content=None,
                    refusal="I can't help with that request.",
                ),
                "stop",
            )
        ],
        model="gpt-test",
    )
    client = _MockOpenAIClient(response)
    provider = OpenAIProvider(client=client)

    with pytest.raises(ValueError, match="I can't help with that request"):
        provider.generate(_openai_request())


# ---------------------------------------------------------------------
# Gap 4: OpenAI never validated for at least one non-system message
# ---------------------------------------------------------------------

def test_openai_provider_rejects_all_system_message_request():
    client = _MockOpenAIClient(
        _OpenAIResponse(choices=[], model="gpt-test")
    )
    provider = OpenAIProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(role="system", content="You are an assistant."),
        ),
        model="gpt-test",
    )

    with pytest.raises(ValueError, match="non-system message"):
        provider.generate(request)


def test_claude_provider_still_rejects_all_system_message_request():
    """
    Non-regression: Claude already had this check; confirm it still
    behaves the same way after this pass's edits.
    """
    client = _MockClaudeClient(
        _ClaudeResponse(content=[], model="claude-test", stop_reason=None)
    )
    provider = ClaudeProvider(client=client)

    request = LLMRequest(
        messages=(
            LLMMessage(role="system", content="You are an assistant."),
        ),
        model="claude-test",
    )

    with pytest.raises(ValueError, match="non-system message"):
        provider.generate(request)


# ---------------------------------------------------------------------
# Gap 5: finish_reason normalization
# ---------------------------------------------------------------------

def test_claude_finish_reason_normalization():
    cases = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_use",
        "something_anthropic_adds_later": "other",
    }

    for raw, expected in cases.items():
        response = _ClaudeResponse(
            content=[_ClaudeBlock("ok")],
            model="claude-test",
            stop_reason=raw,
        )
        provider = ClaudeProvider(client=_MockClaudeClient(response))

        result = provider.generate(_claude_request())

        assert result.finish_reason == expected, (
            f"raw stop_reason={raw!r} normalized to "
            f"{result.finish_reason!r}, expected {expected!r}"
        )


def test_openai_finish_reason_normalization():
    cases = {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_use",
        "content_filter": "content_filter",
        "something_openai_adds_later": "other",
    }

    for raw, expected in cases.items():
        response = _OpenAIResponse(
            choices=[
                _OpenAIChoice(_OpenAIMessage(content="ok"), raw)
            ],
            model="gpt-test",
        )
        provider = OpenAIProvider(client=_MockOpenAIClient(response))

        result = provider.generate(_openai_request())

        assert result.finish_reason == expected, (
            f"raw finish_reason={raw!r} normalized to "
            f"{result.finish_reason!r}, expected {expected!r}"
        )


def test_raw_vendor_response_is_still_available_after_normalization():
    """
    Normalizing finish_reason must not lose the exact vendor value --
    it must remain readable off `.raw` for a caller that needs it.
    """
    response = _OpenAIResponse(
        choices=[_OpenAIChoice(_OpenAIMessage(content="ok"), "length")],
        model="gpt-test",
    )
    provider = OpenAIProvider(client=_MockOpenAIClient(response))

    result = provider.generate(_openai_request())

    assert result.finish_reason == "length"
    assert result.raw.choices[0].finish_reason == "length"


# ---------------------------------------------------------------------
# LLMDecisionEngine: a truncated response must fail with a clear,
# specific diagnosis instead of an opaque JSON-parse error.
# ---------------------------------------------------------------------

class _TruncatingClient(LLMClient):
    def generate(self, request):
        return LLMResponse(
            content='{"action_type": "COMPLETE", "reason": "incomple',
            model="test-model",
            finish_reason="length",
        )


class _NormalClient(LLMClient):
    def generate(self, request):
        return LLMResponse(
            content='{"action_type": "COMPLETE", "reason": "done"}',
            model="test-model",
            finish_reason="stop",
        )


def test_decision_engine_raises_clear_error_on_truncated_response():
    engine = LLMDecisionEngine(_TruncatingClient(), model="test-model")
    context = AgentContext(task="Do something")

    with pytest.raises(ValueError, match="truncated"):
        engine.decide(context)


def test_decision_engine_unaffected_by_normal_stop_reason():
    """
    Non-regression: only finish_reason == "length" triggers the
    truncation guard -- a normal "stop" response must parse as usual.
    """
    engine = LLMDecisionEngine(_NormalClient(), model="test-model")
    context = AgentContext(task="Do something")

    action = engine.decide(context)

    assert action.reason == "done"
