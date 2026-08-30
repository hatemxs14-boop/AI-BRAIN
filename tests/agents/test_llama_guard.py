"""
Tests for core.agents.llama_guard (Build Phase 29): LlamaGuardVerdict's
own validation, the LlamaGuardClient ABC contract, and
OllamaLlamaGuardClient's real request-building/response-parsing logic
against a fake HTTP layer (`http_post=`) -- never a live Ollama server
or real Llama Guard model weights, exactly mirroring tests/tools/
implementations/test_web_search_tool.py's own identical convention for
Serper.dev. Every test here runs for real in any environment, since
`requests` (unlike `voyageai`) is already an established project
dependency and genuinely installed in this sandbox.

Confidence-gate wiring into OutputGuardrailEngine itself is covered
separately in tests/agents/test_guardrails.py, which is where the
"only ONE real classify() call regardless of how many MEDIUM findings
are present" and "degrades gracefully on failure" behaviors are
actually exercised end-to-end.
"""
from __future__ import annotations

import pytest
import requests

from core.agents.llama_guard import (
    DEFAULT_LLAMA_GUARD_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    LlamaGuardClient,
    LlamaGuardError,
    LlamaGuardVerdict,
    OllamaLlamaGuardClient,
)


class _FakeResponse:

    def __init__(self, *, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload configured")
        return self._payload


# ---------------------------------------------------------------------
# LlamaGuardVerdict validation
# ---------------------------------------------------------------------


def test_llama_guard_verdict_accepts_valid_safe_data():
    verdict = LlamaGuardVerdict(is_safe=True)
    assert verdict.is_safe is True
    assert verdict.categories == ()


def test_llama_guard_verdict_accepts_valid_unsafe_data():
    verdict = LlamaGuardVerdict(is_safe=False, categories=("S1", "S9"))
    assert verdict.categories == ("S1", "S9")


def test_llama_guard_verdict_rejects_non_bool_is_safe():
    with pytest.raises(TypeError, match="is_safe"):
        LlamaGuardVerdict(is_safe="yes")


def test_llama_guard_verdict_rejects_non_tuple_categories():
    with pytest.raises(TypeError, match="categories"):
        LlamaGuardVerdict(is_safe=False, categories=["S1"])


def test_llama_guard_verdict_rejects_empty_string_in_categories():
    with pytest.raises(TypeError, match="categories"):
        LlamaGuardVerdict(is_safe=False, categories=("S1", "   "))


# ---------------------------------------------------------------------
# LlamaGuardClient -- ABC contract
# ---------------------------------------------------------------------


def test_llama_guard_client_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LlamaGuardClient()


# ---------------------------------------------------------------------
# OllamaLlamaGuardClient -- construction validation
# ---------------------------------------------------------------------


def test_ollama_client_uses_documented_defaults():
    client = OllamaLlamaGuardClient()
    assert client.base_url == DEFAULT_OLLAMA_BASE_URL
    assert client.model == DEFAULT_LLAMA_GUARD_MODEL


def test_ollama_client_strips_trailing_slash_from_base_url():
    client = OllamaLlamaGuardClient(base_url="http://localhost:11434/")
    assert client.base_url == "http://localhost:11434"


def test_ollama_client_rejects_empty_base_url():
    with pytest.raises(ValueError, match="base_url"):
        OllamaLlamaGuardClient(base_url="   ")


def test_ollama_client_rejects_empty_model():
    with pytest.raises(ValueError, match="model"):
        OllamaLlamaGuardClient(model="")


def test_ollama_client_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout"):
        OllamaLlamaGuardClient(timeout=0)


def test_ollama_client_rejects_bool_timeout():
    with pytest.raises(ValueError, match="timeout"):
        OllamaLlamaGuardClient(timeout=True)


# ---------------------------------------------------------------------
# classify() -- input validation
# ---------------------------------------------------------------------


def test_classify_rejects_empty_text():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload={"response": "safe"})
    )

    with pytest.raises(ValueError, match="text"):
        client.classify("   ")


def test_classify_rejects_non_string_text():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload={"response": "safe"})
    )

    with pytest.raises(ValueError, match="text"):
        client.classify(12345)


# ---------------------------------------------------------------------
# classify() -- real request building (against the fake HTTP layer)
# ---------------------------------------------------------------------


def test_classify_posts_to_the_generate_endpoint_with_expected_payload():
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(payload={"response": "safe"})

    client = OllamaLlamaGuardClient(
        base_url="http://localhost:11434",
        model="llama-guard3",
        timeout=15.0,
        http_post=fake_post,
    )

    client.classify("Please help me with something.")

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["kwargs"]["json"] == {
        "model": "llama-guard3",
        "prompt": "Please help me with something.",
        "stream": False,
    }
    assert captured["kwargs"]["timeout"] == 15.0


# ---------------------------------------------------------------------
# classify() -- response parsing (happy path)
# ---------------------------------------------------------------------


def test_classify_parses_safe_response():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload={"response": "safe"})
    )

    verdict = client.classify("hello")

    assert verdict == LlamaGuardVerdict(is_safe=True, categories=())


def test_classify_parses_unsafe_response_with_categories():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(
            payload={"response": "unsafe\nS1,S9"}
        )
    )

    verdict = client.classify("hello")

    assert verdict == LlamaGuardVerdict(is_safe=False, categories=("S1", "S9"))


def test_classify_parses_unsafe_response_without_a_category_line():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload={"response": "unsafe"})
    )

    verdict = client.classify("hello")

    assert verdict == LlamaGuardVerdict(is_safe=False, categories=())


def test_classify_is_case_insensitive_on_the_verdict_line():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload={"response": "SAFE"})
    )

    assert client.classify("hello").is_safe is True


def test_classify_tolerates_surrounding_whitespace_and_blank_lines():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(
            payload={"response": "\n  unsafe  \n\nS6\n"}
        )
    )

    verdict = client.classify("hello")

    assert verdict == LlamaGuardVerdict(is_safe=False, categories=("S6",))


# ---------------------------------------------------------------------
# classify() -- error paths
# ---------------------------------------------------------------------


def test_classify_wraps_request_exceptions():
    def failing_post(*a, **k):
        raise requests.ConnectionError("connection refused")

    client = OllamaLlamaGuardClient(http_post=failing_post)

    with pytest.raises(LlamaGuardError, match="request .* failed"):
        client.classify("hello")


def test_classify_raises_on_non_200_status():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(
            status_code=500, text="internal error"
        )
    )

    with pytest.raises(LlamaGuardError, match="status=500"):
        client.classify("hello")


def test_classify_raises_on_invalid_json():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload=None)
    )

    with pytest.raises(LlamaGuardError, match="not valid JSON"):
        client.classify("hello")


def test_classify_raises_on_non_dict_json_payload():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload=["not", "a", "dict"])
    )

    with pytest.raises(LlamaGuardError, match="unexpected response shape"):
        client.classify("hello")


def test_classify_raises_when_response_field_is_missing():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload={})
    )

    with pytest.raises(LlamaGuardError, match="no usable 'response' field"):
        client.classify("hello")


def test_classify_raises_when_response_field_is_blank():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(payload={"response": "   "})
    )

    with pytest.raises(LlamaGuardError, match="no usable 'response' field"):
        client.classify("hello")


def test_classify_raises_on_unrecognized_verdict_line():
    client = OllamaLlamaGuardClient(
        http_post=lambda *a, **k: _FakeResponse(
            payload={"response": "maybe unsafe?"}
        )
    )

    with pytest.raises(LlamaGuardError, match="Unrecognized Llama Guard output"):
        client.classify("hello")
