"""
Tests for core.llm.caching_llm_client (Build Phase 20): build_cache_key(),
ResponseCache, and CachingLLMClient.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from core.llm.caching_llm_client import (
    CachingLLMClient,
    ResponseCache,
    build_cache_key,
)

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMMessage, LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.token_usage import TokenUsage


class _CountingLLMClient(LLMClient):
    """A real LLMClient that returns a fixed response and counts every
    real call it actually received -- used to prove a cache hit never
    reaches the wrapped client at all."""

    def __init__(self, response: LLMResponse):
        self._response = response
        self.call_count = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        return self._response


def _request(
    *,
    model: str = "test-model",
    temperature: float | None = 0,
    max_tokens: int | None = 100,
    content: str = "Hello.",
) -> LLMRequest:
    return LLMRequest(
        messages=(LLMMessage(role="user", content=content),),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------
# build_cache_key
# ---------------------------------------------------------------------

def test_build_cache_key_rejects_non_llmrequest():
    with pytest.raises(TypeError, match="LLMRequest"):
        build_cache_key("not a request")


def test_build_cache_key_is_stable_for_structurally_identical_requests():
    first = _request()
    second = _request()  # a separate object, same field values

    assert build_cache_key(first) == build_cache_key(second)


def test_build_cache_key_differs_for_different_model():
    assert build_cache_key(_request(model="a")) != build_cache_key(
        _request(model="b")
    )


def test_build_cache_key_differs_for_different_temperature():
    assert build_cache_key(_request(temperature=0)) != build_cache_key(
        _request(temperature=0.7)
    )


def test_build_cache_key_differs_for_different_max_tokens():
    assert build_cache_key(_request(max_tokens=100)) != build_cache_key(
        _request(max_tokens=200)
    )


def test_build_cache_key_differs_for_different_message_content():
    assert build_cache_key(_request(content="Hello.")) != build_cache_key(
        _request(content="Goodbye.")
    )


def test_build_cache_key_differs_for_different_message_roles():
    first = LLMRequest(
        messages=(LLMMessage(role="user", content="X"),),
        model="m",
        temperature=0,
        max_tokens=10,
    )
    second = LLMRequest(
        messages=(LLMMessage(role="system", content="X"),),
        model="m",
        temperature=0,
        max_tokens=10,
    )

    assert build_cache_key(first) != build_cache_key(second)


# ---------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------

def test_response_cache_rejects_non_integer_max_entries():
    with pytest.raises(TypeError, match="max_entries"):
        ResponseCache(max_entries="256")


def test_response_cache_rejects_bool_max_entries():
    with pytest.raises(TypeError, match="max_entries"):
        ResponseCache(max_entries=True)


def test_response_cache_rejects_non_positive_max_entries():
    with pytest.raises(ValueError, match="max_entries"):
        ResponseCache(max_entries=0)


def test_response_cache_get_returns_none_for_a_missing_key():
    cache = ResponseCache()
    assert cache.get("nonexistent") is None


def test_response_cache_put_rejects_non_llmresponse():
    cache = ResponseCache()
    with pytest.raises(TypeError, match="LLMResponse"):
        cache.put("key", "not a response")


def test_response_cache_put_and_get_round_trip():
    cache = ResponseCache()
    response = LLMResponse(content="Hi", model="m")

    cache.put("key1", response)

    assert cache.get("key1") == response
    assert len(cache) == 1


def test_response_cache_put_overwrites_an_existing_key_without_growing():
    cache = ResponseCache(max_entries=5)
    cache.put("key1", LLMResponse(content="first", model="m"))
    cache.put("key1", LLMResponse(content="second", model="m"))

    assert len(cache) == 1
    assert cache.get("key1").content == "second"


def test_response_cache_evicts_the_least_recently_used_entry_when_full():
    cache = ResponseCache(max_entries=2)
    cache.put("a", LLMResponse(content="A", model="m"))
    cache.put("b", LLMResponse(content="B", model="m"))
    cache.put("c", LLMResponse(content="C", model="m"))  # evicts "a"

    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert len(cache) == 2


def test_response_cache_get_refreshes_recency_and_protects_from_eviction():
    cache = ResponseCache(max_entries=2)
    cache.put("a", LLMResponse(content="A", model="m"))
    cache.put("b", LLMResponse(content="B", model="m"))

    cache.get("a")  # "a" is now most-recently-used; "b" is now oldest

    cache.put("c", LLMResponse(content="C", model="m"))  # evicts "b", not "a"

    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None


# ---------------------------------------------------------------------
# CachingLLMClient
# ---------------------------------------------------------------------

def test_caching_llm_client_implements_llm_client():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    assert isinstance(client, LLMClient)


def test_caching_llm_client_rejects_invalid_wrapped_client():
    with pytest.raises(TypeError, match="LLMClient"):
        CachingLLMClient("not a client")


def test_caching_llm_client_rejects_invalid_cache():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    with pytest.raises(TypeError, match="ResponseCache"):
        CachingLLMClient(wrapped, cache="not a cache")


def test_caching_llm_client_rejects_invalid_request():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    with pytest.raises(TypeError, match="LLMRequest"):
        client.generate("not a request")


def test_caching_llm_client_uses_its_own_default_cache_when_none_supplied():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    assert isinstance(client.cache, ResponseCache)


def test_caching_llm_client_caches_a_temperature_zero_request_and_never_calls_the_wrapped_client_twice():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    first = client.generate(_request(temperature=0))
    second = client.generate(_request(temperature=0))

    assert wrapped.call_count == 1
    assert first.content == second.content == "Hi"


def test_caching_llm_client_does_not_cache_when_temperature_is_none_by_default():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    client.generate(_request(temperature=None))
    client.generate(_request(temperature=None))

    assert wrapped.call_count == 2


def test_caching_llm_client_does_not_cache_when_temperature_is_nonzero_by_default():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    client.generate(_request(temperature=0.7))
    client.generate(_request(temperature=0.7))

    assert wrapped.call_count == 2


def test_caching_llm_client_caches_nondeterministic_requests_when_opted_in():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped, cache_nondeterministic=True)

    client.generate(_request(temperature=None))
    client.generate(_request(temperature=None))

    assert wrapped.call_count == 1


def test_caching_llm_client_cache_miss_reaches_the_wrapped_client_for_different_requests():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    client.generate(_request(temperature=0, content="one"))
    client.generate(_request(temperature=0, content="two"))

    assert wrapped.call_count == 2


def test_caching_llm_client_cache_hit_reports_zero_usage_not_the_original_calls_usage():
    original_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    wrapped = _CountingLLMClient(
        LLMResponse(content="Hi", model="m", usage=original_usage)
    )
    client = CachingLLMClient(wrapped)

    first = client.generate(_request(temperature=0))
    second = client.generate(_request(temperature=0))

    assert first.usage == original_usage  # the real, billed first call
    assert second.usage == TokenUsage(
        prompt_tokens=0, completion_tokens=0, total_tokens=0
    )  # the cache hit: known, explicit zero cost -- never None, never replayed


def test_caching_llm_client_tracks_hits_and_misses():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    assert client.hits == 0
    assert client.misses == 0

    client.generate(_request(temperature=0))  # miss
    client.generate(_request(temperature=0))  # hit
    client.generate(_request(temperature=0))  # hit

    assert client.misses == 1
    assert client.hits == 2


def test_caching_llm_client_never_counts_hits_or_misses_for_uncacheable_requests():
    wrapped = _CountingLLMClient(LLMResponse(content="Hi", model="m"))
    client = CachingLLMClient(wrapped)

    client.generate(_request(temperature=0.5))
    client.generate(_request(temperature=0.5))

    assert client.hits == 0
    assert client.misses == 0


def test_caching_llm_client_shares_a_cache_across_separate_instances():
    shared_cache = ResponseCache()

    wrapped_a = _CountingLLMClient(LLMResponse(content="from A", model="m"))
    wrapped_b = _CountingLLMClient(LLMResponse(content="from B", model="m"))

    client_a = CachingLLMClient(wrapped_a, cache=shared_cache)
    client_b = CachingLLMClient(wrapped_b, cache=shared_cache)

    response_a = client_a.generate(_request(temperature=0, content="same"))
    # client_b never actually calls wrapped_b -- the shared cache
    # already has this exact request cached, from a completely
    # different CachingLLMClient instance.
    response_b = client_b.generate(_request(temperature=0, content="same"))

    assert wrapped_a.call_count == 1
    assert wrapped_b.call_count == 0
    assert response_b.content == response_a.content == "from A"


# ---------------------------------------------------------------------
# Thread safety (Build Phase 21: core.kernel.concurrent_kernel is the
# first caller that can reach a single shared ResponseCache from more
# than one thread at once -- see that module's own top-of-file
# docstring, and ResponseCache's own docstring for what changed here).
# ---------------------------------------------------------------------


def test_response_cache_survives_concurrent_puts_without_exceeding_max_entries():
    cache = ResponseCache(max_entries=50)
    thread_count = 32
    puts_per_thread = 20

    def _hammer(thread_index: int) -> None:
        for i in range(puts_per_thread):
            key = f"thread-{thread_index}-key-{i}"
            cache.put(key, LLMResponse(content=key, model="m"))
            # Interleave reads with writes -- a get() on a key that
            # may or may not exist yet from another thread must never
            # raise or corrupt state, only return a hit or a miss.
            cache.get(key)

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        list(executor.map(_hammer, range(thread_count)))

    # No exception escaped, and the hand-rolled LRU's own invariant
    # (never more than max_entries live at once) held under real
    # concurrent mutation, not just sequential use.
    assert len(cache) <= 50


def test_response_cache_concurrent_gets_never_raise_or_corrupt_order():
    cache = ResponseCache(max_entries=10)

    for i in range(10):
        key = f"key-{i}"
        cache.put(key, LLMResponse(content=key, model="m"))

    def _read_all(_: int) -> int:
        hits = 0
        for i in range(10):
            if cache.get(f"key-{i}") is not None:
                hits += 1
        return hits

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(_read_all, range(64)))

    # Every one of the 10 keys was present before any thread started
    # reading, and nothing ever evicts on a pure get() -- every
    # concurrent reader must see all 10 hits, every time.
    assert all(hit_count == 10 for hit_count in results)
    assert len(cache) == 10


def test_caching_llm_client_under_concurrent_load_hits_the_shared_cache():
    shared_cache = ResponseCache()
    wrapped = _CountingLLMClient(LLMResponse(content="shared", model="m"))
    client = CachingLLMClient(wrapped, cache=shared_cache)

    def _call(_: int) -> str:
        return client.generate(_request(temperature=0, content="same prompt")).content

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(_call, range(64)))

    assert all(result == "shared" for result in results)
    # The real wrapped client may be called more than once if several
    # threads all miss before the first write lands (an accepted,
    # documented cache-aside race -- see CachingLLMClient's own
    # docstring), but it must never be called anywhere near 64 times:
    # the cache is doing real work under real concurrent load.
    assert wrapped.call_count < 64
