from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from core.llm.llm_client import (
    LLMClient,
)

from core.llm.llm_request import (
    LLMRequest,
)

from core.llm.llm_response import (
    LLMResponse,
)

from core.llm.token_usage import (
    TokenUsage,
)


# A cache hit has a real, known cost of exactly zero tokens -- see
# CachingLLMClient's own docstring for why this must be an explicit
# zero TokenUsage, never None (which would misreport a known zero as
# unknown) and never the original call's own usage (which would
# double-count a cost that was not actually paid a second time).
_ZERO_USAGE = TokenUsage(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
)


def build_cache_key(request: LLMRequest) -> str:
    """
    Deterministic cache key for `request` (Build Phase 20): every
    field that actually determines what a real LLM call would produce
    -- model, temperature, max_tokens, and every message's role/
    content, in order -- reduced to a stable SHA-256 hex digest, so
    two structurally identical LLMRequest instances (even different
    Python objects, built by different code paths) always produce the
    exact same key, and any difference in a single message's content,
    the model, or either sampling parameter always produces a
    different one.

    Deliberately uses every field LLMRequest has today (see that
    dataclass's own docstring: no provider-specific logic, no API
    keys, no network state) -- there is nothing else that could
    silently make two "identical" requests actually different.
    """

    if not isinstance(request, LLMRequest):
        raise TypeError("request must be an LLMRequest.")

    payload = {
        "model": request.model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


class ResponseCache:
    """
    A minimal, dependency-free, in-memory, least-recently-used cache
    of LLMResponse keyed by build_cache_key()'s own string key (Build
    Phase 20).

    Real, honestly-scoped v1, per this project's standing minimize-
    external-dependencies constraint: in-memory only (a real process
    restart starts with a cold cache -- no file-backed or persistent
    option exists here, unlike core.memory.memory_store.MemoryStore's
    own append-only file), bounded by `max_entries` via a plain,
    hand-rolled LRU (an ordered list of keys -- no external caching
    library), and process-local (no cross-process or cross-machine
    sharing; a real multi-process deployment would need a shared
    backend, deliberately not built here). Not thread-safe -- this
    project has no concurrent/parallel request handling yet (still an
    open cost-efficiency priority; see the project's own baseline
    doc), so this has never needed to be.
    """

    def __init__(
        self,
        *,
        max_entries: int = 256,
    ) -> None:

        if not isinstance(max_entries, int) or isinstance(max_entries, bool):
            raise TypeError("max_entries must be an integer.")

        if max_entries <= 0:
            raise ValueError("max_entries must be a positive integer.")

        self._max_entries = max_entries
        self._store: dict[str, LLMResponse] = {}
        # Oldest-first list of keys, used purely to track recency for
        # eviction -- a plain list is fine at this cache's intended
        # scale (max_entries defaults to 256; a real deployment
        # needing far more than that should not be relying on an
        # in-memory, single-process cache in the first place, per this
        # class's own scope above).
        self._order: list[str] = []

    def get(self, key: str) -> LLMResponse | None:
        """
        Returns the cached LLMResponse for `key`, or `None` on a
        miss. A hit refreshes `key` as most-recently-used.
        """

        response = self._store.get(key)

        if response is not None:
            self._order.remove(key)
            self._order.append(key)

        return response

    def put(self, key: str, response: LLMResponse) -> None:
        """
        Stores `response` under `key`, evicting the single least-
        recently-used entry first if this insert would otherwise
        exceed `max_entries`.
        """

        if not isinstance(response, LLMResponse):
            raise TypeError("response must be an LLMResponse.")

        if key in self._store:
            self._order.remove(key)

        elif len(self._store) >= self._max_entries:
            oldest_key = self._order.pop(0)
            del self._store[oldest_key]

        self._store[key] = response
        self._order.append(key)

    def __len__(self) -> int:
        return len(self._store)


class CachingLLMClient(LLMClient):
    """
    Wraps a real LLMClient with ResponseCache (Build Phase 20) -- the
    first of this project's cost-efficiency priorities to actually
    reduce real LLM spend, rather than only measure it the way Build
    Phase 19's token-usage tracking does. Built with zero new external
    dependencies, per the project's own standing constraint.

    Deliberately conservative about WHEN a cached response is reused,
    since an LLM call is only truly a pure function of its input when
    sampling is deterministic:

    - By default (`cache_nondeterministic=False`), a request is only
      ever served from -- or written to -- the cache when
      `request.temperature == 0` exactly. `None` (meaning "the
      provider's own default", which is NOT guaranteed to be 0 --
      ClaudeProvider/OpenAIProvider both omit the temperature kwarg
      entirely when it's None, leaving the vendor SDK's own default in
      effect) and any other nonzero value are both treated as
      genuinely non-deterministic and therefore never cached: caching
      a single sample of a creative/varied response and always
      replaying it back would silently defeat the very reason a
      caller asked for nonzero temperature in the first place.
    - Passing `cache_nondeterministic=True` opts out of that caution
      for a caller who has decided the savings are worth it even at
      nonzero/unset temperature (e.g. a prompt whose output is
      deterministic-in-practice, where an occasional different
      phrasing wouldn't matter). Available, but never the default.

    On a cache HIT, the cached LLMResponse is returned with its
    `usage` field replaced by a real, explicit all-zero TokenUsage --
    never `None`, and never the original call's own usage. A cache hit
    has a real, known cost of exactly zero tokens; Build Phase 19's
    own token-usage stack (LLMDecisionEngine.total_usage and
    everything built on top of it) sums whatever real values it is
    given, so silently replaying the ORIGINAL call's usage would
    double-count a cost that was not actually paid a second time,
    while `None` would misreport a known zero as unknown. This is also
    what makes this cache's savings directly visible: a Kernel run
    whose later steps hit cache reports a lower KernelResult.
    token_usage than an otherwise-identical run that never did.

    `hits`/`misses` are plain, always-real running counts kept on this
    one instance, for direct inspection. Not yet threaded any further
    up the stack (e.g. onto KernelResult, the way `token_usage` is) --
    real future work if a caller needs it, not silently claimed here.

    Deliberately NOT built here, per this project's own "narrow,
    honest v1" standard and its standing minimize-external-
    dependencies constraint: any persistence across process restarts,
    any cross-process/shared cache backend, and any time-based expiry
    (TTL) -- a request whose meaning could go stale over time (e.g.
    one that embeds "the current date") is the caller's own
    responsibility to keep out of what it lets this class cache, since
    this class has no way to infer that from the request text alone.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        cache: ResponseCache | None = None,
        cache_nondeterministic: bool = False,
    ) -> None:

        if not isinstance(client, LLMClient):
            raise TypeError("client must implement LLMClient.")

        if cache is not None and not isinstance(cache, ResponseCache):
            raise TypeError("cache must be a ResponseCache or None.")

        self.client = client
        self.cache = cache if cache is not None else ResponseCache()
        self.cache_nondeterministic = cache_nondeterministic
        self.hits = 0
        self.misses = 0

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest.")

        cacheable = (
            self.cache_nondeterministic
            or request.temperature == 0
        )

        if cacheable:

            key = build_cache_key(request)
            cached = self.cache.get(key)

            if cached is not None:
                self.hits += 1
                return replace(cached, usage=_ZERO_USAGE)

        response = self.client.generate(request)

        if cacheable and isinstance(response, LLMResponse):
            self.misses += 1
            self.cache.put(key, response)

        return response
