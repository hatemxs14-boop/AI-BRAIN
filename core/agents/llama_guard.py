"""
core/agents/llama_guard.py

Build Phase 29: a real, self-hosted Llama Guard CONFIDENCE GATE layered
on top of core/agents/guardrails.py's existing regex-based
OutputGuardrailEngine -- not a replacement for it.

The design comes directly from the ECC GitHub skills-repo research
pass (`regex-vs-llm-structured-text`), refined by the user's own
explicit correction of this project's dependency philosophy during
Build Phase 28: the standing constraint was never "minimize dependency
count," it is "minimize real financial/token cost." Running an LLM
call on EVERY guardrail check would violate that real-cost priority
for no real benefit -- the three existing regex/keyword rules in
guardrails.py already resolve the overwhelming majority of cases with
zero marginal cost and zero latency. So this module does the opposite
of "replace the cheap heuristic with an LLM": it uses the heuristic's
OWN existing severity signal to decide when a second opinion is even
worth asking for.

Concretely: `OutputGuardrailEngine`'s three rules already produce two
tiers of confidence on their own. `credential_leak` and the escalated
form of `injection_compliance` are HIGH-severity, deterministic shape/
keyword matches -- there is nothing ambiguous about them, and asking
an LLM to double-check "does this look like an OpenAI API key" would
be pure waste. But the MEDIUM-severity findings (topic_drift's crude
whole-task word-overlap check, and injection_compliance's
un-corroborated variant) are exactly the genuinely ambiguous,
false-positive-prone cases this module's own docstring already
describes as "a heuristic tripwire, not a semantic verdict." THOSE are
the findings worth spending one real classification call on.

Provider choice: **self-hosted, via Ollama** -- chosen because the
user, immediately after being shown that even a paid Llama Guard API
call would cost a negligible ~$0.18-0.20 per million tokens, said
plainly: "نحن من سنستضيف" (we will host it ourselves). Ollama serves
Meta's own Llama Guard 3 model locally, over a simple HTTP API, for
literally zero marginal cost per call -- only the one-time cost of
running Ollama itself, which the user already committed to. This also
means this module needs no new pip dependency at all: it talks to
Ollama's REST API with the `requests` library this project already
depends on and already uses for exactly this kind of "real HTTP call
to a local/external service" job (see core/tools/implementations/
web_search_tool.py's own identical `http_post=` injection-point
convention, mirrored here for the same reason -- so a test can fake
the HTTP layer without a live Ollama server or the real vendor model
weights, exactly like that module's own tests never make a real
Serper.dev call).

`LlamaGuardClient` is a provider-independent ABC (one abstract
`classify(text) -> LlamaGuardVerdict` method) so a future phase could
add a different backend (a paid API, a different self-hosted server)
without touching guardrails.py's own wiring -- the same one-layer-over
shape core.llm.llm_client.LLMClient and core.llm.embeddings.
EmbeddingClient already established for generation and embeddings
respectively, applied here to safety classification.

Honest scope, worth knowing: this module does NOT implement retries,
batching, or response caching -- a real embeddings-style batched call
per MEDIUM-severity finding set is made (see OutputGuardrailEngine's
own confidence-gate wiring: at most ONE classify() call per evaluate(),
regardless of how many MEDIUM findings are present, mirroring
MemoryStore.search_semantic()'s own "never one call per record"
discipline). A genuine Ollama-server failure (not running, wrong
model, network error, malformed response) is caught by the CALLER
(OutputGuardrailEngine) and degrades gracefully to the original,
regex-only findings -- this module itself simply raises a clear
LlamaGuardError for every one of those cases rather than silently
returning a fabricated verdict.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import requests


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Meta's own Llama Guard 3 8B, as published for Ollama
# (https://ollama.com/library/llama-guard3). Never hardcoded anywhere
# else in this project -- callers that want a different tag (e.g. the
# larger 12B "llama-guard3:12b", or a future Llama Guard 4 once it has
# an official Ollama tag) pass `model=` explicitly, the same "never
# store a value that can go stale" precedent core/llm/model_config.py
# and core/llm/model_tier.py already established.
DEFAULT_LLAMA_GUARD_MODEL = "llama-guard3"

DEFAULT_LLAMA_GUARD_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class LlamaGuardVerdict:
    """
    One real classification result from a LlamaGuardClient.

    `is_safe`: Llama Guard's own top-line safe/unsafe judgment.
    `categories`: Llama Guard's own MLCommons-style hazard category
    codes (e.g. "S1" for violent crimes, "S9" for indiscriminate
    weapons) when `is_safe` is False -- empty when safe, and also
    empty (never fabricated) if an unsafe response's own category line
    could not be parsed for some reason.
    """

    is_safe: bool
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:

        if not isinstance(self.is_safe, bool):
            raise TypeError("is_safe must be a boolean.")

        if not isinstance(self.categories, tuple) or not all(
            isinstance(category, str) and category.strip()
            for category in self.categories
        ):
            raise TypeError(
                "categories must be a tuple of non-empty strings."
            )


class LlamaGuardError(RuntimeError):
    """
    Raised when a real Llama Guard call fails or returns a response
    this module cannot make sense of -- a service/network/format
    problem, never a caller-input mistake (see the ValueError/TypeError
    raised directly by classify() for those). Deliberately a
    RuntimeError subclass, not a ValueError, since -- unlike
    core.llm.embeddings.EmbeddingConfigError, which always signals a
    fixable configuration mistake made before any network call -- every
    case this covers happens only once a real request was actually
    attempted (the server wasn't reachable, it wasn't running the
    expected model, its response didn't parse) and is exactly the class
    of failure OutputGuardrailEngine's own confidence-gate wiring is
    designed to catch and gracefully degrade from.
    """


class LlamaGuardClient(ABC):
    """
    Provider-independent interface for classifying one piece of text
    as safe/unsafe -- the same one-layer-over role
    core.llm.llm_client.LLMClient plays for generation and
    core.llm.embeddings.EmbeddingClient plays for embeddings, applied
    here to safety classification.

    This layer does not:

    - decide agent actions
    - execute tools
    - authorize operations
    - access the Security Layer
    - contain provider-specific business logic
    """

    @abstractmethod
    def classify(self, text: str) -> LlamaGuardVerdict:
        """
        Classify `text` as safe or unsafe. Raises ValueError for an
        empty/non-string `text`, and a provider-specific error (e.g.
        LlamaGuardError for OllamaLlamaGuardClient) for any real
        request/response failure -- never silently returns a
        fabricated verdict.
        """
        raise NotImplementedError


class OllamaLlamaGuardClient(LlamaGuardClient):
    """
    Real Llama Guard classification via a self-hosted Ollama server's
    REST API (`POST {base_url}/api/generate`) -- see
    https://ollama.com/library/llama-guard3 and
    https://github.com/ollama/ollama/blob/main/docs/api.md.

    `http_post` is an injection point for tests, exactly mirroring
    core/tools/implementations/web_search_tool.py's own
    `create_serper_web_search_executor(http_post=...)` convention: it
    defaults to `requests.post`, but a test supplies a fake to exercise
    this client's real request-building/response-parsing logic without
    a live Ollama server or the real Llama Guard model weights.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_LLAMA_GUARD_MODEL,
        timeout: float = DEFAULT_LLAMA_GUARD_TIMEOUT_SECONDS,
        http_post: Callable[..., Any] | None = None,
    ) -> None:

        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string.")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._post = http_post if http_post is not None else requests.post

    def classify(self, text: str) -> LlamaGuardVerdict:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")

        try:
            response = self._post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": text,
                    "stream": False,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LlamaGuardError(
                f"Ollama request to {self.base_url} failed: {exc}"
            ) from exc

        status_code = getattr(response, "status_code", None)

        if status_code != 200:
            body_preview = str(getattr(response, "text", ""))[:500]
            raise LlamaGuardError(
                "Ollama returned a non-200 response "
                f"(status={status_code!r}): {body_preview!r}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise LlamaGuardError(
                "Ollama returned a response that is not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise LlamaGuardError(
                "Ollama returned an unexpected response shape: "
                f"{type(payload).__name__}."
            )

        raw_output = payload.get("response")

        if not isinstance(raw_output, str) or not raw_output.strip():
            raise LlamaGuardError(
                "Ollama's /api/generate response carries no usable "
                "'response' field."
            )

        return _parse_llama_guard_output(raw_output)


def _parse_llama_guard_output(raw_output: str) -> LlamaGuardVerdict:
    """
    Parse Llama Guard's own standard, documented output format: the
    single word "safe", or "unsafe" followed by a second line of
    comma-separated MLCommons hazard category codes (e.g.
    "unsafe\\nS1,S9"). Raises LlamaGuardError for anything else --
    never silently guesses a verdict from unrecognized text.
    """

    lines = [
        line.strip()
        for line in raw_output.strip().splitlines()
        if line.strip()
    ]

    if not lines:
        raise LlamaGuardError("Llama Guard returned an empty output.")

    verdict_line = lines[0].lower()

    if verdict_line == "safe":
        return LlamaGuardVerdict(is_safe=True, categories=())

    if verdict_line == "unsafe":
        categories: tuple[str, ...] = ()

        if len(lines) > 1:
            categories = tuple(
                category.strip()
                for category in lines[1].split(",")
                if category.strip()
            )

        return LlamaGuardVerdict(is_safe=False, categories=categories)

    raise LlamaGuardError(
        "Unrecognized Llama Guard output (expected 'safe' or "
        f"'unsafe' as the first line): {raw_output!r}"
    )
