from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.llm.token_usage import (
    TokenUsage,
)


@dataclass(frozen=True)
class LLMResponse:
    """
    Provider-independent response returned by an LLM client.

    The response contains normalized model output and basic
    metadata without exposing provider-specific objects.

    `usage` (Build Phase 19) is this call's real, normalized token
    consumption -- see TokenUsage's own docstring. `None` when the
    provider didn't report any (never a fabricated zero). Defaults to
    `None` so every existing caller/test constructing an LLMResponse
    without `usage=` is completely unaffected.
    """

    content: str
    model: str | None = None
    finish_reason: str | None = None
    raw: Any = None
    usage: TokenUsage | None = None
