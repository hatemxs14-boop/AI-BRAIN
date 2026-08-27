from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """
    Provider-independent response returned by an LLM client.

    The response contains normalized model output and basic
    metadata without exposing provider-specific objects.
    """

    content: str
    model: str | None = None
    finish_reason: str | None = None
    raw: Any = None
