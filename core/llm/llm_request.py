from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMMessage:
    """
    A single message sent to an LLM.

    The LLM layer remains provider-independent.
    """

    role: str
    content: str


@dataclass(frozen=True)
class LLMRequest:
    """
    Provider-independent request sent to an LLM client.

    The request contains only the information required
    to ask a model for a response.

    It does not contain:

    - provider-specific API logic
    - API keys
    - network execution logic
    """

    messages: tuple[LLMMessage, ...]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
