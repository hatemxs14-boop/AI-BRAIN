from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """
    Normalized token-consumption for a single real LLM call
    (Build Phase 19 -- the first step of the user's own "Cost
    Efficiency" priority: you cannot manage or cap what you cannot
    first measure).

    Populated straight from each vendor SDK's own response --
    Anthropic's `response.usage.input_tokens`/`.output_tokens`
    (Anthropic's own response carries no total, so ClaudeProvider
    computes one), or OpenAI's `response.usage.prompt_tokens`/
    `.completion_tokens`/`.total_tokens` directly. Never fabricated:
    a provider either reports real usage for a call (all three fields
    populated together) or it doesn't (the whole LLMResponse.usage is
    `None`) -- there is deliberately no "0 tokens" standing in for
    "we don't know."
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:

        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ):

            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"TokenUsage.{field_name} must be an int."
                )

            if value < 0:
                raise ValueError(
                    f"TokenUsage.{field_name} must be zero or greater."
                )


def combine_token_usage(
    *usages: "TokenUsage | None",
) -> "TokenUsage | None":
    """
    Sum any number of TokenUsage values, treating a missing (`None`)
    usage as "unknown," never as zero.

    Used everywhere a single Kernel-level result can reflect more than
    one real LLM call -- a RECOVER IF NEEDED retry re-runs a fresh
    decision engine from scratch (Kernel._execute_once's own
    docstring: "always a full, fresh attempt, never a resume"), a
    Workflow chains several agents each making their own calls, and an
    auto-triggered independent verification (Build Phase 12) is a real
    extra agent run. Each of those attempts/steps/runs only knows its
    own usage; this is how their real, separately-incurred costs are
    added back together into one honest total.

    Returns `None` only when EVERY given usage is `None` (nothing
    known at all). As soon as at least one real TokenUsage is present,
    this returns the real partial sum rather than silently discarding
    it -- a partial, honest count is more useful for real cost
    tracking than a fabricated "0 tokens used" would be.
    """

    real = [usage for usage in usages if usage is not None]

    if not real:
        return None

    for usage in real:
        if not isinstance(usage, TokenUsage):
            raise TypeError(
                "combine_token_usage() only accepts TokenUsage or None."
            )

    return TokenUsage(
        prompt_tokens=sum(usage.prompt_tokens for usage in real),
        completion_tokens=sum(usage.completion_tokens for usage in real),
        total_tokens=sum(usage.total_tokens for usage in real),
    )
