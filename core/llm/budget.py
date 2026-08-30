"""
core/llm/budget.py

Build Phase 26: a hard token-spend ceiling ("سقف صرف صارم" -- a strict
spending cap), directly serving the ORIGINAL three-point directive's
explicit "protect against expensive bills eating profits" wording.
Build Phases 18-20 (this project's own "Cost Efficiency" arc) built
real MEASUREMENT -- core.llm.token_usage.TokenUsage records exactly
what a call cost, honestly, never a fabricated number -- but nothing
before this phase ever stopped a run once measured spend crossed a
line. This module is that missing stop.

Deliberately narrow, mirroring core.agents.guardrails.py's own
documented restraint:

  - No pricing/dollar-cost table lives here, or anywhere in this
    module. core.llm.model_config.py already established the
    project's own precedent for exactly this situation: "never store
    the real value itself (an API key) if it can go stale or drift
    from truth -- store only a reference the caller supplies, and let
    the caller keep it current." A hardcoded 2026 per-model price
    table would be exactly the kind of value that silently goes stale
    the moment a vendor changes pricing, and this project would have
    no way to know. So TokenBudget caps raw TOKEN COUNTS only --
    real, verifiable numbers TokenUsage already reports today, never
    a derived dollar figure this codebase cannot independently keep
    correct. A dollar-denominated cap remains a real, honest future
    option, but only once (or if) a caller-supplied, always-current
    pricing source exists to build it on -- not before.

  - Unlike OutputGuardrailEngine, there is no `enforce=False` default
    mode here. Enforcement is not optional once a TokenBudget is
    actually configured: the whole point of asking for one (per the
    user's own explicit "hard/strict cap" request) is to stop
    execution, not merely to observe -- observation without
    enforcement is exactly what TokenUsage/combine_token_usage()
    already do, one Build Phase earlier, and would make a
    `TokenBudget` a redundant, confusing second name for the same
    thing. The existing "never becomes so strict it can't execute"
    standing constraint is honoured the same way every other optional
    Kernel/AgentExecutionLoop component honours it: `token_budget`
    defaults to `None` everywhere it is threaded, and an unconfigured
    loop/Kernel behaves exactly as it did before this phase. Only a
    caller who explicitly builds a `TokenBudget` and passes it in asks
    for the stricter behaviour.

  - Honest about what "hard cap" can and cannot mean here: token
    counts for a single LLM call are only known AFTER that call
    returns (no provider reports cost before billing it), so this can
    only ever be a REACTIVE check -- it stops any FURTHER step once
    the configured ceiling has already been reached or crossed, but it
    cannot prevent the one call that crosses the ceiling from having
    already been made and already being billed. See
    AgentExecutionLoop.run()'s own docstring (Build Phase 26 section)
    for exactly where this check runs and why that ordering is the
    best available guarantee, not a silently-glossed-over gap.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.llm.token_usage import TokenUsage


@dataclass(frozen=True)
class TokenBudget:
    """
    A hard ceiling on cumulative token spend for one run.

    `max_total_tokens` is compared against a TokenUsage's own
    `total_tokens` (the same real, normalized field
    core.llm.token_usage.TokenUsage already reports and
    combine_token_usage() already sums) -- never a separately
    tracked/derived count of this module's own invention.

    Required (not optional/`None`-defaulted, unlike every other
    optional component this project threads through in this same
    opt-in style): the OPTIONALITY here lives one level up, in
    whether a caller passes a `TokenBudget` at all
    (`token_budget: TokenBudget | None = None` in
    AgentExecutionLoop/Kernel/WorkflowStage) -- a `TokenBudget` that
    exists but caps nothing would be a contradiction in terms, not a
    useful "off" state.
    """

    max_total_tokens: int

    def __post_init__(self) -> None:

        if (
            isinstance(self.max_total_tokens, bool)
            or not isinstance(self.max_total_tokens, int)
        ):
            raise TypeError(
                "TokenBudget.max_total_tokens must be an int."
            )

        if self.max_total_tokens <= 0:
            raise ValueError(
                "TokenBudget.max_total_tokens must be greater than zero."
            )

    def exceeded_by(self, usage: "TokenUsage | None") -> bool:
        """
        `True` once `usage.total_tokens` has reached or passed
        `max_total_tokens`, `False` otherwise.

        `usage=None` ("no usage known yet," e.g. before the first real
        LLM call this run has made, or a decision engine that doesn't
        expose usage at all) always returns `False` -- exactly
        combine_token_usage()'s own "None means unknown, never
        fabricate a violation (or a zero) from missing data"
        precedent, applied here to a budget check instead of a sum.
        """

        if usage is None:
            return False

        if not isinstance(usage, TokenUsage):
            raise TypeError(
                "TokenBudget.exceeded_by() only accepts a TokenUsage "
                "or None."
            )

        return usage.total_tokens >= self.max_total_tokens
