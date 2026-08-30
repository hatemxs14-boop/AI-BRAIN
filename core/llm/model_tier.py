from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------
# Build Phase 27 -- per-task model-tier routing.
#
# The problem this solves: core/llm/model_config.py (Build Phase 18)
# configures exactly ONE model per Kernel -- switching models is a
# one-file, one-value edit ("absolute model-switching flexibility"),
# but every task still pays that one model's price, even a trivial
# one-line "search for X" request that a much cheaper model could
# answer just as well as the full configured model. This was one of
# two genuine, previously-unidentified gaps the ECC research pass's
# `cost-aware-llm-pipeline` finding surfaced in the already-closed Cost
# Efficiency arc (Build Phases 18-20): (a) a hard budget ceiling --
# built as Build Phase 26's TokenBudget -- and (b) this module,
# per-task model-tier routing: automatically send simple tasks to a
# cheaper model and reserve the full configured model for tasks that
# actually look complex.
#
# Deliberately NOT a real NLU complexity classifier -- there is no
# such subsystem anywhere in this project, and fabricating one here
# would be exactly the kind of "looks done but isn't" shortcut this
# project's standing discipline forbids. Instead, ModelTierRouter uses
# the same kind of finite, hand-maintained, explainable heuristic
# core/kernel/default_kernel.py's own _RESEARCH_AGENT_KEYWORDS/
# _WRITER_AGENT_KEYWORDS/_REVIEWER_AGENT_KEYWORDS already use for agent
# routing: a task is routed to the cheaper "simple" tier unless (a) its
# text contains one of a small set of complexity-signal keywords/
# phrases ("comprehensive", "step by step", "compare", "analyze", ...),
# matched as a whole word/phrase exactly like that module's own
# `\bphrase\b` convention (never a plain substring check -- see this
# module's own `_contains_keyword_phrase` docstring for why), or (b)
# its word count exceeds a configurable threshold (`simple_max_words`,
# default 12) -- a short task is more likely a single, simple ask; a
# long one is more likely carrying multiple clauses/constraints a
# cheaper model is more likely to get wrong. Both signals are honestly
# just proxies for complexity, not a guarantee -- exactly like
# RiskEngine's own keyword-heuristic classification (Pass 3 finding I)
# is an honest proxy for real risk, not a guarantee either.
#
# Deliberately does NOT hardcode any real model name (no "haiku" or
# "gpt-4o-mini" string anywhere in this module) -- the caller supplies
# both `simple_model` and `complex_model` explicitly, exactly like
# core/llm/model_config.py's own ModelConfig never embeds a value that
# can go stale (there: a real API key; here: which model name is
# "cheap" today, which changes constantly as providers release new
# tiers). This module only ever routes to whichever two model names the
# caller configured, never invents or assumes one.
#
# Threading: `Kernel(model_tier_router=...)` (core/kernel/kernel.py)
# and `WorkflowStage.model_tier_router` (core/orchestration/
# multi_agent_workflow.py) both apply this at the exact moment a fresh
# decision engine is built for a specific task's text -- Kernel._plan()
# and Kernel.resume() for a single agent, _make_stage_node() for a
# multi-agent workflow stage -- by overwriting that decision engine's
# own `.model` attribute with this call's ModelTierDecision.model,
# right before it is ever used. This deliberately targets
# LLMDecisionEngine specifically (via an isinstance check at the call
# site, not a duck-typed getattr/hasattr): `.model` is already exactly
# the per-request model override LLMDecisionEngine's own
# `_build_request()` reads fresh on every `decide()` call (see that
# class's own docstring), so overwriting it once, right after
# construction and before the loop ever calls `decide()`, is not a
# hack -- it is using that field for precisely what it was already
# built to do. Any other decision engine (DeterministicDecisionEngine,
# or any test double) has no `.model` attribute at all and is simply
# never touched -- routing is a pure no-op for it, never an error.
#
# Deliberately NOT surfaced as a new inspectable field on KernelResult/
# MultiAgentWorkflowResult in this v1: which model a request used was
# never inspectable Kernel-result data before this phase either (the
# base `model`/`temperature`/`max_tokens` passed to build_default_
# kernel() aren't surfaced there), so this keeps exactly that same,
# already-established scope -- a real, honestly-documented v1 boundary,
# not a silently narrower one. A future phase could add this as
# observability if the project's audit trail ever needs it.
# ---------------------------------------------------------------------


DEFAULT_COMPLEXITY_KEYWORDS: tuple[str, ...] = (
    "comprehensive",
    "detailed",
    "in-depth",
    "in depth",
    "thorough",
    "step by step",
    "step-by-step",
    "multi-step",
    "multiple",
    "several",
    "compare",
    "comparison",
    "analyze",
    "analyse",
    "analysis",
    "and then",
    "after that",
)
"""
A small, finite, hand-maintained set of whole-word/phrase complexity
signals -- see this module's own top-of-file docstring for why this is
a deliberate heuristic, not real NLU. Passed as ModelTierRouter's own
default `complexity_keywords`; a caller may override it entirely with
its own tuple to fit a different domain's vocabulary.
"""

DEFAULT_SIMPLE_MAX_WORDS = 12
"""
Default `ModelTierRouter.simple_max_words` -- a task at or under this
many words is treated as simple (routed to the cheaper tier) unless it
also contains a complexity-signal keyword/phrase; a longer task is
treated as complex regardless of vocabulary. See this module's own
top-of-file docstring for the reasoning.
"""


def _contains_keyword_phrase(text: str, keywords: tuple[str, ...]) -> bool:
    """
    True if `text` (already lowercased) contains any of `keywords` as a
    whole word/phrase (`\\bphrase\\b`), never a plain substring check.

    Deliberately a private, self-contained duplicate of
    core.kernel.kernel.contains_keyword_phrase's own word-boundary
    convention, not an import of it: core/llm/ is a lower-level module
    that core/kernel/kernel.py itself imports from (exactly like
    core/llm/budget.py's TokenBudget, Build Phase 26) -- importing
    kernel.py's own helper from here would be a circular import. See
    that helper's own docstring (and core/kernel/default_kernel.py's
    module docstring) for the original "research_agent's 'find' keyword
    matching inside writer_agent's 'finding(s)'" bug this exact
    word-boundary convention was written to avoid; the same class of
    bug applies just as much here (e.g. "analyze" should not match
    inside some unrelated longer word).
    """

    for phrase in keywords:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text):
            return True

    return False


@dataclass(frozen=True)
class ModelTierDecision:
    """
    The result of one `ModelTierRouter.route()` call: which tier was
    chosen, which model name that tier maps to, and a short, honest,
    human-readable explanation of why -- so a caller inspecting this
    (e.g. in a test, or a future logging hook) never has to re-derive
    the reasoning from scratch.
    """

    tier: str
    model: str
    reason: str


@dataclass(frozen=True)
class ModelTierRouter:
    """
    Routes one task's text to either a cheaper "simple" model or the
    full "complex" model, using the heuristic documented in this
    module's own top-of-file docstring.

    Both `simple_model` and `complex_model` are required, non-empty
    model-name strings the caller supplies directly -- this module
    never hardcodes or guesses either one (see this module's own
    top-of-file docstring for why).
    """

    simple_model: str
    complex_model: str
    complexity_keywords: tuple[str, ...] = DEFAULT_COMPLEXITY_KEYWORDS
    simple_max_words: int = DEFAULT_SIMPLE_MAX_WORDS

    def __post_init__(self) -> None:

        if (
            not isinstance(self.simple_model, str)
            or not self.simple_model.strip()
        ):
            raise ValueError(
                "ModelTierRouter.simple_model must be a non-empty "
                "string."
            )

        if (
            not isinstance(self.complex_model, str)
            or not self.complex_model.strip()
        ):
            raise ValueError(
                "ModelTierRouter.complex_model must be a non-empty "
                "string."
            )

        if not isinstance(self.complexity_keywords, tuple) or not all(
            isinstance(keyword, str) and keyword.strip()
            for keyword in self.complexity_keywords
        ):
            raise TypeError(
                "ModelTierRouter.complexity_keywords must be a tuple "
                "of non-empty strings."
            )

        if (
            isinstance(self.simple_max_words, bool)
            or not isinstance(self.simple_max_words, int)
        ):
            raise TypeError(
                "ModelTierRouter.simple_max_words must be an int."
            )

        if self.simple_max_words <= 0:
            raise ValueError(
                "ModelTierRouter.simple_max_words must be greater "
                "than zero."
            )

    def route(self, task_text: str) -> ModelTierDecision:
        """
        Decide which model tier `task_text` belongs to.

        Raises TypeError/ValueError for a non-string or empty
        `task_text` -- mirroring Kernel._normalize()'s own validation,
        since every real caller (Kernel._plan(), Kernel.resume(),
        _make_stage_node()) only ever calls this with an
        already-normalized, non-empty task string.
        """

        if not isinstance(task_text, str):
            raise TypeError(
                "ModelTierRouter.route() requires task_text to be a "
                "string."
            )

        stripped = task_text.strip()

        if not stripped:
            raise ValueError(
                "ModelTierRouter.route() requires a non-empty "
                "task_text."
            )

        lowered = stripped.lower()

        if _contains_keyword_phrase(lowered, self.complexity_keywords):
            return ModelTierDecision(
                tier="complex",
                model=self.complex_model,
                reason=(
                    "Task text contains a complexity-signal keyword "
                    "or phrase from ModelTierRouter.complexity_"
                    "keywords."
                ),
            )

        word_count = len(stripped.split())

        if word_count > self.simple_max_words:
            return ModelTierDecision(
                tier="complex",
                model=self.complex_model,
                reason=(
                    f"Task text is {word_count} word(s), exceeding "
                    f"simple_max_words={self.simple_max_words}."
                ),
            )

        return ModelTierDecision(
            tier="simple",
            model=self.simple_model,
            reason=(
                f"Task text is {word_count} word(s) (at or under "
                f"simple_max_words={self.simple_max_words}) and "
                "contains no complexity-signal keyword or phrase."
            ),
        )
