from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.agents.llm_decision_engine import (
    LLMDecisionEngine,
)

from core.agents.research_agent import (
    DEFAULT_DOCUMENTS_ROOT,
    DEFAULT_FINDINGS_ROOT,
    DEFAULT_PERMISSIONS_PATH,
    build_research_agent,
)

from core.agents.writer_agent import (
    DEFAULT_REPORTS_ROOT,
    build_writer_agent,
)

from core.agents.reviewer_agent import (
    build_reviewer_agent,
)

from core.kernel.kernel import (
    AgentRegistration,
    Kernel,
    NormalizedTask,
)

from core.llm.llm_client import (
    LLMClient,
)

from core.orchestration.orchestration_engine import (
    OrchestrationEngine,
)

from core.policies.policy_engine import (
    PolicyEngine,
)


# ---------------------------------------------------------------------
# Convenience wiring: a Kernel with research_agent, writer_agent, and
# (as of Build Phase 11) reviewer_agent all registered, the same way
# core/agents/research_agent.py's own build_research_agent(),
# core/agents/writer_agent.py's own build_writer_agent(), and
# core/agents/reviewer_agent.py's own build_reviewer_agent() are
# convenience wirings of each agent's own tool/security stack.
#
# Until Build Phase 8, research_agent was this project's only
# registered agent, so its `can_handle` was `_always_handles`
# (accepted every task) -- Kernel._classify() had nothing to actually
# classify between. With three agents now registered, all three
# predicates below are real: a finite, hand-maintained keyword
# vocabulary per agent, in the same spirit as RiskEngine's own
# keyword-heuristic classification (Pass 3 finding I) -- not a real
# NLU classifier (no such subsystem exists in this project), but a
# genuine, testable discriminator rather than the previous
# accept-everything placeholder. A task matching more than one
# vocabulary (or none) is handled exactly as
# Kernel._classify()/_select_agent() already document: every match is
# collected, and the first in registration order is selected
# (research_agent first, then writer_agent, then reviewer_agent, so an
# earlier-registered agent wins a genuine tie) -- STRATEGY SELECTION
# remains "run one matching agent", unchanged by this phase; only
# which agents can match at all is new. See Kernel._classify()'s own
# docstring in core/kernel/kernel.py for the unchanged selection
# mechanism itself.
#
# Matching is whole-word, not plain substring: a first draft of this
# module matched keywords with a plain `keyword in text` check, and
# tests/kernel/test_kernel_writer_agent_integration.py's own
# "Summarize finding.md." case caught it immediately -- research_
# agent's "find" keyword is a substring of "finding"/"findings", which
# is exactly the vocabulary writer_agent's own domain (reading
# research *findings*) uses constantly, so plain substring matching
# misrouted a writer_agent task to research_agent on nearly every real
# phrasing. Every keyword below is matched with `\bkeyword\b` instead,
# so "find" no longer matches inside "finding" (no word boundary
# between "find" and the following "ing"), while multi-word phrases
# like "read document" still match as a whole phrase.
#
# Build Phase 11 added a third agent, reviewer_agent, and caught a
# second vocabulary-overlap problem of the same kind before it ever
# shipped: reviewer_agent's whole domain is verifying a *report*, so
# almost any realistic review task ("review the report", "verify the
# report's claims") contains the word "report" -- which
# _WRITER_AGENT_KEYWORDS previously listed as a standalone trigger.
# Since research_agent is registered first, writer_agent second, and
# reviewer_agent third, a task matching both writer_agent's and
# reviewer_agent's vocabulary would have been a "genuine tie" that
# always resolved to writer_agent (registration order), silently
# starving reviewer_agent of almost every realistic phrasing of its
# own job. Checked directly against the test suite before removing
# it: no test relied on the bare word "report" alone triggering
# writer_agent (every existing writer_agent test task already
# contains "draft"/"summarize"/"write" too), so "report" was removed
# from _WRITER_AGENT_KEYWORDS entirely -- writer_agent is now reached
# only by its own drafting/publishing verbs, and reviewer_agent's own
# verification verbs no longer collide with it.
# ---------------------------------------------------------------------


_RESEARCH_AGENT_KEYWORDS: tuple[str, ...] = (
    "research",
    "search",
    "find",
    "investigate",
    "look up",
    "gather",
    "read the document",
    "read document",
    "read the webpage",
    "read webpage",
)

_WRITER_AGENT_KEYWORDS: tuple[str, ...] = (
    "write",
    "draft",
    "summarize",
    "summarise",
    "summary",
    "compose",
)

_REVIEWER_AGENT_KEYWORDS: tuple[str, ...] = (
    "review",
    "verify",
    "audit",
    "validate",
    "critique",
    "fact-check",
    "fact check",
    "double-check",
    "double check",
    "cross-check",
    "cross check",
)


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """
    True if `text` (already lowercased) contains any of `keywords` as
    a whole word/phrase -- `\\bkeyword\\b`, not a plain substring
    check. See this module's own docstring for why plain substring
    matching was wrong (it let research_agent's "find" keyword match
    inside "finding"/"findings").
    """

    return any(
        re.search(r"\b" + re.escape(keyword) + r"\b", text)
        for keyword in keywords
    )


def _research_agent_handles(normalized: NormalizedTask) -> bool:
    """
    Real (v1) capability predicate for research_agent: matches when
    the normalized task text contains any of a finite, hand-maintained
    set of research/evidence-gathering keywords, as a whole word or
    phrase (see _contains_keyword). See this module's own docstring
    for why this is a deliberate keyword heuristic, not a real NLU
    classifier.
    """

    return _contains_keyword(normalized.text.lower(), _RESEARCH_AGENT_KEYWORDS)


def _writer_agent_handles(normalized: NormalizedTask) -> bool:
    """
    Real (v1) capability predicate for writer_agent: matches when the
    normalized task text contains any of a finite, hand-maintained set
    of writing/reporting keywords, as a whole word or phrase (see
    _contains_keyword). See this module's own docstring for why this
    is a deliberate keyword heuristic, not a real NLU classifier.
    """

    return _contains_keyword(normalized.text.lower(), _WRITER_AGENT_KEYWORDS)


def _reviewer_agent_handles(normalized: NormalizedTask) -> bool:
    """
    Real (v1) capability predicate for reviewer_agent: matches when
    the normalized task text contains any of a finite, hand-maintained
    set of independent-verification keywords, as a whole word or
    phrase (see _contains_keyword). See this module's own docstring
    for why this is a deliberate keyword heuristic, not a real NLU
    classifier, and for the "report" vocabulary-overlap problem this
    agent's addition surfaced and fixed on _WRITER_AGENT_KEYWORDS.
    """

    return _contains_keyword(normalized.text.lower(), _REVIEWER_AGENT_KEYWORDS)


def build_default_kernel(
    *,
    llm_client_factory: Callable[[], LLMClient] | None = None,
    decision_engine_factory: Callable[[], AgentDecisionEngine] | None = None,
    documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT,
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    serper_api_key: str | None = None,
    permissions_path: str | Path = DEFAULT_PERMISSIONS_PATH,
    audit_log_path: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    orchestration_engine: OrchestrationEngine | None = None,
    max_recovery_attempts: int = 1,
    policy_engine: PolicyEngine | None = None,
) -> Kernel:
    """
    Build a Kernel with research_agent, writer_agent, and
    reviewer_agent all already registered.

    Provide exactly one of `llm_client_factory` (a zero-argument
    callable returning a fresh LLMClient; wrapped in a fresh
    LLMDecisionEngine per Kernel.run() call using `model`/
    `temperature`/`max_tokens`) or `decision_engine_factory` (a
    zero-argument callable returning a fresh AgentDecisionEngine
    directly -- e.g. a DeterministicDecisionEngine for testing, or a
    caller-configured LLMDecisionEngine). The same factory is shared by
    all three agents -- it only encapsulates which model/client is
    used, never per-agent state (each Kernel.run() attempt calls it
    fresh via the selected AgentRegistration's own
    build_decision_engine, per AgentRegistration's own docstring).

    A *factory* is required, not an instance, for the same reason
    AgentRegistration.build_agent is a factory: a decision engine (or
    the LLMClient it wraps) may carry per-run state, and reusing one
    instance across unrelated Kernel.run() calls is a footgun this
    signature avoids by construction.

    `documents_root`/`findings_root`/`serper_api_key` are passed
    straight through to build_research_agent() -- see that function's
    own docstring. `findings_root`/`reports_root` are passed straight
    through to build_writer_agent() -- `findings_root` is deliberately
    the SAME parameter research_agent writes into, so writer_agent
    reads exactly what research_agent has actually persisted (see
    core/agents/writer_agent.py's own docstring for this pipeline
    link). `findings_root`/`reports_root` are likewise passed straight
    through to build_reviewer_agent(), so reviewer_agent reads exactly
    the same findings writer_agent read and exactly what writer_agent
    has actually published. `permissions_path`/`audit_log_path` are
    shared by all three agents' security stacks.

    `policy_engine` is passed straight through to Kernel() -- see its
    own docstring (core/kernel/kernel.py). Defaults to a fresh
    PolicyEngine() when not supplied.
    """

    if decision_engine_factory is None:

        if llm_client_factory is None:
            raise ValueError(
                "Either llm_client_factory or decision_engine_factory "
                "must be provided."
            )

        if not callable(llm_client_factory):
            raise TypeError(
                "llm_client_factory must be callable."
            )

        def decision_engine_factory() -> AgentDecisionEngine:
            return LLMDecisionEngine(
                llm_client_factory(),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    elif not callable(decision_engine_factory):
        raise TypeError(
            "decision_engine_factory must be callable."
        )

    def build_research():
        return build_research_agent(
            documents_root=documents_root,
            findings_root=findings_root,
            serper_api_key=serper_api_key,
            permissions_path=permissions_path,
            audit_log_path=audit_log_path,
        )

    def build_writer():
        return build_writer_agent(
            findings_root=findings_root,
            reports_root=reports_root,
            permissions_path=permissions_path,
            audit_log_path=audit_log_path,
        )

    def build_reviewer():
        return build_reviewer_agent(
            findings_root=findings_root,
            reports_root=reports_root,
            permissions_path=permissions_path,
            audit_log_path=audit_log_path,
        )

    kernel = Kernel(
        orchestration_engine=orchestration_engine,
        max_recovery_attempts=max_recovery_attempts,
        policy_engine=policy_engine,
    )

    kernel.register_agent(
        AgentRegistration(
            subject="research_agent",
            description=(
                "Conducts structured, read-only research and "
                "persists findings when explicitly approved. See "
                "core/agents/RESEARCH_AGENT.md."
            ),
            can_handle=_research_agent_handles,
            build_agent=build_research,
            build_decision_engine=decision_engine_factory,
        )
    )

    kernel.register_agent(
        AgentRegistration(
            subject="writer_agent",
            description=(
                "Synthesizes already-persisted research findings "
                "into a written report and publishes it when "
                "explicitly approved. See "
                "core/agents/WRITER_AGENT.md."
            ),
            can_handle=_writer_agent_handles,
            build_agent=build_writer,
            build_decision_engine=decision_engine_factory,
        )
    )

    kernel.register_agent(
        AgentRegistration(
            subject="reviewer_agent",
            description=(
                "Independently verifies an already-published report "
                "against the research findings it claims to be based "
                "on. See core/agents/REVIEWER_AGENT.md."
            ),
            can_handle=_reviewer_agent_handles,
            build_agent=build_reviewer,
            build_decision_engine=decision_engine_factory,
        )
    )

    return kernel
