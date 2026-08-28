from __future__ import annotations

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
# Convenience wiring: a Kernel with research_agent registered, the
# same way core/agents/research_agent.py's own build_research_agent()
# is a convenience wiring of the tool/security stack.
#
# research_agent is still the only agent this project has, so
# `_always_handles` below is deliberately trivial (accepts every
# task). A real multi-agent classifier belongs here once a second
# agent exists to classify against -- see Kernel._classify()'s own
# docstring in core/kernel/kernel.py.
# ---------------------------------------------------------------------


def _always_handles(normalized: NormalizedTask) -> bool:
    """
    research_agent is currently the only registered agent, so it
    accepts every task. Replace with a real capability predicate
    (keyword/domain matching, or an LLM-based router) once a second
    agent is registered and tasks must actually be classified between
    them.
    """

    return True


def build_default_kernel(
    *,
    llm_client_factory: Callable[[], LLMClient] | None = None,
    decision_engine_factory: Callable[[], AgentDecisionEngine] | None = None,
    documents_root: str | Path = DEFAULT_DOCUMENTS_ROOT,
    findings_root: str | Path = DEFAULT_FINDINGS_ROOT,
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
    Build a Kernel with research_agent already registered.

    Provide exactly one of `llm_client_factory` (a zero-argument
    callable returning a fresh LLMClient; wrapped in a fresh
    LLMDecisionEngine per Kernel.run() call using `model`/
    `temperature`/`max_tokens`) or `decision_engine_factory` (a
    zero-argument callable returning a fresh AgentDecisionEngine
    directly -- e.g. a DeterministicDecisionEngine for testing, or a
    caller-configured LLMDecisionEngine).

    A *factory* is required, not an instance, for the same reason
    AgentRegistration.build_agent is a factory: a decision engine (or
    the LLMClient it wraps) may carry per-run state, and reusing one
    instance across unrelated Kernel.run() calls is a footgun this
    signature avoids by construction.

    `documents_root`/`findings_root`/`serper_api_key`/
    `permissions_path`/`audit_log_path` are passed straight through to
    build_research_agent() -- see that function's own docstring.

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

    def build_agent():
        return build_research_agent(
            documents_root=documents_root,
            findings_root=findings_root,
            serper_api_key=serper_api_key,
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
            can_handle=_always_handles,
            build_agent=build_agent,
            build_decision_engine=decision_engine_factory,
        )
    )

    return kernel
