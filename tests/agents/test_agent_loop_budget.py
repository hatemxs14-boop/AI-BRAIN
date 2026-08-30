"""
Integration tests for AgentExecutionLoop's Build Phase 26
`token_budget` wiring (core/llm/budget.py, core/agents/agent_loop.py).

Uses the same "scripted decision engine exposing a duck-typed
total_usage attribute" style tests/agents/test_agent_loop_token_usage.py
already established, extended so `total_usage` can change from one
decide() call to the next -- exactly like a real LLMDecisionEngine
accumulating usage across multiple calls within one run -- so these
tests can exercise the budget check tripping mid-run, not only on the
very first step.
"""
from __future__ import annotations

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.agent_loop import AgentExecutionLoop
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.llm.budget import TokenBudget
from core.llm.token_usage import TokenUsage

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


def _build_agent(subject: str = "research_agent") -> AgentCore:
    """Same real, LOW-risk, auto-allowed "web_search" tool fixture
    tests/agents/test_agent_loop_guardrails.py already uses -- subject
    defaults to "research_agent" because that is one of the few
    subjects core/security/schemas/permissions.json actually grants
    web_search to; an arbitrary subject would be DENIED, not SUCCESS,
    on a real tool call."""

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="web_search",
            name="Web Search",
            purpose="Search the public web.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "string"},
            permissions=(f"{subject}:web_search:search:public_web",),
            resource="web_search",
            action="search",
            scope="public_web",
            risk_level="LOW",
            error_handling={
                "retryable": True,
                "max_retries": 2,
                "on_failure": "Surface the search error to the agent.",
            },
        )
    )

    security = SecurityDecisionPoint(PERMISSIONS_FILE)
    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(
        tool_id="web_search",
        executor=lambda query: f"RESULT: {query}",
    )

    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject=subject,
        name="Test Agent",
        purpose="A minimal agent used only to exercise the token budget.",
    )

    return AgentCore(identity=identity, tools=interface)


class _UsageAccumulatingEngine(AgentDecisionEngine):
    """
    Returns each action in `steps`, in order, one per `decide()` call --
    and after each call, sets `self.total_usage` to that step's own
    `usage_after`, exactly like a real LLMDecisionEngine's own running
    total after each fresh LLM call. `steps` is a list of
    (AgentAction, TokenUsage) pairs.
    """

    def __init__(self, steps: list[tuple[AgentAction, TokenUsage]]) -> None:
        self._steps = list(steps)
        self._index = 0
        self.total_usage: TokenUsage | None = None

    def decide(self, context: AgentContext) -> AgentAction:
        action, usage_after = self._steps[self._index]
        self._index += 1
        self.total_usage = usage_after
        return action


class _NoUsageAttributeEngine(AgentDecisionEngine):
    """A decision engine that never exposes `total_usage` at all --
    e.g. DeterministicDecisionEngine, or any other test double."""

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Done, with no usage attribute at all.",
        )


def _usage(total: int) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=total,
        completion_tokens=0,
        total_tokens=total,
    )


# ---------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------


def test_loop_rejects_non_token_budget():
    agent = _build_agent()
    agent.start_task("Do something trivial.")
    engine = _UsageAccumulatingEngine(
        [
            (
                AgentAction(action_type=AgentActionType.COMPLETE, reason="Done."),
                _usage(10),
            )
        ]
    )

    with pytest.raises(TypeError, match="token_budget"):
        AgentExecutionLoop(
            agent=agent,
            decision_engine=engine,
            token_budget="not-a-token-budget",
        )


def test_loop_without_token_budget_is_unaffected_by_high_usage():
    agent = _build_agent()
    agent.start_task("Do something trivial.")
    engine = _UsageAccumulatingEngine(
        [
            (
                AgentAction(action_type=AgentActionType.COMPLETE, reason="Done."),
                _usage(1_000_000),
            )
        ]
    )

    loop = AgentExecutionLoop(agent=agent, decision_engine=engine)
    result = loop.run()

    assert result.status == "COMPLETED"
    assert result.token_usage == _usage(1_000_000)


# ---------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------


def test_budget_blocks_the_very_step_that_reaches_the_cap():
    agent = _build_agent()
    agent.start_task("Do something trivial.")
    engine = _UsageAccumulatingEngine(
        [
            (
                AgentAction(action_type=AgentActionType.COMPLETE, reason="Done."),
                _usage(100),
            )
        ]
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        token_budget=TokenBudget(max_total_tokens=100),
    )
    result = loop.run()

    assert result.status == "BUDGET_EXCEEDED"
    # The step is never counted, and the COMPLETE action was never
    # actually executed -- the agent's own task never reached DONE.
    assert result.steps == 0
    assert "100" in result.reason


def test_budget_allows_steps_while_under_the_cap_then_blocks_the_one_that_crosses_it():
    agent = _build_agent()
    agent.start_task("Do something trivial.")
    engine = _UsageAccumulatingEngine(
        [
            (
                AgentAction(
                    action_type=AgentActionType.INVOKE_TOOL,
                    tool_id="web_search",
                    inputs={"query": "first step, well under budget"},
                    reason="Step one.",
                ),
                _usage(40),
            ),
            (
                AgentAction(action_type=AgentActionType.COMPLETE, reason="Done."),
                _usage(150),
            ),
        ]
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        token_budget=TokenBudget(max_total_tokens=100),
    )
    result = loop.run()

    assert result.status == "BUDGET_EXCEEDED"
    # The first (tool) step really ran and is counted; only the second
    # step -- the one that actually crossed the cap -- was blocked.
    assert result.steps == 1
    assert len(result.context.tool_results) == 1


def test_budget_does_not_block_a_run_that_stays_under_the_cap():
    agent = _build_agent()
    agent.start_task("Do something trivial.")
    engine = _UsageAccumulatingEngine(
        [
            (
                AgentAction(action_type=AgentActionType.COMPLETE, reason="Done."),
                _usage(50),
            )
        ]
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        token_budget=TokenBudget(max_total_tokens=100),
    )
    result = loop.run()

    assert result.status == "COMPLETED"
    assert result.token_usage == _usage(50)


def test_budget_check_treats_missing_usage_as_not_exceeded():
    # A decision engine that exposes no total_usage attribute at all
    # (e.g. DeterministicDecisionEngine) must never be treated as
    # having silently blown the budget -- exactly TokenBudget.
    # exceeded_by's own "None means unknown, never a fabricated
    # violation" precedent.
    agent = _build_agent()
    agent.start_task("Do something trivial.")

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_NoUsageAttributeEngine(),
        token_budget=TokenBudget(max_total_tokens=1),
    )
    result = loop.run()

    assert result.status == "COMPLETED"
    assert result.token_usage is None


def test_budget_exceeded_is_not_recoverable_at_the_kernel_policy_level():
    # The Kernel's own PolicyEngine only ever authorizes recovery for
    # DECISION_ERROR/EXECUTION_ERROR -- see core/policies/policy_engine.py.
    # BUDGET_EXCEEDED, like GUARDRAIL_BLOCKED, is a deliberate,
    # considered outcome, not a crash: retrying would just spend
    # further tokens against a budget that has already been reached.
    from core.policies.policy_engine import PolicyEngine

    engine = PolicyEngine()
    assert engine.is_recovery_authorized("BUDGET_EXCEEDED") is False
