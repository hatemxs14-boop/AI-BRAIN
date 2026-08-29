"""
Tests for AgentExecutionLoop/AgentLoopResult.token_usage (Build Phase
19) -- confirms AgentLoopResult._build_result reads a decision
engine's `total_usage` attribute correctly, duck-typed (no
AgentDecisionEngine interface change), and degrades to `None` exactly
when it should: a decision engine that never exposes the attribute, or
an `action_provider`-driven loop with no decision engine at all.

Uses the same minimal zero-tool AgentCore fixture tests/agents/
test_agent_loop.py's own `build_agent()` already established, but
without any registered tool -- these tests only need COMPLETE/FAIL/
MAX_STEPS_EXCEEDED outcomes, not a real tool call.
"""
from __future__ import annotations

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.agent_loop import AgentExecutionLoop
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.llm.token_usage import TokenUsage

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


def _build_zero_tool_agent() -> AgentCore:
    registry = ToolRegistry()
    security = SecurityDecisionPoint(PERMISSIONS_FILE)
    gateway = ToolGateway(security=security, registry=registry)
    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject="test_agent",
        name="Test Agent",
        purpose="A minimal agent used only to exercise token_usage propagation.",
    )

    return AgentCore(identity=identity, tools=interface)


class _UsageReportingCompleteEngine(AgentDecisionEngine):
    """
    Completes on the first decision, exposing a fixed `total_usage`
    attribute the same duck-typed way LLMDecisionEngine does after a
    real call -- used to prove AgentExecutionLoop reads it, without
    needing a real LLM client in these tests.
    """

    def __init__(self, usage: TokenUsage | None):
        self.total_usage = usage

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Done.",
        )


class _UsageReportingFailEngine(AgentDecisionEngine):
    """Same as above but FAILs -- proves token_usage is filled in on
    every _build_result exit path, not only the COMPLETE one."""

    def __init__(self, usage: TokenUsage | None):
        self.total_usage = usage

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.FAIL,
            reason="Deliberately failing.",
        )


class _NoUsageAttributeEngine(AgentDecisionEngine):
    """A decision engine that never exposes `total_usage` at all --
    e.g. DeterministicDecisionEngine, or any other test double."""

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Done, with no usage attribute at all.",
        )


def test_agent_loop_result_carries_the_decision_engines_total_usage():
    agent = _build_zero_tool_agent()
    agent.start_task("Do something trivial.")

    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_UsageReportingCompleteEngine(usage),
        max_steps=5,
    )

    result = loop.run()

    assert result.status == "COMPLETED"
    assert result.token_usage == usage


def test_agent_loop_result_carries_usage_on_a_fail_outcome_too():
    agent = _build_zero_tool_agent()
    agent.start_task("Do something trivial.")

    usage = TokenUsage(prompt_tokens=20, completion_tokens=9, total_tokens=29)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_UsageReportingFailEngine(usage),
        max_steps=5,
    )

    result = loop.run()

    assert result.status == "FAILED"
    assert result.token_usage == usage


def test_agent_loop_result_token_usage_is_none_without_a_usage_attribute():
    agent = _build_zero_tool_agent()
    agent.start_task("Do something trivial.")

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_NoUsageAttributeEngine(),
        max_steps=5,
    )

    result = loop.run()

    assert result.status == "COMPLETED"
    assert result.token_usage is None


def test_agent_loop_result_token_usage_is_none_with_an_action_provider():
    agent = _build_zero_tool_agent()
    agent.start_task("Do something trivial.")

    loop = AgentExecutionLoop(
        agent=agent,
        action_provider=lambda current_agent: AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Done via action_provider, no decision engine at all.",
        ),
        max_steps=5,
    )

    result = loop.run()

    assert result.status == "COMPLETED"
    assert result.token_usage is None
