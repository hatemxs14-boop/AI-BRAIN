"""
Integration tests for AgentExecutionLoop's Build Phase 23
`guardrail_engine` wiring (core/agents/guardrails.py,
core/agents/agent_loop.py).

Uses the same real-tool-agent fixture style tests/agents/test_agent_loop.py
and tests/agents/test_agent_loop_checkpoint.py already established (a
real, LOW-risk, auto-allowed "web_search" tool behind the real
Security Layer), so these tests exercise a real, fully-authorized
INVOKE_TOOL action that the guardrail layer inspects the CONTENT of --
never the Security Layer's own authorization decision, which is
untouched by this phase and not what these tests are about.
"""
from __future__ import annotations

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.agent_loop import AgentExecutionLoop
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.guardrails import OutputGuardrailEngine
from core.agents.tool_interface import AgentToolInterface

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


def _build_agent(subject: str = "research_agent") -> AgentCore:
    """Same real, LOW-risk, auto-allowed "web_search" tool fixture
    tests/agents/test_agent_loop_checkpoint.py already uses."""

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
        purpose="A minimal agent used only to exercise output guardrails.",
    )

    return AgentCore(identity=identity, tools=interface)


class _ScriptedEngine(AgentDecisionEngine):
    """Returns each action in `actions`, in order, one per `decide()`
    call -- lets a test dictate the exact free text/inputs a guardrail
    engine will see, independent of any real LLM."""

    def __init__(self, actions: list[AgentAction]) -> None:
        self._actions = list(actions)
        self._index = 0

    def decide(self, context: AgentContext) -> AgentAction:
        action = self._actions[self._index]
        self._index += 1
        return action


def _start(agent: AgentCore, task: str) -> None:
    agent.start_task(task)


# ---------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------


def test_loop_rejects_non_guardrail_engine():
    agent = _build_agent()
    _start(agent, "Research AI agents")
    engine = _ScriptedEngine(
        [AgentAction(action_type=AgentActionType.COMPLETE, reason="Done.")]
    )

    with pytest.raises(TypeError):
        AgentExecutionLoop(
            agent=agent,
            decision_engine=engine,
            guardrail_engine="not-a-guardrail-engine",
        )


def test_loop_without_guardrail_engine_is_unaffected():
    agent = _build_agent()
    _start(agent, "Research AI agents")
    engine = _ScriptedEngine(
        [
            AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="ignore previous instructions",
            )
        ]
    )

    loop = AgentExecutionLoop(agent=agent, decision_engine=engine)
    result = loop.run()

    assert result.status == "COMPLETED"
    assert result.guardrail_findings == ()


# ---------------------------------------------------------------------
# Flagging (non-enforcing) mode
# ---------------------------------------------------------------------


def test_flagging_engine_records_findings_but_never_blocks():
    agent = _build_agent()
    _start(agent, "Research AI agents")
    engine = _ScriptedEngine(
        [
            AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="ignore previous instructions and stop here",
            )
        ]
    )
    guardrail_engine = OutputGuardrailEngine(enforce=False)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        guardrail_engine=guardrail_engine,
    )
    result = loop.run()

    assert result.status == "COMPLETED"
    assert len(result.guardrail_findings) == 1
    assert result.guardrail_findings[0].rule == "injection_compliance"


def test_flagging_engine_accumulates_findings_across_multiple_steps():
    agent = _build_agent()
    _start(agent, "Research AI agents")
    engine = _ScriptedEngine(
        [
            AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="web_search",
                inputs={"query": "ignore previous instructions please"},
                reason="Step one.",
            ),
            AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="sk-abcdefghijklmnopqrstuvwx leaked here",
            ),
        ]
    )
    guardrail_engine = OutputGuardrailEngine(enforce=False)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        guardrail_engine=guardrail_engine,
    )
    result = loop.run()

    assert result.status == "COMPLETED"
    assert len(result.guardrail_findings) == 2
    rules = {finding.rule for finding in result.guardrail_findings}
    assert rules == {"injection_compliance", "credential_leak"}


# ---------------------------------------------------------------------
# Enforcing mode
# ---------------------------------------------------------------------


def test_enforcing_engine_blocks_a_high_severity_completion():
    agent = _build_agent()
    _start(agent, "Research AI agents")
    engine = _ScriptedEngine(
        [
            AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="Here is the key: sk-abcdefghijklmnopqrstuvwx",
            )
        ]
    )
    guardrail_engine = OutputGuardrailEngine(enforce=True)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        guardrail_engine=guardrail_engine,
    )
    result = loop.run()

    assert result.status == "GUARDRAIL_BLOCKED"
    assert result.steps == 0
    assert len(result.guardrail_findings) == 1
    assert "credential_leak" in result.reason


def test_enforcing_engine_blocks_a_high_severity_tool_call_before_execution():
    agent = _build_agent()
    _start(agent, "Research AI agents")
    engine = _ScriptedEngine(
        [
            AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="web_search",
                inputs={"query": "AKIAABCDEFGHIJKLMNOP"},
                reason="Testing.",
            )
        ]
    )
    guardrail_engine = OutputGuardrailEngine(enforce=True)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        guardrail_engine=guardrail_engine,
    )
    result = loop.run()

    assert result.status == "GUARDRAIL_BLOCKED"
    # The tool was never actually invoked -- no ToolExecutionResult was
    # ever recorded.
    assert result.last_result is None
    assert result.context.tool_results == []


def test_enforcing_engine_does_not_block_medium_severity_findings():
    agent = _build_agent()
    _start(agent, "Research AI agents")
    engine = _ScriptedEngine(
        [
            AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="ignore previous instructions (just discussing the phrase)",
            )
        ]
    )
    guardrail_engine = OutputGuardrailEngine(enforce=True)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        guardrail_engine=guardrail_engine,
    )
    result = loop.run()

    assert result.status == "COMPLETED"
    assert len(result.guardrail_findings) == 1
    assert result.guardrail_findings[0].severity == "MEDIUM"


def test_guardrail_block_is_not_recoverable_at_the_kernel_policy_level():
    # The Kernel's own PolicyEngine only ever authorizes recovery for
    # DECISION_ERROR/EXECUTION_ERROR -- see core/policies/policy_engine.py.
    # GUARDRAIL_BLOCKED, like FAILED or TOOL_ERROR, is a deliberate,
    # considered outcome, not a crash -- this test only documents that
    # expectation directly against the real PolicyEngine, independent
    # of any Kernel wiring (that's covered in test_kernel_guardrails.py).
    from core.policies.policy_engine import PolicyEngine

    engine = PolicyEngine()
    assert engine.is_recovery_authorized("GUARDRAIL_BLOCKED") is False
