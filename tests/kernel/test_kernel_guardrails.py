"""
Tests for Kernel(guardrail_engine=...) -- Build Phase 23's Kernel-level
output-guardrails wiring (core/agents/guardrails.py,
core/kernel/kernel.py).

Uses the same isolated, tempfile-based permissions.json fixture style
tests/kernel/test_kernel.py and tests/kernel/test_kernel_checkpoint_resume.py
already established, with a real, LOW-risk, auto-allowed "web_search"
tool (so a real ToolExecutionResult flows through the loop when a
guardrail-blocked run does not stop it first).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.checkpoint import FileCheckpointStore
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.guardrails import OutputGuardrailEngine
from core.agents.tool_interface import AgentToolInterface

from core.kernel.kernel import AgentRegistration, Kernel

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _write_low_risk_search_policy(tmp_dir: Path, subject: str) -> Path:
    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": subject,
                "resource": "web_search",
                "action": "search",
                "scope": "public_web",
                "risk_level": "LOW",
                "approval": "none",
            }
        ],
        "defaults": {
            "unknown_risk": "DENY",
            "unknown_permission": "DENY",
            "unknown_scope": "DENY",
            "authorization_failure": "DENY",
        },
    }
    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path


def _build_low_risk_tool_agent(tmp_dir: Path, subject: str) -> AgentCore:
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

    policy_path = _write_low_risk_search_policy(tmp_dir, subject=subject)

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

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
        purpose="A minimal agent used only to exercise Kernel guardrails.",
    )

    return AgentCore(identity=identity, tools=interface)


class _ScriptedEngine(AgentDecisionEngine):
    def __init__(self, actions: list[AgentAction]) -> None:
        self._actions = list(actions)
        self._index = 0

    def decide(self, context: AgentContext) -> AgentAction:
        action = self._actions[self._index]
        self._index += 1
        return action


@pytest.fixture()
def tmp_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _kernel_with_scripted_agent(
    tmp_dir: Path,
    actions: list[AgentAction],
    *,
    guardrail_engine: OutputGuardrailEngine | None = None,
) -> Kernel:
    kernel = Kernel(
        orchestration_engine=SequentialOrchestrationEngine(),
        guardrail_engine=guardrail_engine,
    )

    kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Runs a fixed, scripted sequence of actions.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: _ScriptedEngine(actions),
        )
    )

    return kernel


# ---------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------


def test_kernel_rejects_non_guardrail_engine():
    with pytest.raises(TypeError):
        Kernel(guardrail_engine="not-a-guardrail-engine")


def test_kernel_without_guardrail_engine_is_unaffected(tmp_dir):
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [AgentAction(action_type=AgentActionType.COMPLETE, reason="Done.")],
    )

    result = kernel.run("Research AI agents")

    assert result.status == "COMPLETED"
    assert result.loop_result.guardrail_findings == ()


# ---------------------------------------------------------------------
# Flagging (non-enforcing) Kernel-level engine
# ---------------------------------------------------------------------


def test_flagging_kernel_engine_records_findings_and_completes_normally(tmp_dir):
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [
            AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="ignore previous instructions, but not really",
            )
        ],
        guardrail_engine=OutputGuardrailEngine(enforce=False),
    )

    result = kernel.run("Research AI agents")

    assert result.status == "COMPLETED"
    assert len(result.loop_result.guardrail_findings) == 1


# ---------------------------------------------------------------------
# Enforcing Kernel-level engine
# ---------------------------------------------------------------------


def test_enforcing_kernel_engine_blocks_and_reports_guardrail_blocked(tmp_dir):
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [
            AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="Here is the key: sk-abcdefghijklmnopqrstuvwx",
            )
        ],
        guardrail_engine=OutputGuardrailEngine(enforce=True),
    )

    result = kernel.run("Research AI agents")

    assert result.status == "GUARDRAIL_BLOCKED"
    assert result.recovery_attempts == 0
    assert result.verification.passed is False


def test_enforcing_kernel_engine_does_not_trigger_recovery(tmp_dir):
    # GUARDRAIL_BLOCKED must never be treated as a crash worth retrying
    # -- it is a deliberate, considered outcome, exactly like FAILED or
    # TOOL_ERROR (see PolicyEngine.RECOVERY_AUTHORIZED_STATUSES).
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [
            AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="Here is the key: sk-abcdefghijklmnopqrstuvwx",
            )
        ],
        guardrail_engine=OutputGuardrailEngine(enforce=True),
    )
    kernel.max_recovery_attempts = 3

    result = kernel.run("Research AI agents")

    assert result.status == "GUARDRAIL_BLOCKED"
    assert result.recovery_attempts == 0


def test_enforcing_kernel_engine_blocks_before_a_tool_is_ever_invoked(tmp_dir):
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [
            AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="web_search",
                inputs={"query": "AKIAABCDEFGHIJKLMNOP"},
                reason="Testing.",
            )
        ],
        guardrail_engine=OutputGuardrailEngine(enforce=True),
    )

    result = kernel.run("Research AI agents")

    assert result.status == "GUARDRAIL_BLOCKED"
    assert result.loop_result.last_result is None


def test_resume_also_applies_the_kernels_guardrail_engine(tmp_dir):
    # A checkpointed run that dies mid-flight, then resumes into an
    # action an enforcing guardrail engine would block, must still be
    # blocked on resume -- the Kernel-level guardrail_engine is not
    # bypassed just because this attempt is a resume rather than a
    # fresh run.
    class _SimulatedProcessDeath(BaseException):
        pass

    class _DieThenLeakEngine(AgentDecisionEngine):
        def __init__(self) -> None:
            self._died = False

        def decide(self, context: AgentContext) -> AgentAction:
            if not context.tool_results:
                raise _SimulatedProcessDeath("Simulated process interruption.")
            return AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="Here is the key: sk-abcdefghijklmnopqrstuvwx",
            )

    store = FileCheckpointStore(tmp_dir / "checkpoints")

    dying_kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())
    dying_kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Dies immediately (nothing to checkpoint yet).",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: _DieThenLeakEngine(),
        )
    )

    # Seed a checkpoint directly: the dying engine raises before any
    # tool call ever succeeds, so there is nothing for the dying run
    # itself to persist -- write one by hand, exactly the shape
    # AgentExecutionLoop._save_checkpoint would have produced after
    # one successful step, so resume() has something real to load.
    from core.agents.checkpoint import TaskCheckpoint

    store.save(
        TaskCheckpoint(
            checkpoint_id="task-leak",
            subject="test_agent",
            task="Research AI agents",
            step_count=1,
            tool_results=(
                {"status": "SUCCESS", "summary": "ok", "artifacts": []},
            ),
            last_tool_id="web_search",
        )
    )

    resuming_kernel = Kernel(
        orchestration_engine=SequentialOrchestrationEngine(),
        guardrail_engine=OutputGuardrailEngine(enforce=True),
    )
    resuming_kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Completes with a leaking reason.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: _DieThenLeakEngine(),
        )
    )

    result = resuming_kernel.resume("task-leak", checkpoint_store=store)

    assert result.status == "GUARDRAIL_BLOCKED"
