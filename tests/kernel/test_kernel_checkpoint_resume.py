"""
Tests for Kernel.run(checkpoint_store=..., checkpoint_id=...) and
Kernel.resume() -- Build Phase 22's Kernel-level checkpoint/resume
wiring (core/agents/checkpoint.py, core/kernel/kernel.py).

Uses the same isolated, tempfile-based permissions.json fixture style
tests/kernel/test_kernel.py and tests/kernel/test_concurrent_kernel.py
already established, with a real, LOW-risk, auto-allowed "web_search"
tool (so real ToolExecutionResult objects flow into the checkpoint) --
see tests/kernel/test_kernel.py's own `_build_low_risk_tool_agent` for
the pattern this mirrors.
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
        purpose="A minimal agent used only to exercise Kernel checkpoint/resume.",
    )

    return AgentCore(identity=identity, tools=interface)


class _SimulatedProcessDeath(BaseException):
    """Deliberately a BaseException -- see test_agent_loop_checkpoint.py's
    own module docstring for why this is the right way to simulate a
    process-level interruption in a test."""


class _ToolCallsThenDieEngine(AgentDecisionEngine):
    def __init__(self, calls_before_death: int) -> None:
        self.calls_before_death = calls_before_death

    def decide(self, context: AgentContext) -> AgentAction:
        completed = len(context.tool_results)

        if completed >= self.calls_before_death:
            raise _SimulatedProcessDeath("Simulated process interruption.")

        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="web_search",
            inputs={"query": f"{context.task} #{completed}"},
            reason="Need more research.",
        )


class _CompleteAfterEngine(AgentDecisionEngine):
    def __init__(self, total_calls_needed: int) -> None:
        self.total_calls_needed = total_calls_needed

    def decide(self, context: AgentContext) -> AgentAction:
        completed = len(context.tool_results)

        if completed < self.total_calls_needed:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="web_search",
                inputs={"query": f"{context.task} #{completed}"},
                reason="Need more research.",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Research completed.",
        )


@pytest.fixture()
def tmp_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _kernel_with_dying_agent(tmp_dir: Path, calls_before_death: int) -> Kernel:
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Dies after a fixed number of tool calls.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: _ToolCallsThenDieEngine(
                calls_before_death=calls_before_death
            ),
        )
    )

    return kernel


def _kernel_with_completing_agent(tmp_dir: Path, total_calls_needed: int) -> Kernel:
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Completes once enough tool calls have succeeded.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: _CompleteAfterEngine(
                total_calls_needed=total_calls_needed
            ),
        )
    )

    return kernel


# ---------------------------------------------------------------------
# Kernel.run(checkpoint_store=..., checkpoint_id=...)
# ---------------------------------------------------------------------


def test_run_requires_checkpoint_id_when_checkpoint_store_given(tmp_dir):
    kernel = _kernel_with_completing_agent(tmp_dir, total_calls_needed=1)
    store = FileCheckpointStore(tmp_dir / "checkpoints")

    with pytest.raises(ValueError):
        kernel.run("Research AI agents", checkpoint_store=store)


def test_run_with_checkpoint_store_behaves_normally_and_cleans_up(tmp_dir):
    kernel = _kernel_with_completing_agent(tmp_dir, total_calls_needed=2)
    store = FileCheckpointStore(tmp_dir / "checkpoints")

    result = kernel.run(
        "Research AI agents",
        checkpoint_store=store,
        checkpoint_id="kernel-task-1",
    )

    assert result.status == "COMPLETED"
    assert store.load("kernel-task-1") is None


def test_run_without_checkpoint_store_is_unaffected(tmp_dir):
    # The zero-risk default path: no checkpoint_store at all behaves
    # exactly as it did before Build Phase 22 existed.
    kernel = _kernel_with_completing_agent(tmp_dir, total_calls_needed=1)

    result = kernel.run("Research AI agents")

    assert result.status == "COMPLETED"


def test_run_checkpoint_survives_a_simulated_process_death(tmp_dir):
    kernel = _kernel_with_dying_agent(tmp_dir, calls_before_death=2)
    store = FileCheckpointStore(tmp_dir / "checkpoints")

    with pytest.raises(_SimulatedProcessDeath):
        kernel.run(
            "Research AI agents",
            checkpoint_store=store,
            checkpoint_id="kernel-task-2",
        )

    checkpoint = store.load("kernel-task-2")

    assert checkpoint is not None
    assert checkpoint.step_count == 2
    assert checkpoint.subject == "test_agent"


# ---------------------------------------------------------------------
# Kernel.resume()
# ---------------------------------------------------------------------


def test_resume_continues_an_interrupted_task_to_completion(tmp_dir):
    store = FileCheckpointStore(tmp_dir / "checkpoints")

    dying_kernel = _kernel_with_dying_agent(tmp_dir, calls_before_death=2)

    with pytest.raises(_SimulatedProcessDeath):
        dying_kernel.run(
            "Research AI agents",
            checkpoint_store=store,
            checkpoint_id="kernel-task-3",
        )

    assert store.load("kernel-task-3") is not None

    # A brand-new Kernel instance -- simulating a fresh process that
    # never saw the original run() call at all, only the checkpoint on
    # disk -- picks the same task back up.
    resuming_kernel = _kernel_with_completing_agent(
        tmp_dir, total_calls_needed=2
    )

    result = resuming_kernel.resume(
        "kernel-task-3",
        checkpoint_store=store,
    )

    assert result.status == "COMPLETED"
    assert result.subject == "test_agent"
    assert store.load("kernel-task-3") is None


def test_resume_continues_with_a_real_new_tool_call(tmp_dir):
    store = FileCheckpointStore(tmp_dir / "checkpoints")

    dying_kernel = _kernel_with_dying_agent(tmp_dir, calls_before_death=1)

    with pytest.raises(_SimulatedProcessDeath):
        dying_kernel.run(
            "Research AI agents",
            checkpoint_store=store,
            checkpoint_id="kernel-task-4",
        )

    resuming_kernel = _kernel_with_completing_agent(
        tmp_dir, total_calls_needed=2
    )

    result = resuming_kernel.resume(
        "kernel-task-4",
        checkpoint_store=store,
    )

    assert result.status == "COMPLETED"
    assert result.loop_result.last_result is not None
    assert result.loop_result.last_result.status == "SUCCESS"


def test_resume_raises_when_checkpoint_missing(tmp_dir):
    store = FileCheckpointStore(tmp_dir / "checkpoints")
    kernel = _kernel_with_completing_agent(tmp_dir, total_calls_needed=1)

    with pytest.raises(ValueError):
        kernel.resume("does-not-exist", checkpoint_store=store)


def test_resume_raises_when_no_agent_registered_for_subject(tmp_dir):
    store = FileCheckpointStore(tmp_dir / "checkpoints")

    dying_kernel = _kernel_with_dying_agent(tmp_dir, calls_before_death=1)

    with pytest.raises(_SimulatedProcessDeath):
        dying_kernel.run(
            "Research AI agents",
            checkpoint_store=store,
            checkpoint_id="kernel-task-5",
        )

    # A Kernel with no registered agents at all -- the checkpoint names
    # a subject this Kernel has never heard of.
    empty_kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    with pytest.raises(ValueError):
        empty_kernel.resume("kernel-task-5", checkpoint_store=store)


def test_resume_rejects_a_non_checkpoint_store(tmp_dir):
    kernel = _kernel_with_completing_agent(tmp_dir, total_calls_needed=1)

    with pytest.raises(TypeError):
        kernel.resume("kernel-task-1", checkpoint_store="not-a-store")


def test_resumed_kernel_result_includes_verification_and_policy_evaluation(
    tmp_dir,
):
    store = FileCheckpointStore(tmp_dir / "checkpoints")

    dying_kernel = _kernel_with_dying_agent(tmp_dir, calls_before_death=1)

    with pytest.raises(_SimulatedProcessDeath):
        dying_kernel.run(
            "Research AI agents",
            checkpoint_store=store,
            checkpoint_id="kernel-task-6",
        )

    resuming_kernel = _kernel_with_completing_agent(
        tmp_dir, total_calls_needed=1
    )

    result = resuming_kernel.resume("kernel-task-6", checkpoint_store=store)

    assert result.status == "COMPLETED"
    assert result.verification is not None
    assert result.verification.passed is True
