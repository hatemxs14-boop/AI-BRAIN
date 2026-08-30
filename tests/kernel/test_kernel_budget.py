"""
Tests for Kernel(token_budget=...) -- Build Phase 26's Kernel-level
hard spending-cap wiring (core/llm/budget.py, core/kernel/kernel.py).

Uses the same isolated, tempfile-based permissions.json fixture style
tests/kernel/test_kernel_guardrails.py already established, with a
real, LOW-risk, auto-allowed "web_search" tool (so a real
ToolExecutionResult flows through the loop when a budget-blocked run
does not stop it first).
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

from core.llm.budget import TokenBudget
from core.llm.token_usage import TokenUsage

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
        purpose="A minimal agent used only to exercise the Kernel's token budget.",
    )

    return AgentCore(identity=identity, tools=interface)


def _usage(total: int) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=total,
        completion_tokens=0,
        total_tokens=total,
    )


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


@pytest.fixture()
def tmp_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _kernel_with_scripted_agent(
    tmp_dir: Path,
    steps: list[tuple[AgentAction, TokenUsage]],
    *,
    token_budget: TokenBudget | None = None,
) -> Kernel:
    kernel = Kernel(
        orchestration_engine=SequentialOrchestrationEngine(),
        token_budget=token_budget,
    )

    kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Runs a fixed, scripted sequence of actions.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: _UsageAccumulatingEngine(steps),
        )
    )

    return kernel


# ---------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------


def test_kernel_rejects_non_token_budget():
    with pytest.raises(TypeError, match="token_budget"):
        Kernel(token_budget="not-a-token-budget")


def test_kernel_without_token_budget_is_unaffected_by_high_usage(tmp_dir):
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [
            (
                AgentAction(action_type=AgentActionType.COMPLETE, reason="Done."),
                _usage(1_000_000),
            )
        ],
    )

    result = kernel.run("Research AI agents")

    assert result.status == "COMPLETED"


# ---------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------


def test_kernel_token_budget_blocks_and_reports_budget_exceeded(tmp_dir):
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [
            (
                AgentAction(action_type=AgentActionType.COMPLETE, reason="Done."),
                _usage(100),
            )
        ],
        token_budget=TokenBudget(max_total_tokens=100),
    )

    result = kernel.run("Research AI agents")

    assert result.status == "BUDGET_EXCEEDED"
    assert result.recovery_attempts == 0
    assert result.verification.passed is False


def test_kernel_token_budget_does_not_trigger_recovery(tmp_dir):
    # BUDGET_EXCEEDED must never be treated as a crash worth retrying
    # -- retrying would just spend further tokens against a budget
    # that has already been reached (see PolicyEngine.
    # RECOVERY_AUTHORIZED_STATUSES, unchanged by this phase).
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [
            (
                AgentAction(action_type=AgentActionType.COMPLETE, reason="Done."),
                _usage(100),
            )
        ],
        token_budget=TokenBudget(max_total_tokens=100),
    )
    kernel.max_recovery_attempts = 3

    result = kernel.run("Research AI agents")

    assert result.status == "BUDGET_EXCEEDED"
    assert result.recovery_attempts == 0


def test_kernel_token_budget_blocks_before_a_tool_is_ever_invoked(tmp_dir):
    kernel = _kernel_with_scripted_agent(
        tmp_dir,
        [
            (
                AgentAction(
                    action_type=AgentActionType.INVOKE_TOOL,
                    tool_id="web_search",
                    inputs={"query": "over budget already"},
                    reason="Testing.",
                ),
                _usage(500),
            )
        ],
        token_budget=TokenBudget(max_total_tokens=100),
    )

    result = kernel.run("Research AI agents")

    assert result.status == "BUDGET_EXCEEDED"
    assert result.loop_result.last_result is None


def test_resume_also_applies_the_kernels_token_budget(tmp_dir):
    # A checkpointed run that dies mid-flight, then resumes into a step
    # that would push spend past the configured cap, must still be
    # blocked on resume -- the Kernel-level token_budget is not
    # bypassed just because this attempt is a resume rather than a
    # fresh run.
    class _SimulatedProcessDeath(BaseException):
        pass

    class _DieThenOverspendEngine(AgentDecisionEngine):
        def __init__(self) -> None:
            self.total_usage: TokenUsage | None = None

        def decide(self, context: AgentContext) -> AgentAction:
            if not context.tool_results:
                raise _SimulatedProcessDeath("Simulated process interruption.")
            self.total_usage = _usage(500)
            return AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="Done, but over budget by now.",
            )

    store = FileCheckpointStore(tmp_dir / "checkpoints")

    dying_kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())
    dying_kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Dies immediately (nothing to checkpoint yet).",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: _DieThenOverspendEngine(),
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
            checkpoint_id="task-overspend",
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
        token_budget=TokenBudget(max_total_tokens=100),
    )
    resuming_kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Completes, but over budget.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(tmp_dir, "test_agent"),
            build_decision_engine=lambda: _DieThenOverspendEngine(),
        )
    )

    result = resuming_kernel.resume("task-overspend", checkpoint_store=store)

    assert result.status == "BUDGET_EXCEEDED"
