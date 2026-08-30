"""
Tests for Kernel.run_multi_agent_workflow()/resume_multi_agent_workflow()
-- Build Phase 25's Kernel-level wiring of Build Phase 24's
MultiAgentWorkflowEngine.

Uses the same isolated, tempfile-based permissions.json fixture style
tests/kernel/test_kernel_guardrails.py and
tests/kernel/test_kernel_checkpoint_resume.py already established, with
real, LOW-risk, auto-allowed "web_search" tool agents so a real
ToolExecutionResult can flow through a stage when a test wants one.

All real-graph tests here need `langgraph` installed -- skip-guarded
with `pytest.importorskip("langgraph")`, exactly like
tests/orchestration/test_multi_agent_workflow.py's own tier 2. The
pure input-validation tests (empty/bad subjects, unknown subject, bad
approval_gates, unknown thread_id on resume) need no real graph at all
and run in any environment.
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
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.guardrails import OutputGuardrailEngine
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
    policy_path = tmp_dir / f"{subject}_permissions.json"
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
        audit_log_path=str(tmp_dir / f"{subject}_audit.jsonl"),
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
        name=f"{subject} agent",
        purpose="A minimal agent used only to exercise Kernel multi-agent workflows.",
    )

    return AgentCore(identity=identity, tools=interface)


class _ImmediateCompleteEngine(AgentDecisionEngine):
    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Stage work done.",
        )


class _CredentialLeakEngine(AgentDecisionEngine):
    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Here is the key: sk-abcdefghijklmnopqrstuvwx",
        )


class _OverBudgetEngine(AgentDecisionEngine):
    """Completes cleanly, but exposes a `total_usage` already past any
    reasonably small TokenBudget -- the same duck-typed attribute a
    real LLMDecisionEngine exposes after a real call."""

    def __init__(self) -> None:
        self.total_usage = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Stage work done (but over budget).",
        )


@pytest.fixture()
def tmp_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _kernel_with_two_registered_agents(
    tmp_dir: Path,
    *,
    guardrail_engine: OutputGuardrailEngine | None = None,
    token_budget: TokenBudget | None = None,
    second_engine: type[AgentDecisionEngine] = _ImmediateCompleteEngine,
) -> Kernel:
    kernel = Kernel(
        orchestration_engine=SequentialOrchestrationEngine(),
        guardrail_engine=guardrail_engine,
        token_budget=token_budget,
    )

    kernel.register_agent(
        AgentRegistration(
            subject="research_agent",
            description="First stage.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_low_risk_tool_agent(
                tmp_dir, "research_agent"
            ),
            build_decision_engine=_ImmediateCompleteEngine,
        )
    )

    kernel.register_agent(
        AgentRegistration(
            subject="writer_agent",
            description="Second stage.",
            can_handle=lambda normalized: False,
            build_agent=lambda: _build_low_risk_tool_agent(
                tmp_dir, "writer_agent"
            ),
            build_decision_engine=second_engine,
        )
    )

    return kernel


# ---------------------------------------------------------------------
# Pure input validation -- no real langgraph needed.
# ---------------------------------------------------------------------


def test_rejects_non_sequence_subjects(tmp_dir):
    kernel = _kernel_with_two_registered_agents(tmp_dir)

    with pytest.raises(TypeError):
        kernel.run_multi_agent_workflow(
            subjects="research_agent",  # a string, not a sequence of subjects
            task="Research AI agents",
            thread_id="t1",
        )


def test_rejects_empty_subjects(tmp_dir):
    kernel = _kernel_with_two_registered_agents(tmp_dir)

    with pytest.raises(ValueError):
        kernel.run_multi_agent_workflow(
            subjects=(),
            task="Research AI agents",
            thread_id="t1",
        )


def test_rejects_unknown_subject(tmp_dir):
    kernel = _kernel_with_two_registered_agents(tmp_dir)

    with pytest.raises(ValueError, match="no_such_agent"):
        kernel.run_multi_agent_workflow(
            subjects=("research_agent", "no_such_agent"),
            task="Research AI agents",
            thread_id="t1",
        )


def test_rejects_non_mapping_approval_gates(tmp_dir):
    kernel = _kernel_with_two_registered_agents(tmp_dir)

    with pytest.raises(TypeError):
        kernel.run_multi_agent_workflow(
            subjects=("research_agent",),
            task="Research AI agents",
            thread_id="t1",
            approval_gates=["research_agent"],
        )


def test_resume_rejects_unknown_thread_id(tmp_dir):
    kernel = _kernel_with_two_registered_agents(tmp_dir)

    with pytest.raises(ValueError, match="No paused multi-agent workflow"):
        kernel.resume_multi_agent_workflow(
            thread_id="never-started", approval=True
        )


# ---------------------------------------------------------------------
# Real, compiled-graph integration -- needs langgraph installed.
# ---------------------------------------------------------------------


def test_runs_two_registered_agents_to_completion(tmp_dir):
    pytest.importorskip("langgraph")

    kernel = _kernel_with_two_registered_agents(tmp_dir)

    result = kernel.run_multi_agent_workflow(
        subjects=("research_agent", "writer_agent"),
        task="Research AI agents",
        thread_id="thread-k1",
    )

    assert result.status == "COMPLETED"
    assert len(result.stage_results) == 2


def test_unknown_thread_id_does_not_linger_after_completion(tmp_dir):
    pytest.importorskip("langgraph")

    kernel = _kernel_with_two_registered_agents(tmp_dir)

    kernel.run_multi_agent_workflow(
        subjects=("research_agent", "writer_agent"),
        task="Research AI agents",
        thread_id="thread-k2",
    )

    # The run already reached a terminal COMPLETED status -- nothing
    # left to resume, and the Kernel should have dropped its reference
    # to that finished engine.
    with pytest.raises(ValueError, match="No paused multi-agent workflow"):
        kernel.resume_multi_agent_workflow(
            thread_id="thread-k2", approval=True
        )


def test_approval_gate_pauses_and_resumes_through_the_kernel(tmp_dir):
    pytest.importorskip("langgraph")

    kernel = _kernel_with_two_registered_agents(tmp_dir)

    paused = kernel.run_multi_agent_workflow(
        subjects=("research_agent", "writer_agent"),
        task="Research AI agents",
        thread_id="thread-k3",
        approval_gates={"research_agent": True},
    )

    assert paused.status == "AWAITING_APPROVAL"
    assert paused.pending_interrupt["stage"] == "research_agent"
    assert len(paused.stage_results) == 1

    final = kernel.resume_multi_agent_workflow(
        thread_id="thread-k3", approval=True
    )

    assert final.status == "COMPLETED"
    assert len(final.stage_results) == 2

    # Terminal now -- resuming again must fail cleanly.
    with pytest.raises(ValueError, match="No paused multi-agent workflow"):
        kernel.resume_multi_agent_workflow(
            thread_id="thread-k3", approval=True
        )


def test_approval_gate_reject_halts_through_the_kernel(tmp_dir):
    pytest.importorskip("langgraph")

    kernel = _kernel_with_two_registered_agents(tmp_dir)

    kernel.run_multi_agent_workflow(
        subjects=("research_agent", "writer_agent"),
        task="Research AI agents",
        thread_id="thread-k4",
        approval_gates={"research_agent": True},
    )

    final = kernel.resume_multi_agent_workflow(
        thread_id="thread-k4", approval=False
    )

    assert final.status == "HALTED"
    assert len(final.stage_results) == 1


def test_kernels_guardrail_engine_is_threaded_into_every_stage(tmp_dir):
    pytest.importorskip("langgraph")

    kernel = _kernel_with_two_registered_agents(
        tmp_dir,
        guardrail_engine=OutputGuardrailEngine(enforce=True),
        second_engine=_CredentialLeakEngine,
    )

    result = kernel.run_multi_agent_workflow(
        subjects=("research_agent", "writer_agent"),
        task="Research AI agents",
        thread_id="thread-k5",
    )

    # Stage one (research_agent) completes cleanly; stage two
    # (writer_agent) leaks a credential and the Kernel's own
    # enforcing guardrail_engine -- threaded into every stage --
    # must block it there, the same way it would a single-agent run.
    assert result.status == "HALTED"
    assert len(result.stage_results) == 2
    assert result.stage_results[0].status == "COMPLETED"
    assert result.stage_results[1].status == "GUARDRAIL_BLOCKED"


def test_kernel_without_guardrail_engine_leaves_stages_unguarded(tmp_dir):
    pytest.importorskip("langgraph")

    kernel = _kernel_with_two_registered_agents(
        tmp_dir,
        second_engine=_CredentialLeakEngine,
    )

    result = kernel.run_multi_agent_workflow(
        subjects=("research_agent", "writer_agent"),
        task="Research AI agents",
        thread_id="thread-k6",
    )

    assert result.status == "COMPLETED"
    assert len(result.stage_results) == 2


def test_kernels_token_budget_is_threaded_into_every_stage(tmp_dir):
    pytest.importorskip("langgraph")

    kernel = _kernel_with_two_registered_agents(
        tmp_dir,
        token_budget=TokenBudget(max_total_tokens=100),
        second_engine=_OverBudgetEngine,
    )

    result = kernel.run_multi_agent_workflow(
        subjects=("research_agent", "writer_agent"),
        task="Research AI agents",
        thread_id="thread-k7",
    )

    # Stage one (research_agent) completes cleanly with no usage at
    # all reported (_ImmediateCompleteEngine exposes none); stage two
    # (writer_agent) reports usage past the Kernel's own token_budget
    # -- threaded into every stage -- and must be blocked there, the
    # same way it would a single-agent run.
    assert result.status == "HALTED"
    assert len(result.stage_results) == 2
    assert result.stage_results[0].status == "COMPLETED"
    assert result.stage_results[1].status == "BUDGET_EXCEEDED"


def test_kernel_without_token_budget_leaves_stages_uncapped(tmp_dir):
    pytest.importorskip("langgraph")

    kernel = _kernel_with_two_registered_agents(
        tmp_dir,
        second_engine=_OverBudgetEngine,
    )

    result = kernel.run_multi_agent_workflow(
        subjects=("research_agent", "writer_agent"),
        task="Research AI agents",
        thread_id="thread-k8",
    )

    assert result.status == "COMPLETED"
    assert len(result.stage_results) == 2
