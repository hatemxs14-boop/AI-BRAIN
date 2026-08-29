"""
Tests for core.orchestration.multi_agent_workflow -- Build Phase 24's
real multi-agent LangGraph workflow (one graph node per agent, plus an
optional native human-approval gate between stages).

Two tiers, deliberately, same pattern already established in
tests/orchestration/test_langgraph_orchestration_engine.py:

1. Construction/input-validation tests that need no real `langgraph`
   at all (WorkflowStage's own validation, MultiAgentWorkflowEngine's
   own stage-list validation, and the "raises a clear ImportError when
   langgraph is missing" test) -- these run for real in ANY
   environment, this sandbox included. The "missing langgraph" test
   simulates the absence deterministically via
   `monkeypatch.setitem(sys.modules, "langgraph", None)` rather than
   depending on the host environment genuinely lacking the package --
   see tests/orchestration/test_langgraph_orchestration_engine.py's own
   docstring for why (the same fix applied there for the same reason).

2. Real-integration tests that build and actually run a compiled
   graph -- skip-guarded with `pytest.importorskip("langgraph")`, so
   they only run where `langgraph` (and its checkpoint/types surface)
   is actually installed. Per multi_agent_workflow.py's own module
   docstring, the interrupt/resume-specific tests here
   (`test_workflow_pauses_for_approval_and_resumes_on_approve`,
   `test_workflow_halts_on_reject`,
   `test_workflow_resume_does_not_rerun_the_approved_stages_agent`) are
   the FIRST real verification that this project's `interrupt()`/
   `Command(resume=...)`/`MemorySaver` usage matches the installed
   langgraph API -- treat this file as unverified for that specific
   piece until they have passed once on a real machine.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.agent_loop import AgentLoopResult
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.orchestration.multi_agent_workflow import (
    MultiAgentWorkflowEngine,
    WorkflowStage,
    _default_task_template,
)

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------


def _build_zero_tool_agent(tmp_dir: Path, subject: str) -> AgentCore:
    registry = ToolRegistry()

    policy = {
        "version": "1.0",
        "permissions": [],
        "defaults": {
            "unknown_risk": "DENY",
            "unknown_permission": "DENY",
            "unknown_scope": "DENY",
            "authorization_failure": "DENY",
        },
    }
    policy_path = tmp_dir / f"{subject}_permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / f"{subject}_audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)
    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject=subject,
        name=f"{subject} agent",
        purpose="A minimal agent used only to exercise multi-agent workflows.",
    )

    return AgentCore(identity=identity, tools=interface)


class _ImmediateCompleteEngine(AgentDecisionEngine):
    def __init__(self, calls: list | None = None) -> None:
        self._calls = calls

    def decide(self, context):
        if self._calls is not None:
            self._calls.append(1)
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Stage work done.",
        )


class _ImmediateFailEngine(AgentDecisionEngine):
    def decide(self, context):
        return AgentAction(
            action_type=AgentActionType.FAIL,
            reason="Deliberately failing this stage.",
        )


@pytest.fixture()
def tmp_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------
# Tier 1: pure validation, no langgraph required.
# ---------------------------------------------------------------------


def test_default_task_template_with_no_prior_results_returns_task_verbatim():
    assert _default_task_template("Research X", ()) == "Research X"


def test_default_task_template_includes_previous_stage_context():
    context = AgentLoopResult(
        status="COMPLETED",
        steps=1,
        last_result=None,
        reason="Found three relevant sources.",
        context=None,
    )

    text = _default_task_template("Write a summary", (context,))

    assert "Write a summary" in text
    assert "COMPLETED" in text
    assert "Found three relevant sources." in text


def test_workflow_stage_rejects_empty_name(tmp_dir):
    with pytest.raises(ValueError):
        WorkflowStage(
            name="   ",
            build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
            build_decision_engine=_ImmediateCompleteEngine,
        )


def test_workflow_stage_rejects_non_callable_build_agent():
    with pytest.raises(TypeError):
        WorkflowStage(
            name="research",
            build_agent="not-callable",
            build_decision_engine=_ImmediateCompleteEngine,
        )


def test_workflow_stage_rejects_non_callable_build_decision_engine(tmp_dir):
    with pytest.raises(TypeError):
        WorkflowStage(
            name="research",
            build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
            build_decision_engine="not-callable",
        )


def test_workflow_stage_rejects_non_callable_task_template(tmp_dir):
    with pytest.raises(TypeError):
        WorkflowStage(
            name="research",
            build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
            build_decision_engine=_ImmediateCompleteEngine,
            task_template="not-callable",
        )


def test_workflow_stage_rejects_zero_or_negative_max_steps(tmp_dir):
    with pytest.raises(ValueError):
        WorkflowStage(
            name="research",
            build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
            build_decision_engine=_ImmediateCompleteEngine,
            max_steps=0,
        )


def test_workflow_stage_rejects_non_int_max_steps(tmp_dir):
    with pytest.raises(TypeError):
        WorkflowStage(
            name="research",
            build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
            build_decision_engine=_ImmediateCompleteEngine,
            max_steps="ten",
        )


def test_workflow_stage_rejects_non_bool_requires_human_approval(tmp_dir):
    with pytest.raises(TypeError):
        WorkflowStage(
            name="research",
            build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
            build_decision_engine=_ImmediateCompleteEngine,
            requires_human_approval="yes",
        )


def test_workflow_stage_accepts_valid_minimal_definition(tmp_dir):
    stage = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
        build_decision_engine=_ImmediateCompleteEngine,
    )

    assert stage.name == "research"
    assert stage.max_steps == 10
    assert stage.requires_human_approval is False


def test_engine_rejects_non_sequence_stages():
    with pytest.raises(TypeError):
        MultiAgentWorkflowEngine("not-a-sequence-of-stages")


def test_engine_rejects_empty_stages():
    with pytest.raises(ValueError):
        MultiAgentWorkflowEngine([])


def test_engine_rejects_non_workflow_stage_items():
    with pytest.raises(TypeError):
        MultiAgentWorkflowEngine([object()])


def test_engine_rejects_duplicate_stage_names(tmp_dir):
    stage_a = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
        build_decision_engine=_ImmediateCompleteEngine,
    )
    stage_b = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "b"),
        build_decision_engine=_ImmediateCompleteEngine,
    )

    with pytest.raises(ValueError, match="unique"):
        MultiAgentWorkflowEngine([stage_a, stage_b])


# Every exact dotted name MultiAgentWorkflowEngine.__init__ imports.
# Blanking only "langgraph" itself is NOT sufficient once a not-yet-
# cached submodule (e.g. "langgraph.checkpoint.memory") needs to be
# resolved through an already-None-valued parent -- confirmed as a
# real machine failure (AttributeError: 'NoneType' object has no
# attribute '__path__') the first time this test used only
# `sys.modules["langgraph"] = None`. See
# tests/orchestration/test_langgraph_orchestration_engine.py's own
# comment on the same technique for the full explanation. Blanking
# every literal dotted name actually imported, regardless of whether
# it happens to be cached already, sidesteps the issue entirely.
_LANGGRAPH_IMPORT_PATHS = (
    "langgraph",
    "langgraph.graph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
)


def test_engine_raises_clear_import_error_when_langgraph_missing(
    tmp_dir, monkeypatch
):
    for name in _LANGGRAPH_IMPORT_PATHS:
        monkeypatch.setitem(sys.modules, name, None)

    stage = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "a"),
        build_decision_engine=_ImmediateCompleteEngine,
    )

    with pytest.raises(ImportError, match="langgraph is not installed"):
        MultiAgentWorkflowEngine([stage])


# ---------------------------------------------------------------------
# Tier 2: real, compiled-graph integration -- needs langgraph installed.
# ---------------------------------------------------------------------


def test_workflow_runs_two_stages_and_passes_context_between_them(tmp_dir):
    pytest.importorskip("langgraph")

    research = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "research"),
        build_decision_engine=_ImmediateCompleteEngine,
    )
    writer = WorkflowStage(
        name="writer",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "writer"),
        build_decision_engine=_ImmediateCompleteEngine,
    )

    engine = MultiAgentWorkflowEngine([research, writer])

    result = engine.run(task="Research AI agents", thread_id="thread-1")

    assert result.status == "COMPLETED"
    assert len(result.stage_results) == 2
    assert all(r.status == "COMPLETED" for r in result.stage_results)


def test_workflow_halts_when_a_stage_fails_and_never_runs_later_stages(tmp_dir):
    pytest.importorskip("langgraph")

    research = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "research"),
        build_decision_engine=_ImmediateFailEngine,
    )
    writer_calls: list = []
    writer = WorkflowStage(
        name="writer",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "writer"),
        build_decision_engine=lambda: _ImmediateCompleteEngine(writer_calls),
    )

    engine = MultiAgentWorkflowEngine([research, writer])

    result = engine.run(task="Research AI agents", thread_id="thread-2")

    assert result.status == "HALTED"
    assert "research" in result.halt_reason
    assert len(result.stage_results) == 1
    assert writer_calls == []


def test_workflow_pauses_for_approval_and_resumes_on_approve(tmp_dir):
    pytest.importorskip("langgraph")

    research = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "research"),
        build_decision_engine=_ImmediateCompleteEngine,
        requires_human_approval=True,
    )
    writer = WorkflowStage(
        name="writer",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "writer"),
        build_decision_engine=_ImmediateCompleteEngine,
    )

    engine = MultiAgentWorkflowEngine([research, writer])

    paused = engine.run(task="Research AI agents", thread_id="thread-3")

    assert paused.status == "AWAITING_APPROVAL"
    assert paused.pending_interrupt["stage"] == "research"
    assert len(paused.stage_results) == 1

    final = engine.resume(thread_id="thread-3", approval=True)

    assert final.status == "COMPLETED"
    assert len(final.stage_results) == 2


def test_workflow_halts_on_reject(tmp_dir):
    pytest.importorskip("langgraph")

    research = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "research"),
        build_decision_engine=_ImmediateCompleteEngine,
        requires_human_approval=True,
    )
    writer_calls: list = []
    writer = WorkflowStage(
        name="writer",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "writer"),
        build_decision_engine=lambda: _ImmediateCompleteEngine(writer_calls),
    )

    engine = MultiAgentWorkflowEngine([research, writer])

    engine.run(task="Research AI agents", thread_id="thread-4")
    final = engine.resume(thread_id="thread-4", approval=False)

    assert final.status == "HALTED"
    assert "not approved" in final.halt_reason
    assert len(final.stage_results) == 1
    assert writer_calls == []


def test_workflow_resume_does_not_rerun_the_approved_stages_agent(tmp_dir):
    # This is THE test for the correctness point this whole module's
    # docstring is about: interrupt()'s node re-runs on resume, but the
    # stage's own agent-running node must not, or approving a stage
    # would silently execute its agent a second time.
    pytest.importorskip("langgraph")

    research_calls: list = []
    research = WorkflowStage(
        name="research",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "research"),
        build_decision_engine=lambda: _ImmediateCompleteEngine(research_calls),
        requires_human_approval=True,
    )
    writer = WorkflowStage(
        name="writer",
        build_agent=lambda: _build_zero_tool_agent(tmp_dir, "writer"),
        build_decision_engine=_ImmediateCompleteEngine,
    )

    engine = MultiAgentWorkflowEngine([research, writer])

    engine.run(task="Research AI agents", thread_id="thread-5")
    assert len(research_calls) == 1

    engine.resume(thread_id="thread-5", approval=True)
    assert len(research_calls) == 1
