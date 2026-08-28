"""
Tests for core.kernel.kernel's Build Phase 15 workflow mechanism:
WorkflowStep, WorkflowDefinition, WorkflowStepResult, WorkflowRunResult,
Kernel.register_workflow(), Kernel.run_workflow(), and the shared
extract_first_artifact_path() helper.

Uses the same minimal, isolated fixtures tests/kernel/test_kernel.py
already established (a zero-tool AgentCore, a real-but-synthetic
write_report-shaped tool, isolated tempfile-based permissions.json)
rather than the real research_agent/writer_agent/reviewer_agent stack,
so these tests exercise the Kernel's own workflow mechanics in
isolation. tests/kernel/test_kernel_workflow_integration.py covers the
full real-stack, real-agent "research_write_review" workflow
build_default_kernel() now wires.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.agent_loop import AgentLoopResult
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.kernel.kernel import (
    AgentRegistration,
    Kernel,
    NormalizedTask,
    WorkflowDefinition,
    WorkflowStep,
    extract_first_artifact_path,
)

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------
# Fixtures -- mirrors tests/kernel/test_kernel.py's own helpers.
# ---------------------------------------------------------------------

def _write_empty_policy(tmp_dir: Path) -> Path:
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
    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path


def _write_report_policy(
    tmp_dir: Path,
    *,
    subject: str,
    approval: str,
    risk_level: str,
) -> Path:
    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": subject,
                "resource": "report",
                "action": "write",
                "scope": "workspace",
                "risk_level": risk_level,
                "approval": approval,
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


def _build_zero_tool_agent(tmp_dir: Path, subject: str = "test_agent") -> AgentCore:
    registry = ToolRegistry()
    policy_path = _write_empty_policy(tmp_dir)

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)
    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject=subject,
        name="Test Agent",
        purpose="A minimal agent used only to exercise workflow mechanics.",
    )

    return AgentCore(identity=identity, tools=interface)


def _build_write_report_tool_agent(
    tmp_dir: Path,
    *,
    subject: str,
    approval: str = "none",
    risk_level: str = "LOW",
) -> AgentCore:
    """
    A real "write_report"-id tool whose executor returns a real
    `{"path": ..., "size_bytes": ...}` artifact -- the same shape
    tests/kernel/test_kernel.py's own _build_write_report_tool_agent
    uses, reproduced here (self-contained per this file's own
    convention -- see tests/kernel/test_kernel_reviewer_agent_
    integration.py's own docstring for why sibling test files don't
    import fixtures from one another) with an `approval`/`risk_level`
    knob so the same helper can build both an auto-allowed and an
    approval-gated version.
    """

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="write_report",
            name="Write Report",
            purpose="Publish a written report.",
            input_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "size_bytes": {"type": "integer"},
                },
                "required": ["path", "size_bytes"],
            },
            permissions=(f"{subject}:report:write:workspace",),
            resource="report",
            action="write",
            scope="workspace",
            risk_level=risk_level,
            error_handling={
                "retryable": False,
                "on_failure": "Surface the write error to the agent.",
            },
        )
    )

    policy_path = _write_report_policy(
        tmp_dir, subject=subject, approval=approval, risk_level=risk_level
    )

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(
        tool_id="write_report",
        executor=lambda filename, content: {
            "path": filename,
            "size_bytes": len(content.encode("utf-8")),
        },
    )

    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject=subject,
        name="Test Agent",
        purpose="A minimal agent used only to exercise workflow mechanics.",
    )

    return AgentCore(identity=identity, tools=interface)


class _ImmediateCompleteEngine(AgentDecisionEngine):
    def decide(self, context):
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Nothing to do.",
        )


class _WriteReportThenCompleteEngine(AgentDecisionEngine):
    def __init__(self, *, filename: str = "report.md"):
        self._invoked = False
        self._filename = filename

    def decide(self, context):
        if not self._invoked:
            self._invoked = True
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="write_report",
                inputs={
                    "filename": self._filename,
                    "content": "# Report\n\nContent.",
                },
                reason="Publish before completing.",
            )
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Report published.",
        )


class _WriteReportWithoutApprovalEngine(AgentDecisionEngine):
    def __init__(self, *, filename: str = "report.md"):
        self._filename = filename

    def decide(self, context):
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="write_report",
            inputs={
                "filename": self._filename,
                "content": "Attempting to publish without approval.",
            },
            reason="Attempt a gated write.",
        )


class _NeverCompleteEngine(AgentDecisionEngine):
    def decide(self, context):
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="does_not_exist",
            inputs={},
            reason="Deliberately invoke a tool that isn't registered.",
        )


class _RaiseNTimesThenCompleteEngine(AgentDecisionEngine):
    def __init__(self, counter: dict):
        self._counter = counter

    def decide(self, context):
        if self._counter["remaining_failures"] > 0:
            self._counter["remaining_failures"] -= 1
            raise RuntimeError("Simulated transient decision failure.")
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Recovered.",
        )


class _ScriptedOrchestrationEngine:
    """
    Delegates to a real SequentialOrchestrationEngine for every call,
    except calls whose 1-based call index is a key in `overrides`,
    which instead return the corresponding canned AgentLoopResult
    verbatim.

    Used to exercise Kernel.run_workflow()'s VERIFICATION_FAILED branch
    with a combination that is not actually reachable by driving a real
    AgentExecutionLoop -- a real loop that reports COMPLETED never does
    so right after a tool call it itself just saw fail (see AgentLoop's
    own behavior: a DENIED/ERROR tool result already ends the loop with
    a terminal status of its own, e.g. TOOL_ERROR, without ever
    returning to the decision engine to claim COMPLETE afterwards).
    tests/kernel/test_kernel.py's own
    test_verify_fails_when_completed_but_last_tool_result_was_not_success
    exercises the same otherwise-unreachable combination the same way:
    by constructing the AgentLoopResult directly rather than through a
    real loop.
    """

    def __init__(self, overrides: dict):
        self._overrides = overrides
        self._delegate = SequentialOrchestrationEngine()
        self._call_count = 0

    def run(self, *, agent, decision_engine, max_steps):
        self._call_count += 1
        if self._call_count in self._overrides:
            return self._overrides[self._call_count]
        return self._delegate.run(
            agent=agent, decision_engine=decision_engine, max_steps=max_steps
        )


def _step_1_task(original_task: str, previous_result) -> str:
    return original_task


def _step_2_task_from_artifact(original_task: str, previous_result) -> str:
    if previous_result is None:
        raise ValueError("step 2 requires step 1's own completed result.")
    path = extract_first_artifact_path(previous_result)
    if path is None:
        raise ValueError("step 1's result carries no usable artifact.")
    return f"Review {path}."


# ---------------------------------------------------------------------
# extract_first_artifact_path
# ---------------------------------------------------------------------

def _loop_result(*, status="COMPLETED", last_result=None) -> AgentLoopResult:
    return AgentLoopResult(
        status=status,
        steps=0,
        last_result=last_result,
        reason="test",
        context=AgentContext(task="test"),
    )


def test_extract_first_artifact_path_returns_none_when_no_last_result():
    assert extract_first_artifact_path(_loop_result(last_result=None)) is None


def test_extract_first_artifact_path_returns_none_when_no_artifacts():
    last_result = SimpleNamespace(artifacts=())
    assert extract_first_artifact_path(_loop_result(last_result=last_result)) is None


def test_extract_first_artifact_path_returns_none_when_artifacts_is_none():
    last_result = SimpleNamespace(artifacts=None)
    assert extract_first_artifact_path(_loop_result(last_result=last_result)) is None


def test_extract_first_artifact_path_reads_a_dict_artifact():
    last_result = SimpleNamespace(artifacts=({"path": "finding.md"},))
    assert extract_first_artifact_path(_loop_result(last_result=last_result)) == "finding.md"


def test_extract_first_artifact_path_reads_an_object_artifact():
    last_result = SimpleNamespace(
        artifacts=(SimpleNamespace(path="report.md"),)
    )
    assert extract_first_artifact_path(_loop_result(last_result=last_result)) == "report.md"


def test_extract_first_artifact_path_returns_none_for_a_missing_path():
    last_result = SimpleNamespace(artifacts=({"size_bytes": 10},))
    assert extract_first_artifact_path(_loop_result(last_result=last_result)) is None


def test_extract_first_artifact_path_returns_none_for_an_empty_path():
    last_result = SimpleNamespace(artifacts=({"path": "   "},))
    assert extract_first_artifact_path(_loop_result(last_result=last_result)) is None


# ---------------------------------------------------------------------
# WorkflowStep / WorkflowDefinition validation
# ---------------------------------------------------------------------

def test_workflow_step_rejects_empty_subject():
    with pytest.raises(ValueError, match="subject must be"):
        WorkflowStep(subject="", build_task=_step_1_task)


def test_workflow_step_rejects_non_callable_build_task():
    with pytest.raises(TypeError, match="build_task must be"):
        WorkflowStep(subject="research_agent", build_task="not callable")


def test_workflow_definition_rejects_empty_name():
    with pytest.raises(ValueError, match="name must be"):
        WorkflowDefinition(
            name="",
            description="A workflow.",
            can_handle=lambda normalized: True,
            steps=(
                WorkflowStep(subject="a", build_task=_step_1_task),
                WorkflowStep(subject="b", build_task=_step_2_task_from_artifact),
            ),
        )


def test_workflow_definition_rejects_empty_description():
    with pytest.raises(ValueError, match="description must be"):
        WorkflowDefinition(
            name="wf",
            description="",
            can_handle=lambda normalized: True,
            steps=(
                WorkflowStep(subject="a", build_task=_step_1_task),
                WorkflowStep(subject="b", build_task=_step_2_task_from_artifact),
            ),
        )


def test_workflow_definition_rejects_non_callable_can_handle():
    with pytest.raises(TypeError, match="can_handle must be"):
        WorkflowDefinition(
            name="wf",
            description="A workflow.",
            can_handle="not callable",
            steps=(
                WorkflowStep(subject="a", build_task=_step_1_task),
                WorkflowStep(subject="b", build_task=_step_2_task_from_artifact),
            ),
        )


def test_workflow_definition_rejects_non_workflow_step_entries():
    with pytest.raises(TypeError, match="steps must be"):
        WorkflowDefinition(
            name="wf",
            description="A workflow.",
            can_handle=lambda normalized: True,
            steps=("not a step", "also not a step"),
        )


def test_workflow_definition_rejects_fewer_than_two_steps():
    with pytest.raises(ValueError, match="at least two"):
        WorkflowDefinition(
            name="wf",
            description="A workflow.",
            can_handle=lambda normalized: True,
            steps=(WorkflowStep(subject="a", build_task=_step_1_task),),
        )


# ---------------------------------------------------------------------
# register_workflow
# ---------------------------------------------------------------------

def _register_two_agents(kernel: Kernel, tmp_dir: Path) -> None:
    kernel.register_agent(
        AgentRegistration(
            subject="research_agent",
            description="Does nothing.",
            can_handle=lambda normalized: False,
            build_agent=lambda: _build_zero_tool_agent(tmp_dir, subject="research_agent"),
            build_decision_engine=lambda: _ImmediateCompleteEngine(),
        )
    )
    kernel.register_agent(
        AgentRegistration(
            subject="writer_agent",
            description="Publishes a report, then completes.",
            can_handle=lambda normalized: False,
            build_agent=lambda: _build_write_report_tool_agent(
                tmp_dir, subject="writer_agent"
            ),
            build_decision_engine=lambda: _WriteReportThenCompleteEngine(),
        )
    )


def test_register_workflow_rejects_a_non_workflow_definition():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())
    with pytest.raises(TypeError, match="WorkflowDefinition"):
        kernel.register_workflow("not a workflow")


def test_register_workflow_rejects_duplicate_name():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())
        _register_two_agents(kernel, tmp_dir)

        workflow = WorkflowDefinition(
            name="pipeline",
            description="A workflow.",
            can_handle=lambda normalized: True,
            steps=(
                WorkflowStep(subject="research_agent", build_task=_step_1_task),
                WorkflowStep(
                    subject="writer_agent", build_task=_step_2_task_from_artifact
                ),
            ),
        )

        kernel.register_workflow(workflow)

        with pytest.raises(ValueError, match="already registered"):
            kernel.register_workflow(workflow)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_register_workflow_rejects_a_step_naming_an_unregistered_subject():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())
        _register_two_agents(kernel, tmp_dir)

        workflow = WorkflowDefinition(
            name="pipeline",
            description="A workflow.",
            can_handle=lambda normalized: True,
            steps=(
                WorkflowStep(subject="research_agent", build_task=_step_1_task),
                WorkflowStep(
                    subject="nonexistent_agent",
                    build_task=_step_2_task_from_artifact,
                ),
            ),
        )

        with pytest.raises(ValueError, match="nonexistent_agent"):
            kernel.register_workflow(workflow)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# Kernel.run_workflow()
# ---------------------------------------------------------------------

def test_run_workflow_returns_no_workflow_available_when_nothing_matches():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())
        _register_two_agents(kernel, tmp_dir)

        kernel.register_workflow(
            WorkflowDefinition(
                name="pipeline",
                description="A workflow.",
                can_handle=lambda normalized: False,
                steps=(
                    WorkflowStep(subject="research_agent", build_task=_step_1_task),
                    WorkflowStep(
                        subject="writer_agent",
                        build_task=_step_2_task_from_artifact,
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Do the pipeline thing.")

        assert result.status == "NO_WORKFLOW_AVAILABLE"
        assert result.completed_steps == ()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_workflow_completes_a_full_two_step_pipeline():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Publishes a report-shaped artifact, then completes.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_write_report_tool_agent(
                    tmp_dir, subject="research_agent"
                ),
                build_decision_engine=lambda: _WriteReportThenCompleteEngine(
                    filename="finding.md"
                ),
            )
        )
        kernel.register_agent(
            AgentRegistration(
                subject="reviewer_agent",
                description="Completes without touching any tool.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(
                    tmp_dir, subject="reviewer_agent"
                ),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        kernel.register_workflow(
            WorkflowDefinition(
                name="pipeline",
                description="A workflow.",
                can_handle=lambda normalized: True,
                steps=(
                    WorkflowStep(subject="research_agent", build_task=_step_1_task),
                    WorkflowStep(
                        subject="reviewer_agent",
                        build_task=_step_2_task_from_artifact,
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Research and review it.")

        assert result.status == "COMPLETED"
        assert result.workflow_name == "pipeline"
        assert [s.subject for s in result.completed_steps] == [
            "research_agent",
            "reviewer_agent",
        ]
        assert all(s.loop_result.status == "COMPLETED" for s in result.completed_steps)
        assert all(s.verification.passed for s in result.completed_steps)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_workflow_stops_at_a_step_awaiting_approval():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Publishes a report-shaped artifact, then completes.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_write_report_tool_agent(
                    tmp_dir, subject="research_agent"
                ),
                build_decision_engine=lambda: _WriteReportThenCompleteEngine(
                    filename="finding.md"
                ),
            )
        )
        kernel.register_agent(
            AgentRegistration(
                subject="writer_agent",
                description="Attempts a HIGH-risk gated write.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_write_report_tool_agent(
                    tmp_dir,
                    subject="writer_agent",
                    approval="policy",
                    risk_level="HIGH",
                ),
                build_decision_engine=lambda: _WriteReportWithoutApprovalEngine(
                    filename="report.md"
                ),
            )
        )

        kernel.register_workflow(
            WorkflowDefinition(
                name="pipeline",
                description="A workflow.",
                can_handle=lambda normalized: True,
                steps=(
                    WorkflowStep(subject="research_agent", build_task=_step_1_task),
                    WorkflowStep(
                        subject="writer_agent",
                        build_task=_step_2_task_from_artifact,
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Research and write it up.")

        assert result.status == "AWAITING_APPROVAL"
        assert len(result.completed_steps) == 2
        assert result.completed_steps[0].loop_result.status == "COMPLETED"
        assert result.completed_steps[1].loop_result.status == "APPROVAL_REQUIRED"
        # The gated write must never have actually happened.
        assert not (tmp_dir / "report.md").exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_workflow_passes_through_a_non_completed_status_unchanged():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Publishes a report-shaped artifact, then completes.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_write_report_tool_agent(
                    tmp_dir, subject="research_agent"
                ),
                build_decision_engine=lambda: _WriteReportThenCompleteEngine(
                    filename="finding.md"
                ),
            )
        )
        kernel.register_agent(
            AgentRegistration(
                subject="broken_agent",
                description="Always invokes a tool that doesn't exist.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(
                    tmp_dir, subject="broken_agent"
                ),
                build_decision_engine=lambda: _NeverCompleteEngine(),
            )
        )

        kernel.register_workflow(
            WorkflowDefinition(
                name="pipeline",
                description="A workflow.",
                can_handle=lambda normalized: True,
                steps=(
                    WorkflowStep(subject="research_agent", build_task=_step_1_task),
                    WorkflowStep(
                        subject="broken_agent",
                        build_task=_step_2_task_from_artifact,
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Research then break.")

        assert result.status not in ("COMPLETED", "AWAITING_APPROVAL")
        assert result.status == result.completed_steps[-1].loop_result.status
        assert len(result.completed_steps) == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_workflow_reports_step_task_build_error_when_build_task_raises():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Completes without ever calling a tool.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(
                    tmp_dir, subject="research_agent"
                ),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )
        kernel.register_agent(
            AgentRegistration(
                subject="writer_agent",
                description="Never actually reached.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(
                    tmp_dir, subject="writer_agent"
                ),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        kernel.register_workflow(
            WorkflowDefinition(
                name="pipeline",
                description="A workflow.",
                can_handle=lambda normalized: True,
                steps=(
                    # research_agent here never calls a tool, so it has
                    # no artifact -- step 2's build_task must raise.
                    WorkflowStep(subject="research_agent", build_task=_step_1_task),
                    WorkflowStep(
                        subject="writer_agent",
                        build_task=_step_2_task_from_artifact,
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Research then hand off.")

        assert result.status == "STEP_TASK_BUILD_ERROR"
        # Only the first step actually ran.
        assert len(result.completed_steps) == 1
        assert result.completed_steps[0].subject == "research_agent"
        assert "usable artifact" in result.reason
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_workflow_stops_at_verification_failed():
    """
    A step whose AgentLoopResult claims COMPLETED but whose own last
    tool result did not succeed must stop the workflow with
    VERIFICATION_FAILED -- exactly the same rule Kernel.run()'s own
    _verify() already applies to a standalone task (see
    KernelVerification's own docstring). This combination is not
    reachable by driving a real AgentExecutionLoop (see
    _ScriptedOrchestrationEngine's own docstring for why), so the
    second step's result is canned via _ScriptedOrchestrationEngine,
    the same way tests/kernel/test_kernel.py's own
    test_verify_fails_when_completed_but_last_tool_result_was_not_success
    constructs it directly rather than through a real loop -- while the
    first step still runs for real, so run_workflow()'s own hand-off
    (build_task calling extract_first_artifact_path on a genuine
    result) is still exercised for real up to the point of failure.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        failed_tool_result = SimpleNamespace(status="ERROR")
        canned_step_2_result = AgentLoopResult(
            status="COMPLETED",
            steps=1,
            last_result=failed_tool_result,
            reason="Claims success despite a failed tool call.",
            context=AgentContext(task="Review finding.md."),
        )

        kernel = Kernel(
            orchestration_engine=_ScriptedOrchestrationEngine(
                {2: canned_step_2_result}
            )
        )

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Publishes a report-shaped artifact, then completes.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_write_report_tool_agent(
                    tmp_dir, subject="research_agent"
                ),
                build_decision_engine=lambda: _WriteReportThenCompleteEngine(
                    filename="finding.md"
                ),
            )
        )
        kernel.register_agent(
            AgentRegistration(
                subject="writer_agent",
                description="Never actually reached for real (step 2 is canned).",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(
                    tmp_dir, subject="writer_agent"
                ),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        kernel.register_workflow(
            WorkflowDefinition(
                name="pipeline",
                description="A workflow.",
                can_handle=lambda normalized: True,
                steps=(
                    WorkflowStep(subject="research_agent", build_task=_step_1_task),
                    WorkflowStep(
                        subject="writer_agent",
                        build_task=_step_2_task_from_artifact,
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Research and write it up.")

        assert result.status == "VERIFICATION_FAILED"
        assert len(result.completed_steps) == 2
        assert result.completed_steps[0].verification.passed is True
        assert result.completed_steps[1].verification.passed is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_workflow_applies_per_step_recovery():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        counter = {"remaining_failures": 1}

        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            max_recovery_attempts=1,
        )

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Publishes a report-shaped artifact, then completes.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_write_report_tool_agent(
                    tmp_dir, subject="research_agent"
                ),
                build_decision_engine=lambda: _WriteReportThenCompleteEngine(
                    filename="finding.md"
                ),
            )
        )
        kernel.register_agent(
            AgentRegistration(
                subject="reviewer_agent",
                description="Fails once transiently, then completes.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(
                    tmp_dir, subject="reviewer_agent"
                ),
                build_decision_engine=lambda: _RaiseNTimesThenCompleteEngine(counter),
            )
        )

        kernel.register_workflow(
            WorkflowDefinition(
                name="pipeline",
                description="A workflow.",
                can_handle=lambda normalized: True,
                steps=(
                    WorkflowStep(subject="research_agent", build_task=_step_1_task),
                    WorkflowStep(
                        subject="reviewer_agent",
                        build_task=_step_2_task_from_artifact,
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Research and review it.")

        assert result.status == "COMPLETED"
        assert counter["remaining_failures"] == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
