"""
Tests for Kernel/KernelResult/WorkflowStepResult/WorkflowRunResult.
token_usage (Build Phase 19).

Self-contained per this project's own sibling-test-file convention
(see tests/kernel/test_kernel_workflow.py's own docstring for why
fixtures are not imported across test files) -- reproduces the same
minimal zero-tool-agent / write_report-tool-agent fixtures tests/
kernel/test_kernel.py and tests/kernel/test_kernel_workflow.py already
established, plus a handful of decision-engine test doubles that
expose a fixed or scripted `total_usage` attribute the same duck-typed
way LLMDecisionEngine does after a real call (see AgentExecutionLoop.
_build_result's own docstring) -- so these tests can exercise Kernel's
own accumulation logic (Kernel.run()'s RECOVER IF NEEDED retry loop
and independent-verification fold-in; Kernel.run_workflow()'s per-step
retry loop and cross-step summation) without needing a real LLM
client.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.kernel.kernel import (
    AgentRegistration,
    Kernel,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowVerifierRegistration,
    extract_first_artifact_path,
)

from core.llm.token_usage import TokenUsage, combine_token_usage

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------
# Fixtures -- mirrors tests/kernel/test_kernel.py's/test_kernel_
# workflow.py's own helpers.
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


def _write_report_policy(tmp_dir: Path, subject: str) -> Path:
    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": subject,
                "resource": "report",
                "action": "write",
                "scope": "workspace",
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
        purpose="A minimal agent used only to exercise token_usage propagation.",
    )

    return AgentCore(identity=identity, tools=interface)


def _build_write_report_tool_agent(tmp_dir: Path, subject: str = "writer_agent") -> AgentCore:
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
            risk_level="LOW",
            error_handling={
                "retryable": False,
                "on_failure": "Surface the write error to the agent.",
            },
        )
    )

    policy_path = _write_report_policy(tmp_dir, subject)

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
        purpose="A minimal agent used only to exercise token_usage propagation.",
    )

    return AgentCore(identity=identity, tools=interface)


class _UsageReportingCompleteEngine(AgentDecisionEngine):
    """Completes on the first decision, exposing a fixed `total_usage`
    -- the same duck-typed shape LLMDecisionEngine exposes after a
    real call, without needing a real LLM client here."""

    def __init__(self, usage: TokenUsage | None):
        self.total_usage = usage

    def decide(self, context):
        return AgentAction(action_type=AgentActionType.COMPLETE, reason="Done.")


class _UsageReportingWriteReportThenCompleteEngine(AgentDecisionEngine):
    """Invokes write_report exactly once, then completes -- exposing a
    fixed `total_usage` representing this whole run's accumulated
    cost, the same way a real LLMDecisionEngine's `total_usage` would
    read by the time the loop finally completes."""

    def __init__(self, *, filename: str = "report.md", usage: TokenUsage | None):
        self._invoked = False
        self._filename = filename
        self.total_usage = usage

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
        return AgentAction(action_type=AgentActionType.COMPLETE, reason="Report published.")


class _RaiseThenCompleteWithUsageEngine(AgentDecisionEngine):
    """
    Raises on decide() while `counter["remaining_failures"]` is
    positive, then completes -- like tests/kernel/test_kernel.py's own
    _RaiseNTimesThenCompleteEngine, but each attempt also records that
    attempt's own usage (from `usage_per_attempt`, indexed by
    `counter["attempt"]`) into `self.total_usage` BEFORE deciding
    whether to raise -- mirroring LLMDecisionEngine.decide()'s own
    real ordering (usage is accumulated before any later check that
    might raise; see that class's own docstring). `counter` is a
    plain dict shared across every fresh engine instance Kernel.
    _execute_once() builds for each retry, so the shared attempt index
    and failure countdown survive across instances the same way the
    original fixture's shared counter does.
    """

    def __init__(self, counter: dict, usage_per_attempt: list):
        self._counter = counter
        self._usage_per_attempt = usage_per_attempt
        self.total_usage: TokenUsage | None = None

    def decide(self, context):
        attempt_index = self._counter["attempt"]
        self._counter["attempt"] += 1
        self.total_usage = self._usage_per_attempt[attempt_index]

        if self._counter["remaining_failures"] > 0:
            self._counter["remaining_failures"] -= 1
            raise RuntimeError("Simulated transient decision failure.")

        return AgentAction(action_type=AgentActionType.COMPLETE, reason="Recovered.")


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
# Kernel.run() -- KernelResult.token_usage
# ---------------------------------------------------------------------

def test_kernel_result_token_usage_is_none_when_no_agent_available():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Never matches.",
            can_handle=lambda normalized: False,
            build_agent=lambda: None,
            build_decision_engine=lambda: None,
        )
    )

    result = kernel.run("Do something.")

    assert result.status == "NO_AGENT_AVAILABLE"
    assert result.token_usage is None


def test_kernel_result_token_usage_is_none_when_decision_engine_exposes_none():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Completes immediately, reports no usage.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _UsageReportingCompleteEngine(None),
            )
        )

        result = kernel.run("Do something trivial.")

        assert result.status == "COMPLETED"
        assert result.token_usage is None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_kernel_result_carries_the_completed_runs_token_usage():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Completes immediately.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _UsageReportingCompleteEngine(usage),
            )
        )

        result = kernel.run("Do something trivial.")

        assert result.status == "COMPLETED"
        assert result.token_usage == usage
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_kernel_result_accumulates_token_usage_across_recovery_retries():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        # Attempt 1 fails (but was still genuinely billed); attempt 2
        # (a fresh engine instance, per Kernel._execute_once's own
        # "always a fresh agent + decision engine" rule) succeeds.
        counter = {"remaining_failures": 1, "attempt": 0}
        usage_per_attempt = [
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
        ]

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Fails once (billed), then completes on retry.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=(
                    lambda: _RaiseThenCompleteWithUsageEngine(counter, usage_per_attempt)
                ),
            )
        )

        result = kernel.run("Do something that transiently fails.")

        assert result.status == "COMPLETED"
        assert result.recovery_attempts == 1
        # Both attempts' real cost is present -- the failed first
        # attempt's usage must not be silently dropped just because
        # its own result was ultimately discarded.
        assert result.token_usage == combine_token_usage(*usage_per_attempt)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_kernel_result_combines_primary_and_independent_verification_usage():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        primary_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        verifier_usage = TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)

        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            independent_verifier=WorkflowVerifierRegistration(
                subject="reviewer_agent",
                build_agent=lambda: _build_zero_tool_agent(tmp_dir, subject="reviewer_agent"),
                build_decision_engine=lambda: _UsageReportingCompleteEngine(verifier_usage),
            ),
        )

        kernel.register_agent(
            AgentRegistration(
                subject="writer_agent",
                description="Publishes a report, then completes.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_write_report_tool_agent(tmp_dir),
                build_decision_engine=(
                    lambda: _UsageReportingWriteReportThenCompleteEngine(
                        filename="report.md", usage=primary_usage
                    )
                ),
            )
        )

        result = kernel.run("Draft and publish a report.")

        assert result.status == "COMPLETED"
        assert result.independent_verification is not None
        assert result.independent_verification.status == "COMPLETED"
        # The secondary, automatically-triggered verifier run is a
        # real, separately-billed agent run -- its cost must be folded
        # into the same total, not left invisible.
        assert result.token_usage == combine_token_usage(primary_usage, verifier_usage)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# Kernel.run_workflow() -- WorkflowStepResult.token_usage /
# WorkflowRunResult.token_usage
# ---------------------------------------------------------------------

def test_workflow_run_result_token_usage_is_none_when_no_workflow_available():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Never selected via run_workflow's own can_handle.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir, subject="research_agent"),
                build_decision_engine=lambda: _UsageReportingCompleteEngine(None),
            )
        )

        result = kernel.run_workflow("Nothing matches this.")

        assert result.status == "NO_WORKFLOW_AVAILABLE"
        assert result.token_usage is None
        assert result.completed_steps == ()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_workflow_step_and_run_token_usage_are_summed_across_a_two_step_pipeline():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        step_1_usage = TokenUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16)
        step_2_usage = TokenUsage(prompt_tokens=6, completion_tokens=2, total_tokens=8)

        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Publishes a report-shaped artifact, then completes.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_write_report_tool_agent(
                    tmp_dir, subject="research_agent"
                ),
                build_decision_engine=(
                    lambda: _UsageReportingWriteReportThenCompleteEngine(
                        filename="finding.md", usage=step_1_usage
                    )
                ),
            )
        )
        kernel.register_agent(
            AgentRegistration(
                subject="reviewer_agent",
                description="Completes without touching any tool.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir, subject="reviewer_agent"),
                build_decision_engine=lambda: _UsageReportingCompleteEngine(step_2_usage),
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
                        subject="reviewer_agent", build_task=_step_2_task_from_artifact
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Research and review it.")

        assert result.status == "COMPLETED"
        assert len(result.completed_steps) == 2
        assert result.completed_steps[0].token_usage == step_1_usage
        assert result.completed_steps[1].token_usage == step_2_usage
        assert result.token_usage == combine_token_usage(step_1_usage, step_2_usage)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_workflow_step_token_usage_accumulates_across_that_steps_own_retries():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        step_1_usage = TokenUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16)

        counter = {"remaining_failures": 1, "attempt": 0}
        step_2_usage_per_attempt = [
            TokenUsage(prompt_tokens=9, completion_tokens=3, total_tokens=12),
            TokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        ]

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
                build_decision_engine=(
                    lambda: _UsageReportingWriteReportThenCompleteEngine(
                        filename="finding.md", usage=step_1_usage
                    )
                ),
            )
        )
        kernel.register_agent(
            AgentRegistration(
                subject="reviewer_agent",
                description="Fails once transiently (billed), then completes on retry.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir, subject="reviewer_agent"),
                build_decision_engine=(
                    lambda: _RaiseThenCompleteWithUsageEngine(
                        counter, step_2_usage_per_attempt
                    )
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
                        subject="reviewer_agent", build_task=_step_2_task_from_artifact
                    ),
                ),
            )
        )

        result = kernel.run_workflow("Research and review it.")

        expected_step_2_usage = combine_token_usage(*step_2_usage_per_attempt)

        assert result.status == "COMPLETED"
        assert counter["remaining_failures"] == 0
        assert result.completed_steps[0].token_usage == step_1_usage
        # The step's own token_usage already folds in both of its
        # retry attempts -- separate from loop_result.token_usage,
        # which only ever reflects the final, kept attempt.
        assert result.completed_steps[1].token_usage == expected_step_2_usage
        assert result.completed_steps[1].loop_result.token_usage == (
            step_2_usage_per_attempt[-1]
        )
        assert result.token_usage == combine_token_usage(
            step_1_usage, expected_step_2_usage
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
