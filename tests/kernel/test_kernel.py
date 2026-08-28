"""
Tests for core.kernel.kernel (Kernel v1).

Uses minimal, isolated fixtures (a zero-tool AgentCore, an inline
DeterministicDecisionEngine-style engine, an isolated tempfile-based
permissions.json) rather than the real research_agent stack, so these
tests exercise the Kernel's own mechanics -- registration, NORMALIZE/
CLASSIFY/PLAN/EXECUTE/VERIFY/FINAL RESULT -- independently of any one
agent's real tools. tests/kernel/test_kernel_research_agent_integration.
py covers the full real-stack, real-agent path.
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
    KernelVerification,
    NormalizedTask,
    RetrievedContext,
    WorkflowVerifierRegistration,
)

from core.memory.memory_store import MemoryEntry, MemoryStore

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)

from core.policies.policy_engine import (
    ExternalActionEvaluation,
    PolicyEngine,
    WorkflowTriggerEvaluation,
)

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


class _ImmediateCompleteEngine(AgentDecisionEngine):
    """Completes on the very first decision -- no tool ever invoked."""

    def decide(self, context):
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Nothing to do.",
        )


class _NeverCompleteEngine(AgentDecisionEngine):
    """Always invokes a nonexistent tool -- used to force MAX_STEPS_EXCEEDED
    or, with no such tool registered, EXECUTION_ERROR from the very first
    step. Only used where the test doesn't care which non-terminal-success
    status results, just that it passes through unchanged."""

    def decide(self, context):
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="does_not_exist",
            inputs={},
            reason="Deliberately invoke a tool that isn't registered.",
        )


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


def _write_shell_policy(tmp_dir: Path) -> Path:
    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": "test_agent",
                "resource": "shell",
                "action": "execute",
                "scope": "workspace",
                "risk_level": "HIGH",
                "approval": "policy",
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


def _write_low_risk_search_policy(
    tmp_dir: Path, subject: str = "test_agent"
) -> Path:
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


def _build_low_risk_tool_agent(tmp_dir: Path, subject: str = "test_agent") -> AgentCore:
    """
    A real, LOW-risk, auto-allowed tool (unlike _build_zero_tool_agent,
    which has no tools at all, and _build_shell_tool_agent, whose tool
    always requires approval) -- used specifically to exercise a real
    SUCCESS-path ToolExecutionResult (with a real SecurityDecision)
    through the full Kernel.run() lifecycle.
    """

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
        purpose="A minimal agent used only to exercise Kernel mechanics.",
    )

    return AgentCore(identity=identity, tools=interface)


def _write_write_report_policy(
    tmp_dir: Path, subject: str = "writer_agent"
) -> Path:
    """
    A synthetic LOW-risk/no-approval permission for a fake "write_report"
    tool, isolated to `tmp_dir` -- same isolation pattern as every other
    _write_*_policy helper in this file. LOW/none (rather than the real
    write_report tool's actual HIGH/policy) is a deliberate
    simplification here: these Kernel-level tests exist to exercise
    _trigger_independent_verification's own mechanics (Build Phase 12),
    not the approval gate itself, which tests/tools/implementations/
    test_write_report_tool.py already covers directly.
    """

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


def _build_write_report_tool_agent(
    tmp_dir: Path, subject: str = "writer_agent"
) -> AgentCore:
    """
    A real, LOW-risk, auto-allowed "write_report"-id tool whose
    executor returns a real `{"path": ..., "size_bytes": ...}` artifact
    -- the exact shape the real write_report tool's own executor
    returns (core/tools/implementations/write_report_tool.py) -- so a
    resulting ToolExecutionResult.artifacts[0] exercises
    Kernel._trigger_independent_verification's own artifact-path
    extraction the same way the real tool would, without needing the
    real tool's sandboxing/approval machinery here.
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
            risk_level="LOW",
            error_handling={
                "retryable": False,
                "on_failure": "Surface the write error to the agent.",
            },
        )
    )

    policy_path = _write_write_report_policy(tmp_dir, subject=subject)

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
        purpose="A minimal agent used only to exercise Kernel mechanics.",
    )

    return AgentCore(identity=identity, tools=interface)


class _WriteReportThenCompleteEngine(AgentDecisionEngine):
    """Invokes write_report exactly once, then completes."""

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


class _SearchThenCompleteEngine(AgentDecisionEngine):
    """Invokes web_search exactly once, then completes."""

    def __init__(self):
        self._invoked = False

    def decide(self, context):
        if not self._invoked:
            self._invoked = True

            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="web_search",
                inputs={"query": "AI agents"},
                reason="Search before completing.",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Search complete.",
        )


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
        purpose="A minimal agent used only to exercise Kernel mechanics.",
    )

    return AgentCore(identity=identity, tools=interface)


def _build_shell_tool_agent(tmp_dir: Path, subject: str = "test_agent") -> AgentCore:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="shell",
            name="Shell",
            purpose="Execute shell commands.",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            output_schema={"type": "string"},
            permissions=(f"{subject}:shell:execute:workspace",),
            resource="shell",
            action="execute",
            scope="workspace",
            risk_level="HIGH",
            error_handling={
                "retryable": False,
                "on_failure": "Surface for human review.",
            },
        )
    )

    policy_path = _write_shell_policy(tmp_dir)

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(
        tool_id="shell",
        executor=lambda command: "SHOULD NOT EXECUTE WITHOUT APPROVAL",
    )

    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject=subject,
        name="Test Agent",
        purpose="A minimal agent used only to exercise Kernel mechanics.",
    )

    return AgentCore(identity=identity, tools=interface)


class _RequestShellEngine(AgentDecisionEngine):
    def decide(self, context):
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="shell",
            inputs={"command": "echo test"},
            reason="Attempt a HIGH-risk operation without approval.",
        )


class _RaiseNTimesThenCompleteEngine(AgentDecisionEngine):
    """Raises on decide() while `counter["remaining_failures"]` is
    positive, then completes.

    `counter` is a plain dict shared by the caller (not owned by any
    one engine instance) so that a *fresh* engine built for each
    RECOVER IF NEEDED retry -- Kernel._execute_once() always calls
    AgentRegistration.build_decision_engine() again, per this
    project's stateful-agent design -- still observes state left by
    the previous attempt. This is what actually simulates a transient
    failure (e.g. a network blip) that resolves itself on retry,
    rather than one that would resolve itself even without any
    Kernel-level retry logic.
    """

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


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------

def test_register_agent_rejects_duplicate_subject():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    registration = AgentRegistration(
        subject="test_agent",
        description="A test agent.",
        can_handle=lambda normalized: True,
        build_agent=lambda: None,
        build_decision_engine=lambda: None,
    )

    kernel.register_agent(registration)

    with pytest.raises(ValueError, match="already registered"):
        kernel.register_agent(registration)


# ---------------------------------------------------------------------
# Full Kernel.run() lifecycle, real (but minimal) agent/security stack
# ---------------------------------------------------------------------

def test_run_returns_no_agent_available_when_nothing_matches():
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
    assert result.subject is None
    assert result.loop_result is None


def test_run_completes_and_verifies_when_no_tool_is_ever_invoked():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Completes immediately.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        result = kernel.run("Do something trivial.")

        assert result.status == "COMPLETED"
        assert result.subject == "test_agent"
        assert result.verification is not None
        assert result.verification.passed is True
        assert result.loop_result.status == "COMPLETED"
        # No tool was ever invoked -- there is no external action for
        # the Policy Layer to answer the six questions about.
        assert result.policy_evaluation is None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_surfaces_approval_required_as_awaiting_approval():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Requests a HIGH-risk tool with no approval.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_shell_tool_agent(tmp_dir),
                build_decision_engine=lambda: _RequestShellEngine(),
            )
        )

        result = kernel.run("Run a shell command.")

        assert result.status == "AWAITING_APPROVAL"
        assert result.loop_result.status == "APPROVAL_REQUIRED"
        # The Kernel must never have silently resolved this itself.
        assert result.loop_result.last_result.status == "APPROVAL_REQUIRED"
        # The Policy Layer's six-question answer must reflect the real,
        # HIGH-risk, approval-required decision the Security Layer
        # actually made for this tool call.
        assert result.policy_evaluation is not None
        assert result.policy_evaluation.action == "execute"
        assert result.policy_evaluation.subject == "test_agent"
        assert result.policy_evaluation.tool_id == "shell"
        assert result.policy_evaluation.risk_level == "HIGH"
        assert result.policy_evaluation.approval_required is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_passes_through_non_completed_statuses_unchanged():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Always invokes a tool that doesn't exist.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _NeverCompleteEngine(),
            )
        )

        result = kernel.run("Do something impossible.")

        # Whatever the loop reports for an unregistered tool id, the
        # Kernel must report the same status verbatim -- it invents no
        # new vocabulary for cases the loop already names.
        assert result.status == result.loop_result.status
        assert result.status != "COMPLETED"
        assert result.verification.passed is False
        # An unregistered tool id produces EXECUTION_ERROR (confirmed
        # empirically, not assumed), which IS one of the two
        # RECOVER-IF-NEEDED-eligible statuses -- so with the default
        # max_recovery_attempts=1 this run retries exactly once (a
        # fresh agent + decision engine, per _execute_once) before
        # still surfacing the same EXECUTION_ERROR, since retrying an
        # identical always-broken plan reproduces the same outcome.
        # Asserted explicitly here so the retry-then-still-fail
        # behavior is tested, not just incidentally correct.
        assert result.status == "EXECUTION_ERROR"
        assert result.recovery_attempts == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_builds_a_fresh_agent_and_decision_engine_every_call():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        agent_build_count = {"n": 0}
        engine_build_count = {"n": 0}

        def build_agent():
            agent_build_count["n"] += 1
            return _build_zero_tool_agent(tmp_dir)

        def build_decision_engine():
            engine_build_count["n"] += 1
            return _ImmediateCompleteEngine()

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Counts how many times it's built.",
                can_handle=lambda normalized: True,
                build_agent=build_agent,
                build_decision_engine=build_decision_engine,
            )
        )

        kernel.run("First task.")
        kernel.run("Second task.")

        assert agent_build_count["n"] == 2
        assert engine_build_count["n"] == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# NORMALIZE
# ---------------------------------------------------------------------

def test_normalize_rejects_empty_task():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    with pytest.raises(ValueError, match="must not be empty"):
        kernel.run("   ")


def test_normalize_strips_surrounding_whitespace():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    normalized = kernel._normalize("  do the thing  ")

    assert normalized == NormalizedTask(text="do the thing")


# ---------------------------------------------------------------------
# VERIFY (unit-level: constructs AgentLoopResult directly, since the
# real AgentExecutionLoop can never itself produce a COMPLETED result
# whose last tool call did not succeed -- see this module's own
# reasoning in core/kernel/kernel.py's KernelVerification docstring).
# ---------------------------------------------------------------------

def test_verify_passes_when_completed_with_no_tool_ever_run():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    loop_result = AgentLoopResult(
        status="COMPLETED",
        steps=1,
        last_result=None,
        reason="Done.",
        context=AgentContext(task="x"),
    )

    verification = kernel._verify(loop_result)

    assert verification.passed is True


def test_verify_fails_when_completed_but_last_tool_result_was_not_success():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    # A minimal stand-in for ToolExecutionResult: Kernel._verify() only
    # ever reads `.status` off the last tool result (via getattr, to
    # stay tolerant of exactly this kind of substitute), so a full
    # ToolExecutionResult (which also requires a real SecurityDecision)
    # isn't needed to exercise this branch in isolation.
    failed_tool_result = SimpleNamespace(status="ERROR")

    loop_result = AgentLoopResult(
        status="COMPLETED",
        steps=2,
        last_result=failed_tool_result,
        reason="Done.",
        context=AgentContext(task="x"),
    )

    verification = kernel._verify(loop_result)

    assert verification.passed is False


def test_verify_does_not_apply_to_non_completed_results():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    loop_result = AgentLoopResult(
        status="FAILED",
        steps=1,
        last_result=None,
        reason="Failed.",
        context=AgentContext(task="x"),
    )

    verification = kernel._verify(loop_result)

    assert verification.passed is False


# ---------------------------------------------------------------------
# RECOVER IF NEEDED
# ---------------------------------------------------------------------

def test_kernel_rejects_non_integer_max_recovery_attempts():
    with pytest.raises(
        TypeError,
        match="max_recovery_attempts must be an integer",
    ):
        Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            max_recovery_attempts="1",
        )


def test_kernel_rejects_negative_max_recovery_attempts():
    with pytest.raises(
        ValueError,
        match="max_recovery_attempts must be zero or greater",
    ):
        Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            max_recovery_attempts=-1,
        )


def test_should_recover_returns_true_only_for_decision_or_execution_error():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    def make(status: str) -> AgentLoopResult:
        return AgentLoopResult(
            status=status,
            steps=1,
            last_result=None,
            reason=None,
            context=AgentContext(task="x"),
        )

    assert kernel._should_recover(make("DECISION_ERROR")) is True
    assert kernel._should_recover(make("EXECUTION_ERROR")) is True

    # Deliberate outcomes the loop reported on purpose -- retrying
    # would mean second-guessing the agent/loop, not recovering from
    # an unexpected crash.
    assert kernel._should_recover(make("FAILED")) is False
    assert kernel._should_recover(make("TOOL_ERROR")) is False
    assert kernel._should_recover(make("APPROVAL_REQUIRED")) is False
    assert kernel._should_recover(make("MAX_STEPS_EXCEEDED")) is False
    assert kernel._should_recover(make("INVALID_ACTION")) is False
    assert kernel._should_recover(make("COMPLETED")) is False


def test_recovery_retries_and_succeeds_after_a_transient_decision_error():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        # Fails on the first decide() call, succeeds on the second --
        # simulating a transient failure (e.g. a network blip) that
        # resolves itself on a fresh retry.
        counter = {"remaining_failures": 1}

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Fails once, then completes on retry.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=(
                    lambda: _RaiseNTimesThenCompleteEngine(counter)
                ),
            )
        )

        result = kernel.run("Do something that transiently fails.")

        assert result.status == "COMPLETED"
        assert result.recovery_attempts == 1
        assert result.loop_result.status == "COMPLETED"
        assert result.verification.passed is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_recovery_gives_up_after_max_recovery_attempts_and_reports_the_failure():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            max_recovery_attempts=2,
        )

        # Always fails -- exercises the bound itself, not recovery.
        counter = {"remaining_failures": 999}

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Always fails.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=(
                    lambda: _RaiseNTimesThenCompleteEngine(counter)
                ),
            )
        )

        result = kernel.run("Do something that always fails.")

        # Exactly max_recovery_attempts retries happened -- no more,
        # no fewer -- and the original failure is still reported
        # verbatim, never silently swallowed.
        assert result.status == "DECISION_ERROR"
        assert result.recovery_attempts == 2
        assert result.loop_result.status == "DECISION_ERROR"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_max_recovery_attempts_zero_disables_recovery():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            max_recovery_attempts=0,
        )

        counter = {"remaining_failures": 999}

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Always fails.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=(
                    lambda: _RaiseNTimesThenCompleteEngine(counter)
                ),
            )
        )

        result = kernel.run("Do something that always fails.")

        assert result.status == "DECISION_ERROR"
        assert result.recovery_attempts == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_kernel_defaults_to_a_real_policy_engine():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    assert isinstance(kernel.policy_engine, PolicyEngine)


def test_should_recover_genuinely_delegates_to_the_injected_policy_engine():
    """
    Proves _should_recover() actually calls out to
    self.policy_engine.is_recovery_authorized() rather than deciding
    on its own -- a policy engine that authorizes recovery for a
    status Kernel's own default PolicyEngine never would (FAILED) is
    obeyed, and one that denies recovery for a status the default
    PolicyEngine always would (EXECUTION_ERROR) is also obeyed. This
    is what makes RECOVER IF NEEDED a real Policy Layer decision
    (POLICY_SPEC.md's Failure Policy step 4 / "Policy Enforcement"
    section) rather than a hardcoded Kernel heuristic in disguise.
    """

    class _AuthorizeEverythingPolicyEngine(PolicyEngine):
        def is_recovery_authorized(self, status: str) -> bool:
            return True

    class _AuthorizeNothingPolicyEngine(PolicyEngine):
        def is_recovery_authorized(self, status: str) -> bool:
            return False

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # A policy engine that authorizes recovery even for FAILED (a
        # deliberate agent decision the default PolicyEngine never
        # retries) causes a real retry when injected.
        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            policy_engine=_AuthorizeEverythingPolicyEngine(),
        )

        agent_build_count = {"n": 0}

        def build_failing_agent():
            agent_build_count["n"] += 1
            return _build_zero_tool_agent(tmp_dir)

        class _FailEngine(AgentDecisionEngine):
            def decide(self, context):
                return AgentAction(
                    action_type=AgentActionType.FAIL,
                    reason="Deliberate failure.",
                )

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Always fails deliberately.",
                can_handle=lambda normalized: True,
                build_agent=build_failing_agent,
                build_decision_engine=lambda: _FailEngine(),
            )
        )

        result = kernel.run("Do something that deliberately fails.")

        assert result.status == "FAILED"
        assert result.recovery_attempts == 1
        assert agent_build_count["n"] == 2

        # A policy engine that authorizes nothing suppresses recovery
        # even for EXECUTION_ERROR (a status the default PolicyEngine
        # always authorizes recovery for).
        kernel2 = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            policy_engine=_AuthorizeNothingPolicyEngine(),
        )

        kernel2.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Always invokes a tool that doesn't exist.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _NeverCompleteEngine(),
            )
        )

        result2 = kernel2.run("Do something impossible.")

        assert result2.status == "EXECUTION_ERROR"
        assert result2.recovery_attempts == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_recovery_does_not_trigger_for_a_deliberate_approval_required_outcome():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        agent_build_count = {"n": 0}
        engine_build_count = {"n": 0}

        def build_agent():
            agent_build_count["n"] += 1
            return _build_shell_tool_agent(tmp_dir)

        def build_decision_engine():
            engine_build_count["n"] += 1
            return _RequestShellEngine()

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Requests a HIGH-risk tool with no approval.",
                can_handle=lambda normalized: True,
                build_agent=build_agent,
                build_decision_engine=build_decision_engine,
            )
        )

        result = kernel.run("Run a shell command.")

        # APPROVAL_REQUIRED is a deliberate outcome, not a crash --
        # retrying wouldn't change anything (still waiting on a
        # human), so the Kernel must not have retried at all: exactly
        # one agent and one decision engine were ever built.
        assert result.status == "AWAITING_APPROVAL"
        assert result.recovery_attempts == 0
        assert agent_build_count["n"] == 1
        assert engine_build_count["n"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# POLICY EVALUATION (Build Phase 7: PolicyEngine.evaluate_external_
# action() wired into a real Kernel call site -- see Kernel.
# _evaluate_policy's own docstring)
# ---------------------------------------------------------------------

def test_run_answers_the_policy_six_questions_for_a_successful_tool_call():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Searches once, then completes.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_low_risk_tool_agent(tmp_dir),
                build_decision_engine=lambda: _SearchThenCompleteEngine(),
            )
        )

        result = kernel.run("Search for something.")

        assert result.status == "COMPLETED"
        assert result.loop_result.last_result.status == "SUCCESS"
        assert result.policy_evaluation is not None
        assert result.policy_evaluation.action == "search"
        assert result.policy_evaluation.subject == "test_agent"
        assert result.policy_evaluation.tool_id == "web_search"
        assert result.policy_evaluation.risk_level == "LOW"
        assert result.policy_evaluation.approval_required is False
        assert result.policy_evaluation.verification_required is True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_evaluate_policy_returns_none_when_no_tool_was_invoked():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    loop_result = AgentLoopResult(
        status="COMPLETED",
        steps=1,
        last_result=None,
        reason="Done.",
        context=AgentContext(task="x"),
    )

    assert kernel._evaluate_policy(loop_result) is None


def test_evaluate_policy_degrades_to_none_when_security_decision_is_incomplete():
    """
    _evaluate_policy must never let a Kernel.run() call crash over
    incomplete identifying/security data -- see its own docstring for
    why (this project's standing constraint that the system must
    never become so strict it refuses to execute anything). A
    SimpleNamespace last_result with no real SecurityDecision (e.g. a
    caller-supplied ToolRuntime/AgentCore substitute outside this
    project's own ToolGateway -- see tests/agents/
    test_agent_llm_integration.py's own MockToolRuntime, which can
    construct exactly this shape) triggers PolicyEngine.
    evaluate_external_action()'s ValueError, which this method must
    catch and turn into None rather than propagate.
    """

    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    incomplete_tool_result = SimpleNamespace(
        status="SUCCESS",
        subject="test_agent",
        tool_id="web_search",
        action="search",
        security_decision=None,
    )

    loop_result = AgentLoopResult(
        status="COMPLETED",
        steps=1,
        last_result=incomplete_tool_result,
        reason="Done.",
        context=AgentContext(task="x"),
    )

    assert kernel._evaluate_policy(loop_result) is None


def test_evaluate_policy_genuinely_delegates_to_the_injected_policy_engine():
    """
    Proves _evaluate_policy() actually calls out to
    self.policy_engine.evaluate_external_action() rather than
    answering the six questions itself -- an injected PolicyEngine
    subclass that overrides evaluate_external_action() to return a
    fixed, otherwise-impossible answer is exactly what the Kernel
    surfaces, the same genuine-delegation proof already applied to
    _should_recover (see test_should_recover_genuinely_delegates_to_
    the_injected_policy_engine above).
    """

    class _FixedAnswerPolicyEngine(PolicyEngine):
        def evaluate_external_action(self, **kwargs):
            return ExternalActionEvaluation(
                action="INJECTED",
                subject="INJECTED",
                tool_id="INJECTED",
                risk_level="INJECTED",
                approval_required=True,
                verification_required=False,
            )

    kernel = Kernel(
        orchestration_engine=SequentialOrchestrationEngine(),
        policy_engine=_FixedAnswerPolicyEngine(),
    )

    real_shaped_tool_result = SimpleNamespace(
        status="SUCCESS",
        subject="test_agent",
        tool_id="web_search",
        action="search",
        security_decision=SimpleNamespace(
            authorization=SimpleNamespace(effective_risk="LOW"),
            approval=SimpleNamespace(required=False),
        ),
    )

    loop_result = AgentLoopResult(
        status="COMPLETED",
        steps=1,
        last_result=real_shaped_tool_result,
        reason="Done.",
        context=AgentContext(task="x"),
    )

    evaluation = kernel._evaluate_policy(loop_result)

    assert evaluation == ExternalActionEvaluation(
        action="INJECTED",
        subject="INJECTED",
        tool_id="INJECTED",
        risk_level="INJECTED",
        approval_required=True,
        verification_required=False,
    )


# ---------------------------------------------------------------------
# WorkflowVerifierRegistration -- Build Phase 12's own registration
# dataclass, validated the same way AgentRegistration.__post_init__ is.
# ---------------------------------------------------------------------

def test_workflow_verifier_registration_rejects_empty_subject():
    with pytest.raises(ValueError, match="subject must be"):
        WorkflowVerifierRegistration(
            subject="",
            build_agent=lambda: None,
            build_decision_engine=lambda: None,
        )


def test_workflow_verifier_registration_rejects_non_callable_build_agent():
    with pytest.raises(TypeError, match="build_agent must be"):
        WorkflowVerifierRegistration(
            subject="reviewer_agent",
            build_agent="not callable",
            build_decision_engine=lambda: None,
        )


def test_workflow_verifier_registration_rejects_non_callable_build_decision_engine():
    with pytest.raises(TypeError, match="build_decision_engine must be"):
        WorkflowVerifierRegistration(
            subject="reviewer_agent",
            build_agent=lambda: None,
            build_decision_engine="not callable",
        )


def test_kernel_rejects_a_non_registration_independent_verifier():
    with pytest.raises(TypeError, match="WorkflowVerifierRegistration"):
        Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            independent_verifier="not a registration",
        )


# ---------------------------------------------------------------------
# INDEPENDENT VERIFICATION (Build Phase 12: PolicyEngine.
# evaluate_workflow_trigger() wired into a real Kernel call site -- see
# Kernel._trigger_independent_verification's own docstring)
# ---------------------------------------------------------------------

def test_independent_verification_is_none_when_no_verifier_is_configured():
    """
    The default, unconfigured case: even though writer_agent's
    write_report call here genuinely matches the one declared
    transition, an unconfigured Kernel (independent_verifier=None, the
    default) must never trigger anything -- see Kernel.__init__'s own
    docstring for why this purely-additive capability must be
    completely inert unless a caller explicitly opts in.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="writer_agent",
                description="Publishes a report, then completes.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_write_report_tool_agent(tmp_dir),
                build_decision_engine=lambda: _WriteReportThenCompleteEngine(),
            )
        )

        result = kernel.run("Draft and publish a report.")

        assert result.status == "COMPLETED"
        assert result.independent_verification is None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_independent_verification_triggers_reviewer_agent_after_a_successful_write_report():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        verifier_build_count = {"n": 0}

        def build_verifier_agent():
            verifier_build_count["n"] += 1
            return _build_zero_tool_agent(tmp_dir, subject="reviewer_agent")

        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            independent_verifier=WorkflowVerifierRegistration(
                subject="reviewer_agent",
                build_agent=build_verifier_agent,
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            ),
        )

        kernel.register_agent(
            AgentRegistration(
                subject="writer_agent",
                description="Publishes a report, then completes.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_write_report_tool_agent(tmp_dir),
                build_decision_engine=lambda: _WriteReportThenCompleteEngine(
                    filename="report.md"
                ),
            )
        )

        result = kernel.run("Draft and publish a report.")

        # The primary task's own outcome is completely unaffected by
        # the secondary verification step.
        assert result.status == "COMPLETED"
        assert result.subject == "writer_agent"

        assert result.independent_verification is not None
        assert result.independent_verification.status == "COMPLETED"
        assert verifier_build_count["n"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_independent_verification_not_triggered_for_an_unrelated_completed_subject():
    """
    A configured independent_verifier for "reviewer_agent" must not
    fire just because *some* agent completed a tool call -- only the
    one declared transition (writer_agent's write_report SUCCESS)
    triggers it. Here the primary agent is registered under subject
    "research_agent" using the same write_report-shaped tool, which
    PolicyEngine.evaluate_workflow_trigger() does not recognize as a
    trigger for that subject.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        verifier_build_count = {"n": 0}

        def build_verifier_agent():
            verifier_build_count["n"] += 1
            return _build_zero_tool_agent(tmp_dir, subject="reviewer_agent")

        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            independent_verifier=WorkflowVerifierRegistration(
                subject="reviewer_agent",
                build_agent=build_verifier_agent,
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            ),
        )

        kernel.register_agent(
            AgentRegistration(
                subject="research_agent",
                description="Publishes a report, then completes.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_write_report_tool_agent(
                    tmp_dir, subject="research_agent"
                ),
                build_decision_engine=lambda: _WriteReportThenCompleteEngine(),
            )
        )

        result = kernel.run("Draft and publish a report.")

        assert result.status == "COMPLETED"
        assert result.independent_verification is None
        assert verifier_build_count["n"] == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_independent_verification_returns_none_when_no_tool_was_invoked():
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    loop_result = AgentLoopResult(
        status="COMPLETED",
        steps=1,
        last_result=None,
        reason="Done.",
        context=AgentContext(task="x"),
    )

    assert (
        kernel._trigger_independent_verification(
            completed_subject="writer_agent",
            loop_result=loop_result,
            max_steps=10,
        )
        is None
    )


def test_independent_verification_degrades_to_none_when_artifact_has_no_usable_path():
    """
    Mirrors test_evaluate_policy_degrades_to_none_when_security_
    decision_is_incomplete's own reasoning: a caller-supplied
    ToolRuntime/AgentCore substitute outside this project's own
    ToolGateway may report a SUCCESS write_report result with an
    artifact shape that doesn't carry a usable "path" -- this must
    degrade to None, never raise.
    """
    kernel = Kernel(
        orchestration_engine=SequentialOrchestrationEngine(),
        independent_verifier=WorkflowVerifierRegistration(
            subject="reviewer_agent",
            build_agent=lambda: (_ for _ in ()).throw(
                AssertionError("must not be called")
            ),
            build_decision_engine=lambda: (_ for _ in ()).throw(
                AssertionError("must not be called")
            ),
        ),
    )

    incomplete_tool_result = SimpleNamespace(
        status="SUCCESS",
        subject="writer_agent",
        tool_id="write_report",
        action="write",
        artifacts=({"no_path_here": True},),
    )

    loop_result = AgentLoopResult(
        status="COMPLETED",
        steps=1,
        last_result=incomplete_tool_result,
        reason="Done.",
        context=AgentContext(task="x"),
    )

    assert (
        kernel._trigger_independent_verification(
            completed_subject="writer_agent",
            loop_result=loop_result,
            max_steps=10,
        )
        is None
    )


def test_independent_verification_genuinely_delegates_to_the_injected_policy_engine():
    """
    Proves _trigger_independent_verification() actually calls out to
    self.policy_engine.evaluate_workflow_trigger() rather than
    hardcoding "write_report"/"SUCCESS" itself -- the same
    genuine-delegation proof already applied to _should_recover and
    _evaluate_policy (see those tests above): an injected PolicyEngine
    subclass whose evaluate_workflow_trigger() always names
    "reviewer_agent" as the next subject, for an otherwise-impossible
    (tool_id/status) combination, is exactly what the Kernel obeys.
    """

    class _AlwaysTriggerReviewerPolicyEngine(PolicyEngine):
        def evaluate_workflow_trigger(self, **kwargs):
            return WorkflowTriggerEvaluation(
                completed_subject=kwargs["completed_subject"],
                tool_id=kwargs["tool_id"],
                tool_status=kwargs["tool_status"],
                should_trigger=True,
                next_subject="reviewer_agent",
            )

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        verifier_build_count = {"n": 0}

        def build_verifier_agent():
            verifier_build_count["n"] += 1
            return _build_zero_tool_agent(tmp_dir, subject="reviewer_agent")

        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            policy_engine=_AlwaysTriggerReviewerPolicyEngine(),
            independent_verifier=WorkflowVerifierRegistration(
                subject="reviewer_agent",
                build_agent=build_verifier_agent,
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            ),
        )

        incomplete_tool_result = SimpleNamespace(
            status="ANYTHING",
            subject="some_agent",
            tool_id="some_tool",
            action="execute",
            artifacts=({"path": "whatever.md"},),
        )

        loop_result = AgentLoopResult(
            status="COMPLETED",
            steps=1,
            last_result=incomplete_tool_result,
            reason="Done.",
            context=AgentContext(task="x"),
        )

        independent_verification = kernel._trigger_independent_verification(
            completed_subject="some_agent",
            loop_result=loop_result,
            max_steps=10,
        )

        assert independent_verification is not None
        assert independent_verification.status == "COMPLETED"
        assert verifier_build_count["n"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# Kernel._retrieve_context / KernelResult.retrieved_context
# (Build Phase 14 -- see core/memory/MEMORY_SPEC.md and
# core/kernel/kernel.py's own RetrievedContext docstring)
# ---------------------------------------------------------------------

def test_kernel_defaults_to_no_memory_store_and_retrieved_context_is_none():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())
        assert kernel.memory_store is None

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Completes immediately.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        result = kernel.run("Do something trivial.")

        assert result.status == "COMPLETED"
        assert result.retrieved_context is None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_kernel_rejects_a_non_memory_store():
    with pytest.raises(TypeError, match="memory_store"):
        Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            memory_store="not a memory store",
        )


def test_kernel_rejects_non_integer_context_retrieval_limit():
    with pytest.raises(TypeError, match="context_retrieval_limit"):
        Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            context_retrieval_limit="5",
        )


def test_kernel_rejects_non_positive_context_retrieval_limit():
    with pytest.raises(ValueError, match="context_retrieval_limit"):
        Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            context_retrieval_limit=0,
        )


def test_retrieve_context_returns_real_matching_records_when_configured():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        store.write(
            MemoryEntry(
                subject="research_agent",
                kind="note",
                content="The quarterly report is finished.",
            )
        )

        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            memory_store=store,
        )

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Completes immediately.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        result = kernel.run("Please summarize the quarterly report.")

        assert result.status == "COMPLETED"
        assert result.retrieved_context is not None
        assert isinstance(result.retrieved_context, RetrievedContext)
        assert result.retrieved_context.query == (
            "Please summarize the quarterly report."
        )
        assert len(result.retrieved_context.records) == 1
        assert result.retrieved_context.records[0].content == (
            "The quarterly report is finished."
        )
        assert result.retrieved_context.records[0].verified is False

        # Untrusted-context guarantee: retrieval must never rewrite the
        # task actually executed.
        assert result.loop_result.context.task == (
            "Please summarize the quarterly report."
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_retrieve_context_returns_empty_records_when_nothing_matches():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        store.write(
            MemoryEntry(
                subject="research_agent",
                kind="note",
                content="Completely unrelated content.",
            )
        )

        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            memory_store=store,
        )

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Completes immediately.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        result = kernel.run("Something about quarterly reports.")

        assert result.retrieved_context is not None
        assert result.retrieved_context.records == ()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_retrieve_context_is_populated_even_when_no_agent_matches():
    """
    CONTEXT RETRIEVAL happens before CLASSIFY in Kernel.run() (per
    KERNEL_SPEC.md's own lifecycle ordering) -- retrieved_context
    should still be real, inspectable data on a NO_AGENT_AVAILABLE
    result, not silently dropped just because nothing matched.
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        store.write(
            MemoryEntry(
                subject="research_agent",
                kind="note",
                content="A note about widgets.",
            )
        )

        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            memory_store=store,
        )

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Never matches.",
                can_handle=lambda normalized: False,
                build_agent=lambda: None,
                build_decision_engine=lambda: None,
            )
        )

        result = kernel.run("Tell me about widgets.")

        assert result.status == "NO_AGENT_AVAILABLE"
        assert result.retrieved_context is not None
        assert len(result.retrieved_context.records) == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_retrieve_context_degrades_to_none_when_search_raises():
    """
    Genuine-delegation / fail-safe proof: _retrieve_context must
    degrade to None (never raise, never fail an otherwise-real Kernel
    run) when the configured memory_store's own search() raises --
    the same tolerance _evaluate_policy/_trigger_independent_verification
    already established for their own optional, additive steps.
    """

    class _AlwaysRaisingMemoryStore(MemoryStore):
        def search(self, *args, **kwargs):
            raise ValueError("simulated search failure")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            memory_store=_AlwaysRaisingMemoryStore(
                str(tmp_dir / "memory.jsonl")
            ),
        )

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Completes immediately.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        result = kernel.run("Do something trivial.")

        assert result.status == "COMPLETED"
        assert result.retrieved_context is None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_context_retrieval_limit_is_passed_through_to_search():
    """
    Genuine-delegation proof: Kernel.context_retrieval_limit must
    actually reach MemoryStore.search()'s own `limit`, not be a
    constructor argument nothing reads.
    """

    captured: dict[str, object] = {}

    class _CapturingMemoryStore(MemoryStore):
        def search(self, query, *, limit=10, **kwargs):
            captured["limit"] = limit
            return ()

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(
            orchestration_engine=SequentialOrchestrationEngine(),
            memory_store=_CapturingMemoryStore(str(tmp_dir / "memory.jsonl")),
            context_retrieval_limit=3,
        )

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Completes immediately.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        kernel.run("Do something trivial.")

        assert captured["limit"] == 3
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
