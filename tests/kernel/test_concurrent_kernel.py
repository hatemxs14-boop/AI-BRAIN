"""
Tests for core.kernel.concurrent_kernel (Build Phase 21: this
project's first real concurrent/parallel request handling).

Uses the exact same minimal, isolated fixture style
tests/kernel/test_kernel.py established (a zero-tool AgentCore, an
inline AgentDecisionEngine, an isolated tempfile-based
permissions.json) -- these tests exercise ConcurrentKernelRunner's own
mechanics (real concurrency, ordering, per-task error isolation,
validation, shutdown), independently of any one agent's real tools.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.kernel.concurrent_kernel import (
    ConcurrentKernelRunner,
    ConcurrentTaskResult,
)

from core.kernel.kernel import (
    AgentRegistration,
    Kernel,
    WorkflowDefinition,
    WorkflowStep,
)

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


_POLICY_WRITE_LOCK = threading.Lock()


def _write_empty_policy(tmp_dir: Path) -> Path:
    """
    Write (once) a permissions.json with no permissions at all.

    Guarded by `_POLICY_WRITE_LOCK` and made idempotent (skips the
    write entirely if the file already exists): several of this
    module's own tests register a `build_agent` factory that calls
    this indirectly (via `_build_zero_tool_agent`) once per task, and
    ConcurrentKernelRunner is specifically what now lets several of
    those factory calls run on different threads at the same time --
    without this guard, two threads could concurrently truncate/
    rewrite the SAME `permissions.json` (Path.write_text is not an
    atomic replace), and a third thread's SecurityDecisionPoint could
    then read a half-written file mid-write. This is a test-fixture
    concern only (every real, non-test build_agent in this project
    reads an already-fully-written permissions.json from disk -- see
    core/kernel/default_kernel.py -- it never writes one itself), but
    it must not be allowed to flake this module's own tests.
    """

    policy_path = tmp_dir / "permissions.json"

    with _POLICY_WRITE_LOCK:
        if not policy_path.exists():
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
        purpose="A minimal agent used only to exercise concurrent Kernel mechanics.",
    )

    return AgentCore(identity=identity, tools=interface)


class _ImmediateCompleteEngine(AgentDecisionEngine):
    """Completes on the very first decision -- no tool ever invoked."""

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Nothing to do.",
        )


class _AlwaysFailEngine(AgentDecisionEngine):
    """Deliberately fails on the very first decision."""

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.FAIL,
            reason="Deliberate failure.",
        )


class _RaisingDecisionEngineFactory:
    """
    A build_decision_engine callable that raises for one specific task
    text and returns a normal _ImmediateCompleteEngine for every other
    -- used to prove one task's own exception never disrupts any other
    task's result (see test_one_tasks_exception_does_not_affect_others
    below). Cannot inspect the task text from a decision-engine factory
    directly (it is a zero-argument callable, per AgentRegistration's
    own docstring) -- so this instead keys off call order combined with
    a lock-protected counter, deterministic because
    ConcurrentKernelRunner._run_batch submits one _execute_once call
    per task, each of which calls build_decision_engine() exactly once.
    """

    def __init__(self, raise_on_call_number: int) -> None:
        self._raise_on_call_number = raise_on_call_number
        self._lock = threading.Lock()
        self._call_count = 0

    def __call__(self) -> AgentDecisionEngine:
        with self._lock:
            self._call_count += 1
            call_number = self._call_count

        if call_number == self._raise_on_call_number:
            raise RuntimeError("Deliberate build_decision_engine failure.")

        return _ImmediateCompleteEngine()


class _SleepThenCompleteEngine(AgentDecisionEngine):
    """
    Sleeps for `delay_seconds` (parsed from the task text itself, since
    a single shared build_decision_engine factory has no other way to
    vary behavior per task -- see AgentContext.task) before completing.
    Used to prove real concurrency (wall-clock time for N tasks well
    under N * delay) and that `max_workers` is a real, enforced cap
    (wall-clock time for N tasks run one-at-a-time is close to
    N * delay).

    Task text convention: "sleep <seconds> <anything>" -- the delay is
    always the second whitespace-separated token.
    """

    def decide(self, context: AgentContext) -> AgentAction:
        delay_seconds = float(context.task.split()[1])
        time.sleep(delay_seconds)
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason=f"Completed after sleeping {delay_seconds}s.",
        )


def _sleepy_kernel(tmp_dir: Path) -> Kernel:
    kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

    kernel.register_agent(
        AgentRegistration(
            subject="test_agent",
            description="Sleeps for a task-specified duration, then completes.",
            can_handle=lambda normalized: True,
            build_agent=lambda: _build_zero_tool_agent(tmp_dir),
            build_decision_engine=lambda: _SleepThenCompleteEngine(),
        )
    )

    return kernel


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_rejects_a_non_kernel():
    with pytest.raises(TypeError, match="Kernel"):
        ConcurrentKernelRunner(kernel=object())


def test_rejects_a_non_integer_max_workers():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        with pytest.raises(TypeError, match="max_workers"):
            ConcurrentKernelRunner(kernel=kernel, max_workers="4")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rejects_a_bool_max_workers():
    # bool is a subclass of int in Python -- must be rejected
    # explicitly, exactly like every other integer-typed parameter in
    # this project (e.g. ResponseCache.max_entries).
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        with pytest.raises(TypeError, match="max_workers"):
            ConcurrentKernelRunner(kernel=kernel, max_workers=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rejects_a_zero_or_negative_max_workers():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        with pytest.raises(ValueError, match="max_workers"):
            ConcurrentKernelRunner(kernel=kernel, max_workers=0)
        with pytest.raises(ValueError, match="max_workers"):
            ConcurrentKernelRunner(kernel=kernel, max_workers=-1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rejects_empty_tasks():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=2)
        try:
            with pytest.raises(ValueError, match="tasks"):
                runner.run_many(())
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rejects_a_non_sequence_tasks():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=2)
        try:
            with pytest.raises(TypeError, match="tasks"):
                runner.run_many("sleep 0 not a real sequence of tasks")
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rejects_a_non_string_task_in_tasks():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=2)
        try:
            with pytest.raises(TypeError, match="string"):
                runner.run_many(["sleep 0 one", 123])
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# Real concurrency and ordering
# ---------------------------------------------------------------------


def test_run_many_runs_tasks_concurrently_not_sequentially():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        delay_seconds = 0.2
        task_count = 5

        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=task_count)
        try:
            tasks = tuple(
                f"sleep {delay_seconds} task-{i}" for i in range(task_count)
            )

            started_at = time.monotonic()
            results = runner.run_many(tasks)
            elapsed = time.monotonic() - started_at

            assert len(results) == task_count
            assert all(result.result.status == "COMPLETED" for result in results)

            # Sequential execution would take at least
            # task_count * delay_seconds (1.0s here). Running all five
            # concurrently, bounded only by max_workers == task_count,
            # should take roughly one delay_seconds, not five. A
            # generous threshold well below the sequential total keeps
            # this from being flaky on a loaded machine while still
            # proving genuine concurrency.
            assert elapsed < (task_count * delay_seconds) * 0.6
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_max_workers_is_a_real_enforced_cap():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        delay_seconds = 0.2
        task_count = 3

        # max_workers=1 forces the three tasks to run one at a time --
        # wall-clock time should be close to the fully sequential
        # total, proving this is a real cap enforced by the underlying
        # ThreadPoolExecutor, not merely accepted and ignored.
        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=1)
        try:
            tasks = tuple(
                f"sleep {delay_seconds} task-{i}" for i in range(task_count)
            )

            started_at = time.monotonic()
            results = runner.run_many(tasks)
            elapsed = time.monotonic() - started_at

            assert len(results) == task_count
            assert elapsed >= (task_count * delay_seconds) * 0.9
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_many_preserves_input_order_regardless_of_completion_order():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)

        # Deliberately decreasing delays -- the FIRST task submitted is
        # the SLOWEST, so if results were collected in completion
        # order instead of input order, task-0 would not be first.
        tasks = (
            "sleep 0.3 task-0",
            "sleep 0.2 task-1",
            "sleep 0.1 task-2",
            "sleep 0.0 task-3",
        )

        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=len(tasks))
        try:
            results = runner.run_many(tasks)

            assert tuple(result.task for result in results) == tasks
            assert all(result.result.status == "COMPLETED" for result in results)
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# Per-task error isolation
# ---------------------------------------------------------------------


def test_one_tasks_exception_does_not_affect_others():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        # Five tasks; the third build_decision_engine() call (in
        # submission order -- deterministic, since each task calls it
        # exactly once) raises. The other four must still complete
        # normally.
        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Raises on exactly one call, completes otherwise.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=_RaisingDecisionEngineFactory(
                    raise_on_call_number=3
                ),
            )
        )

        tasks = tuple(f"task-{i}" for i in range(5))

        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=5)
        try:
            results = runner.run_many(tasks)

            assert len(results) == 5
            assert tuple(result.task for result in results) == tasks

            failing = [result for result in results if result.error is not None]
            succeeding = [result for result in results if result.error is None]

            assert len(failing) == 1
            assert failing[0].result is None
            assert "Deliberate build_decision_engine failure" in failing[0].error

            assert len(succeeding) == 4
            assert all(
                result.result.status == "COMPLETED" for result in succeeding
            )
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_a_deliberate_agent_failure_is_a_normal_result_not_an_error():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="test_agent",
                description="Always fails deliberately.",
                can_handle=lambda normalized: True,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir),
                build_decision_engine=lambda: _AlwaysFailEngine(),
            )
        )

        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=2)
        try:
            (result,) = runner.run_many(("do something",))

            # A deliberate FAILED outcome is a real, typed
            # KernelResult -- not an unexpected exception -- exactly
            # as it would be from a synchronous kernel.run() call.
            assert result.error is None
            assert result.result is not None
            assert result.result.status == "FAILED"
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# run_workflow_many
# ---------------------------------------------------------------------


def test_run_workflow_many_runs_a_registered_workflow_concurrently():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = Kernel(orchestration_engine=SequentialOrchestrationEngine())

        kernel.register_agent(
            AgentRegistration(
                subject="step_one",
                description="First step.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir, subject="step_one"),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        kernel.register_agent(
            AgentRegistration(
                subject="step_two",
                description="Second step.",
                can_handle=lambda normalized: False,
                build_agent=lambda: _build_zero_tool_agent(tmp_dir, subject="step_two"),
                build_decision_engine=lambda: _ImmediateCompleteEngine(),
            )
        )

        kernel.register_workflow(
            WorkflowDefinition(
                name="two_step",
                description="A trivial two-step workflow for concurrency tests.",
                can_handle=lambda normalized: "run-workflow" in normalized.text,
                steps=(
                    WorkflowStep(
                        subject="step_one",
                        build_task=lambda original, previous: original,
                    ),
                    WorkflowStep(
                        subject="step_two",
                        build_task=lambda original, previous: original,
                    ),
                ),
            )
        )

        tasks = tuple(f"run-workflow {i}" for i in range(3))

        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=3)
        try:
            results = runner.run_workflow_many(tasks)

            assert len(results) == 3
            assert tuple(result.task for result in results) == tasks
            assert all(result.error is None for result in results)
            assert all(
                result.result.status == "COMPLETED" for result in results
            )
            assert all(
                len(result.result.completed_steps) == 2 for result in results
            )
        finally:
            runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# Lifecycle: shutdown / context manager
# ---------------------------------------------------------------------


def test_shutdown_is_idempotent():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)
        runner = ConcurrentKernelRunner(kernel=kernel, max_workers=2)
        runner.shutdown()
        # A second call must not raise.
        runner.shutdown()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_context_manager_shuts_down_on_exit():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        kernel = _sleepy_kernel(tmp_dir)

        with ConcurrentKernelRunner(kernel=kernel, max_workers=2) as runner:
            (result,) = runner.run_many(("sleep 0 quick",))
            assert result.result.status == "COMPLETED"

        # The executor is shut down; submitting more work now must
        # raise rather than silently accept work no thread will ever
        # run.
        with pytest.raises(RuntimeError):
            runner.run_many(("sleep 0 too late",))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_concurrent_task_result_is_immutable():
    result = ConcurrentTaskResult(task="x", result=None, error="boom")
    with pytest.raises(Exception):
        result.task = "y"
