from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Sequence

from core.kernel.kernel import (
    Kernel,
    KernelResult,
    WorkflowRunResult,
)


# ---------------------------------------------------------------------
# Build Phase 21: this project's first real concurrent/parallel request
# handling, closing the gap this project's own baseline doc and
# core/llm/caching_llm_client.py's own ResponseCache docstring have
# both named since Build Phase 20 ("this project has no concurrent/
# parallel request handling yet").
#
# Before this phase, Kernel.run()/Kernel.run_workflow() were each a
# single, fully synchronous, blocking call: a caller with N independent
# tasks had no way to run them at once through the same Kernel except
# calling run() N times in sequence, one after another, each one
# blocking on its own real LLM round-trip(s) before the next could even
# begin. That is the single biggest reason Build Phase 20's own
# profitable-activities assessment rated "serving many simultaneous
# requests" (a live customer-facing chat, a SaaS endpoint handling many
# requests at once) low: nothing in this project could actually do
# that.
#
# What this phase adds is real, and deliberately narrow: a Kernel's own
# per-run mechanics were already safe to call from multiple threads at
# once WITHOUT any change, because of decisions earlier phases already
# made for unrelated reasons --
#
#   - Kernel._registrations/_workflows are only ever mutated by
#     register_agent()/register_workflow(), which every existing
#     caller already calls before ever invoking run()/run_workflow();
#     run() and run_workflow() themselves only ever READ these two
#     lists, never write them.
#   - AgentRegistration.build_agent/build_decision_engine (and
#     WorkflowVerifierRegistration's own pair) are FACTORIES, not
#     shared instances -- Kernel._plan() and
#     _trigger_independent_verification build a fresh AgentCore and a
#     fresh AgentDecisionEngine for every single call, precisely so
#     "reusing one instance across unrelated Kernel.run() calls" (a
#     footgun even for purely sequential callers, per
#     AgentRegistration's own docstring) was never possible in the
#     first place. Two concurrent Kernel.run() calls therefore never
#     share an AgentCore, an AgentState, or a decision engine.
#   - core.policies.policy_engine.PolicyEngine (confirmed by reading
#     the whole module: no `__init__`, no `self.` attribute
#     assignment anywhere) is a stateless class -- every method is a
#     pure function of its own arguments. Nothing to race.
#   - core.orchestration.orchestration_engine.SequentialOrchestrationEngine
#     holds no state of its own either; `run()` only ever builds and
#     drives a fresh AgentExecutionLoop from the arguments it's given.
#
# The two places that WERE genuinely unsafe under concurrent use --
# because nothing before this phase could ever call them from more
# than one thread, so nothing had ever needed to make them safe -- have
# been hardened alongside this module, not left as a silent trap this
# new capability would have exposed:
#
#   - core.llm.caching_llm_client.ResponseCache (Build Phase 20): the
#     ONE component `build_default_kernel(enable_response_cache=True)`
#     deliberately shares across every fresh CachingLLMClient it
#     builds (see that function's own docstring) -- now guarded by its
#     own `threading.Lock` (see ResponseCache's own docstring).
#   - core.memory.memory_store.MemoryStore's `_append` and
#     core.security.engine.audit_logger.AuditLogger's `record` -- both
#     append-only JSON Lines files that more than one concurrently-
#     running agent's own tool calls can now write to at once -- now
#     serialized via a module-level `threading.Lock` each (see their
#     own docstrings for why module-level, not per-instance).
#
# ConcurrentKernelRunner below is the actual new capability: a thin,
# dependency-free wrapper (stdlib `concurrent.futures.ThreadPoolExecutor`
# only -- no new external dependency, per this project's standing
# constraint) around one already-built Kernel, letting a caller submit
# a whole batch of independent task strings and have them genuinely run
# concurrently, each through the Kernel's own real, unchanged
# run()/run_workflow() lifecycle. Threads, not asyncio or
# multiprocessing: every real unit of work this Kernel does today is
# either a blocking network call (an LLM API request) or a fast,
# in-process computation -- exactly the profile threads suit, since
# Python releases the GIL around blocking I/O, and introducing asyncio
# would mean an async-colored rewrite of AgentExecutionLoop, every
# decision engine, and every tool implementation this project has
# built since Pass 1, for no real benefit this profile needs.
# Multiprocessing would mean pickling Kernel/AgentCore/LLMClient across
# a process boundary for no reason either, since there is no CPU-bound
# work here to escape the GIL for.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ConcurrentTaskResult:
    """
    One task's outcome from a batch submitted to ConcurrentKernelRunner.

    `result` is the real KernelResult (from run_many) or
    WorkflowRunResult (from run_workflow_many) that `task` produced --
    exactly, field for field, what a synchronous `kernel.run(task)` /
    `kernel.run_workflow(task)` call would have returned. Concurrency
    changes nothing about what running one task means or returns; it
    only lets more than one be in flight on the same Kernel at once.
    `None` only when `error` is set instead.

    `error` is the exception's own message when running `task` raised
    instead of returning normally. Kernel.run()/run_workflow() are
    designed to degrade into a typed terminal KernelResult/
    WorkflowRunResult status for almost every failure mode
    (DECISION_ERROR, EXECUTION_ERROR, TOOL_ERROR, and so on -- see
    Kernel's own module docstring) rather than raise, so reaching this
    field at all should be rare in practice: most realistically, a
    caller-supplied AgentRegistration/WorkflowVerifierRegistration
    factory that itself raises (e.g. `build_agent()` returning
    something that is not a real AgentCore -- Kernel._plan()'s own
    TypeError, per AgentRegistration's own docstring, is a genuine
    build-time misconfiguration bug, deliberately not swallowed there
    either). One task's exception must never lose, corrupt, or delay
    any other task's own result -- see ConcurrentKernelRunner._run_batch
    below for exactly how that isolation is enforced (each task's own
    call is wrapped individually; nothing here uses a shared
    fail-fast primitive that would let one exception cancel the rest).
    """

    task: str
    result: KernelResult | WorkflowRunResult | None
    error: str | None


class ConcurrentKernelRunner:
    """
    Runs a batch of independent task strings through one already-built
    Kernel concurrently (Build Phase 21) -- see this module's own
    top-of-file docstring for why this is safe and why threads.

    Wraps exactly one `Kernel` (never builds or owns one itself -- a
    caller builds it however it already does, e.g.
    `core.kernel.default_kernel.build_default_kernel()`, with every
    agent/workflow already registered, and hands it to this class).
    Holds one `concurrent.futures.ThreadPoolExecutor`, created once at
    construction time and reused across every `run_many`/
    `run_workflow_many` call this instance makes -- deliberately a
    long-lived pool, not one created and torn down per call, so a
    caller that serves many separate batches over its own lifetime
    (the realistic "commercial value" shape this phase targets: a long-
    running process handling many requests over time, not a single
    one-off script) pays the thread-creation cost once, not per batch.
    Call `shutdown()` (or use this class as a context manager) when
    done with it, to release its worker threads.

    `max_workers` bounds how many of a batch's tasks can genuinely run
    at once -- a real, enforced ceiling (ThreadPoolExecutor's own
    semantics: at most `max_workers` submitted callables run
    concurrently; the rest queue until a worker frees up), not just a
    suggestion. Must be a positive integer.
    """

    def __init__(self, kernel: Kernel, *, max_workers: int = 4) -> None:

        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be a Kernel.")

        if not isinstance(max_workers, int) or isinstance(max_workers, bool):
            raise TypeError("max_workers must be an integer.")

        if max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")

        self.kernel = kernel
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ai-brain-kernel",
        )
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

    def run_many(
        self,
        tasks: Sequence[str],
        *,
        max_steps: int = 10,
    ) -> tuple[ConcurrentTaskResult, ...]:
        """
        Runs every task in `tasks` through `self.kernel.run()`
        concurrently (bounded by `max_workers`), and returns one
        ConcurrentTaskResult per task, IN THE SAME ORDER `tasks` was
        given in -- regardless of which task's own real LLM call(s)
        happened to finish first. See _run_batch's own docstring for
        exactly how ordering and per-task error isolation are both
        guaranteed.
        """

        return self._run_batch(
            tasks,
            call=lambda task: self.kernel.run(task, max_steps=max_steps),
        )

    def run_workflow_many(
        self,
        tasks: Sequence[str],
        *,
        max_steps: int = 10,
    ) -> tuple[ConcurrentTaskResult, ...]:
        """
        The `run_workflow()` counterpart to `run_many` above -- runs
        every task in `tasks` through `self.kernel.run_workflow()`
        concurrently, same ordering and error-isolation guarantees.
        """

        return self._run_batch(
            tasks,
            call=lambda task: self.kernel.run_workflow(
                task, max_steps=max_steps
            ),
        )

    def _run_batch(
        self,
        tasks: Sequence[str],
        *,
        call: Callable[[str], KernelResult | WorkflowRunResult],
    ) -> tuple[ConcurrentTaskResult, ...]:
        """
        Shared batch mechanics for run_many/run_workflow_many.

        Ordering: `futures` is built as a plain list, in `tasks`' own
        order, one `self._executor.submit(...)` per task; the return
        value then calls `.result()` on each future in that SAME list
        order. `Future.result()` blocks until THAT future is done, but
        blocking on an earlier future that happens to finish later than
        a subsequent one costs nothing extra -- every submitted task is
        already running (or queued) concurrently the moment `submit()`
        returns, so this only reorders when results are COLLECTED, not
        when work happens. This is deliberately simpler than
        `executor.map` (which offers the same ordering guarantee) so
        that per-task exception isolation, below, can be handled
        explicitly rather than relying on `map`'s own re-raise-on-
        iteration behavior.

        Error isolation: `_run_one` catches any exception `call(task)`
        raises and reports it as `ConcurrentTaskResult.error` instead
        of letting it propagate -- so one task's own unexpected
        exception can never cancel, corrupt, or block collection of any
        other task's already-completed (or still in-flight) result.
        """

        if not isinstance(tasks, (list, tuple)):
            raise TypeError("tasks must be a list or tuple of strings.")

        if not tasks:
            raise ValueError("tasks must not be empty.")

        if not all(isinstance(task, str) for task in tasks):
            raise TypeError("every task in tasks must be a string.")

        def _run_one(task: str) -> ConcurrentTaskResult:
            try:
                return ConcurrentTaskResult(
                    task=task,
                    result=call(task),
                    error=None,
                )
            except Exception as exc:  # noqa: BLE001 -- deliberately
                # broad: this boundary's entire job is to make sure NO
                # exception from one task's own call ever escapes this
                # worker thread and disrupts any other task's result
                # (see this method's own docstring, "Error isolation").
                return ConcurrentTaskResult(
                    task=task,
                    result=None,
                    error=str(exc),
                )

        futures = [self._executor.submit(_run_one, task) for task in tasks]

        return tuple(future.result() for future in futures)

    def shutdown(self, *, wait: bool = True) -> None:
        """
        Releases this runner's worker threads. Safe to call more than
        once -- a second call is a no-op, not an error, mirroring this
        project's own "must never become so strict it refuses to
        execute/accept something" standing constraint applied to
        ordinary cleanup rather than to a security decision.
        """

        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True

        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "ConcurrentKernelRunner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
