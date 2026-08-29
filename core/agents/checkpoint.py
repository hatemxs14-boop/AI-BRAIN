from __future__ import annotations

import json
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# Build Phase 22 -- lightweight, dependency-free checkpoint/resume.
#
# The gap this closes (see the Claude Project doc's Gemini-gap analysis,
# carried since Build Phase 19): a long-running Kernel.run() call that
# is interrupted by something OUTSIDE the running process -- the
# process is killed, the machine restarts, the sandbox is recycled --
# currently has no way to pick up where it left off. Every already-
# completed, already-billed tool call and LLM decision is lost, and
# the only option is to call Kernel.run() again from scratch, re-paying
# for all of it. This is a real, direct Cost Efficiency gap (the same
# standing product directive Build Phases 18-21 have all been serving).
#
# This is a NARROW fix, not a general durability/persistence layer:
#
# - It is entirely opt-in. A caller that never passes a
#   `checkpoint_store` to Kernel.run() gets exactly the same behavior
#   as before this phase existed -- zero risk to any of the 577
#   already-passing tests that don't ask for it.
#
# - It only protects against a PROCESS-level interruption mid-loop. It
#   is not related to, and does not change, Kernel's own existing
#   RECOVER IF NEEDED retry policy (`Kernel._should_recover` /
#   `max_recovery_attempts`), which already handles a DECISION_ERROR/
#   EXECUTION_ERROR that a still-running process can retry on its own
#   -- see `Kernel._execute_once`'s own docstring ("always a full,
#   fresh attempt, never a resume"). A checkpoint is what lets a
#   *different, later* process resume a run that attempt never got the
#   chance to finish because the process itself stopped existing.
#
# - A checkpoint deliberately does NOT persist a byte-for-byte replay
#   of every real ToolExecutionResult (which would require fabricating
#   a matching SecurityDecision/RiskAssessment/AuthorizationResult/
#   ApprovalDecision for each one just to satisfy that dataclass's own
#   required fields -- exactly the kind of "fabricate to look
#   complete" this project has refused to do since Pass 4). Instead it
#   persists the same {status, summary, artifacts} projection of each
#   tool result that LLMDecisionEngine._build_request already puts in
#   front of the LLM (see LLMDecisionEngine._serialize_tool_result) --
#   real, honest, and exactly as much fidelity as resuming actually
#   needs: enough for the decision engine to see what already happened
#   and decide what to do next, nothing more.
#
# - Because of that, a resumed run's `AgentLoopResult.last_result` is
#   `None` until a NEW tool call happens after resume, even if real
#   tool calls happened before the interruption -- this is not a new
#   code path. `_verify()`/`_evaluate_policy()` already treat
#   `last_result is None` as a normal, valid case (a task can
#   legitimately COMPLETE/FAIL without ever calling a tool), so a
#   resumed run whose very first post-resume decision is COMPLETE/FAIL
#   is handled by the exact same, already-tested branch -- not a new,
#   unverified one.
#
# - Checkpoint/resume is wired directly through AgentExecutionLoop
#   (core/agents/agent_loop.py), bypassing the pluggable
#   OrchestrationEngine seam (core/orchestration/orchestration_engine.py)
#   entirely. Extending that abstract interface would also require
#   updating LangGraphOrchestrationEngine, which this project cannot
#   install or execute even once in this sandbox (no package-index
#   access -- see orchestration_engine.py's own module docstring), and
#   shipping an unverified change to it would break this project's own
#   "nothing is done until a real pytest run confirms it" rule. A
#   future phase can revisit surfacing this through the
#   OrchestrationEngine abstraction once LangGraph is genuinely
#   testable here. Until then, checkpoint/resume always runs through
#   the same dependency-free sequential loop SequentialOrchestrationEngine
#   itself calls internally.
# ---------------------------------------------------------------------


_CHECKPOINT_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def _project_tool_result(result: Any) -> dict[str, Any]:
    """
    Reduce one AgentContext.tool_results entry to the same
    {status, summary, artifacts} shape LLMDecisionEngine.
    _serialize_tool_result produces for the LLM's own prompt.

    Accepts either a real ToolExecutionResult (duck-typed via
    `hasattr(result, "status")`, not `isinstance`, so this module has
    no import dependency on core.tools.engine.tool_gateway) or an
    already-projected dict restored from an earlier checkpoint (see
    TaskCheckpoint.tool_results) -- a run that is resumed and then
    interrupted again has a `context.tool_results` made of BOTH kinds
    at once, and both must serialize into the same shape.

    Deliberately an independent, small implementation rather than a
    shared import from LLMDecisionEngine -- keeps this module free of
    any dependency on one concrete AgentDecisionEngine implementation.
    Both are covered by their own tests; a future change to one that
    is meant to apply to both must be made in both places.
    """

    if result is None:
        return {"status": None, "summary": None, "artifacts": []}

    if isinstance(result, dict) and {
        "status",
        "summary",
        "artifacts",
    } <= result.keys():

        artifacts = result["artifacts"]

        if isinstance(artifacts, (list, tuple)):
            safe_artifacts = [str(item) for item in artifacts]
        else:
            safe_artifacts = [str(artifacts)]

        return {
            "status": result["status"],
            "summary": result["summary"],
            "artifacts": safe_artifacts,
        }

    if not hasattr(result, "status"):
        # Unrecognized shape (e.g. a plain string) -- mirror
        # LLMDecisionEngine._serialize_tool_result's own fallback.
        return {"status": None, "summary": str(result), "artifacts": []}

    status = getattr(result, "status", None)
    summary = getattr(result, "summary", None)
    artifacts = getattr(result, "artifacts", ())

    if isinstance(artifacts, (list, tuple)):
        safe_artifacts = [str(item) for item in artifacts]
    else:
        safe_artifacts = [str(artifacts)]

    return {
        "status": status,
        "summary": summary,
        "artifacts": safe_artifacts,
    }


@dataclass(frozen=True)
class TaskCheckpoint:
    """
    A durable, resumable snapshot of one in-progress
    AgentExecutionLoop run, taken right after a step's tool call has
    already succeeded (see AgentExecutionLoop's own checkpoint-saving
    docstring for exactly when).

    `tool_results` is a tuple of plain {status, summary, artifacts}
    dicts, not real ToolExecutionResult objects -- see this module's
    own docstring for why. `step_count` and `last_tool_id` mirror the
    matching AgentContext/AgentState fields at the moment this
    checkpoint was taken.
    """

    checkpoint_id: str
    subject: str
    task: str
    step_count: int
    tool_results: tuple[dict[str, Any], ...]
    last_tool_id: str | None = None

    def __post_init__(self) -> None:

        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty string.")

        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("subject must be a non-empty string.")

        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string.")

        if not isinstance(self.step_count, int) or isinstance(self.step_count, bool):
            raise TypeError("step_count must be an integer.")

        if self.step_count < 0:
            raise ValueError("step_count must not be negative.")

        if not isinstance(self.tool_results, tuple):
            raise TypeError("tool_results must be a tuple.")

        for entry in self.tool_results:
            if not isinstance(entry, dict) or not {
                "status",
                "summary",
                "artifacts",
            } <= entry.keys():
                raise ValueError(
                    "Every tool_results entry must be a dict "
                    "containing 'status', 'summary', and 'artifacts'."
                )

        if self.last_tool_id is not None and not isinstance(
            self.last_tool_id, str
        ):
            raise TypeError("last_tool_id must be a string or None.")

    @staticmethod
    def from_tool_results(
        *,
        checkpoint_id: str,
        subject: str,
        task: str,
        step_count: int,
        tool_results: list[Any],
        last_tool_id: str | None,
    ) -> "TaskCheckpoint":
        """
        Build a TaskCheckpoint from a live AgentContext.tool_results
        list (a mix of real ToolExecutionResult objects and/or already
        -projected dicts), projecting every entry through
        `_project_tool_result` first.
        """

        return TaskCheckpoint(
            checkpoint_id=checkpoint_id,
            subject=subject,
            task=task,
            step_count=step_count,
            tool_results=tuple(
                _project_tool_result(result) for result in tool_results
            ),
            last_tool_id=last_tool_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "subject": self.subject,
            "task": self.task,
            "step_count": self.step_count,
            "tool_results": [dict(entry) for entry in self.tool_results],
            "last_tool_id": self.last_tool_id,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TaskCheckpoint":

        if not isinstance(data, dict):
            raise TypeError("Checkpoint data must be a dict.")

        required = {
            "checkpoint_id",
            "subject",
            "task",
            "step_count",
            "tool_results",
        }

        missing = required - data.keys()

        if missing:
            raise ValueError(
                f"Checkpoint data is missing required fields: {sorted(missing)}"
            )

        tool_results = data["tool_results"]

        if not isinstance(tool_results, list):
            raise TypeError("Checkpoint 'tool_results' must be a list.")

        return TaskCheckpoint(
            checkpoint_id=data["checkpoint_id"],
            subject=data["subject"],
            task=data["task"],
            step_count=data["step_count"],
            tool_results=tuple(dict(entry) for entry in tool_results),
            last_tool_id=data.get("last_tool_id"),
        )


class CheckpointStore(ABC):
    """
    Abstract checkpoint store.

    Mirrors this project's other pluggable-backend interfaces
    (OrchestrationEngine, LLMClient, AgentDecisionEngine): a Kernel
    only depends on this shape, never on FileCheckpointStore
    specifically.
    """

    @abstractmethod
    def save(self, checkpoint: TaskCheckpoint) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint_id: str) -> TaskCheckpoint | None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, checkpoint_id: str) -> None:
        raise NotImplementedError


class FileCheckpointStore(CheckpointStore):
    """
    Lightweight, dependency-free, file-backed CheckpointStore.

    One JSON file per checkpoint_id under `directory`. Unlike
    MemoryStore's or AuditLogger's single shared append-only log
    (Build Phases 13/14), a checkpoint is mutable, single-record, and
    short-lived by design: `save()` overwrites the same file every
    time progress is persisted, and the owning AgentExecutionLoop
    deletes it the moment its own `run()` call returns for ANY reason
    -- a checkpoint exists only to survive an interruption WHILE the
    loop is still actively running, never to represent one of that
    loop's own terminal states (COMPLETED, FAILED, APPROVAL_REQUIRED,
    etc. all have their own, already-established ways of being
    represented -- KernelResult/AgentLoopResult -- a leftover
    checkpoint file would just be a second, stale copy of that).

    Each different checkpoint_id gets its own file, so two different
    tasks checkpointing concurrently (e.g. through Build Phase 21's
    ConcurrentKernelRunner) never contend on the same path. `save()`
    still writes through a temp-file-then-atomic-rename (`Path.
    replace`), so a crash mid-write can never leave a half-written,
    unreadable checkpoint behind for the next resume attempt to trip
    over -- the same category of hazard Build Phase 21 found and fixed
    in this project's own test fixtures, closed here from the start.
    """

    def __init__(self, directory: str | Path) -> None:

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, checkpoint_id: str) -> Path:

        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty string.")

        if not _CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id):
            raise ValueError(
                "checkpoint_id must contain only letters, digits, "
                "'-', and '_' (it is used as a filename)."
            )

        return self.directory / f"{checkpoint_id}.json"

    def save(self, checkpoint: TaskCheckpoint) -> None:

        if not isinstance(checkpoint, TaskCheckpoint):
            raise TypeError("checkpoint must be a TaskCheckpoint.")

        path = self._path_for(checkpoint.checkpoint_id)
        payload = json.dumps(
            checkpoint.to_dict(),
            ensure_ascii=False,
            indent=2,
        )

        with self._lock:
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(path)

    def load(self, checkpoint_id: str) -> TaskCheckpoint | None:

        path = self._path_for(checkpoint_id)

        with self._lock:
            if not path.exists():
                return None
            raw = path.read_text(encoding="utf-8")

        data = json.loads(raw)
        return TaskCheckpoint.from_dict(data)

    def delete(self, checkpoint_id: str) -> None:

        path = self._path_for(checkpoint_id)

        with self._lock:
            path.unlink(missing_ok=True)

    def exists(self, checkpoint_id: str) -> bool:

        return self._path_for(checkpoint_id).exists()
