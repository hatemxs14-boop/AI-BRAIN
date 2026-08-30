"""
api/service.py

Build Phase 33 (real UI, Part 1): the pure business-logic layer behind
this project's first real HTTP API. Deliberately contains ZERO
references to `fastapi` (or any web framework) anywhere in this file
-- unlike `api/app.py` (the actual HTTP wiring, a thin layer on top of
this module), everything here is plain Python, so it can be fully
unit-tested in ANY environment, including this sandbox, where
`fastapi` itself is not installed (no PyPI access here -- confirmed
the same situation as voyageai/deepeval/langfuse before it).

Nothing in this module talks to a real network, a real LLM, or a real
Ollama/Langfuse server. It orchestrates the project's own already-real,
already-tested pieces (Kernel, TokenUsage, the audit log's own JSONL
format) into UI-friendly shapes -- exactly the same "thin translation
layer, no new business logic" role api/app.py's own docstring
describes for the HTTP layer one level up.
"""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.kernel.kernel import Kernel, KernelResult


# ---------------------------------------------------------------------
# Agent registry -- a UI-friendly summary of the agents
# build_default_kernel() actually registers.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSummary:
    """
    A UI-friendly, human-readable summary of one registered agent.
    """

    subject: str
    display_name: str
    description: str


# Intentionally a static, hand-maintained tuple, NOT dynamically
# introspected from a live Kernel/build_default_kernel() call: Kernel
# exposes no public "list every registered agent" API (its internal
# `AgentRegistration` bookkeeping is deliberately private -- see
# core/kernel/kernel.py's own docstrings), so reflecting these three
# agents here means updating this tuple by hand whenever
# build_default_kernel() registers a new one. This is the exact same
# hand-maintained-vocabulary trade-off this project already accepts
# for Kernel._classify()'s own keyword predicates (Build Phase 8) --
# a real, testable, honestly-static list, not a fabricated "auto-
# discovered" one this project's actual architecture doesn't support.
DEFAULT_AGENT_SUMMARIES: tuple[AgentSummary, ...] = (
    AgentSummary(
        subject="research_agent",
        display_name="Research Agent",
        description=(
            "Searches the web (Serper) and reads/writes project "
            "documents to answer research tasks."
        ),
    ),
    AgentSummary(
        subject="writer_agent",
        display_name="Writer Agent",
        description="Drafts reports from research_agent's own findings.",
    ),
    AgentSummary(
        subject="reviewer_agent",
        display_name="Reviewer Agent",
        description=(
            "Independently verifies a writer_agent report against the "
            "underlying findings it was drafted from."
        ),
    ),
)


def list_agents() -> tuple[AgentSummary, ...]:
    """
    Return every agent this project's own build_default_kernel()
    registers. See DEFAULT_AGENT_SUMMARIES's own docstring for why
    this is a static list, not a live introspection.
    """
    return DEFAULT_AGENT_SUMMARIES


# ---------------------------------------------------------------------
# System status -- a live, runtime-checkable version of this
# project's own PRODUCTION ACTIVATION CHECKLIST (the Claude Project
# doc, claude/ai-brain-repo-baseline.md).
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentStatus:
    """
    Whether one optional/required external activation (an API key, a
    pip package) is present, right now, in this real process's own
    environment -- never a cached or assumed value.
    """

    name: str
    configured: bool
    detail: str


@dataclass(frozen=True)
class SystemStatus:
    components: tuple[ComponentStatus, ...]

    @property
    def all_configured(self) -> bool:
        return all(component.configured for component in self.components)


def _env_present(var_name: str) -> bool:
    return bool(os.environ.get(var_name))


def _package_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_system_status() -> SystemStatus:
    """
    A live, runtime version of this project's own PRODUCTION
    ACTIVATION CHECKLIST -- whenever that checklist (the Claude
    Project doc) gains a new entry, this function should grow with
    it, so the dashboard never silently drifts out of sync with the
    documented checklist.

    Every check here is cheap, local, and side-effect-free: an
    environment variable being non-empty, or a package being
    importable via `importlib.util.find_spec` (never actually
    imported -- importing a heavy vendor SDK just to render a status
    dot would be wasteful and could itself have side effects). This
    deliberately NEVER makes a real network call (it does not ping a
    real Ollama server or a real Langfuse instance) -- "is Ollama
    actually reachable right now" is a materially different, heavier
    check (see this module's own module docstring) that a future
    Build Phase can add as its own explicit, separately-named
    endpoint/function, not silently folded into this always-fast,
    always-safe status check.
    """
    components = (
        ComponentStatus(
            name="llm_provider",
            configured=_env_present("ANTHROPIC_API_KEY")
            or _env_present("OPENAI_API_KEY"),
            detail="ANTHROPIC_API_KEY or OPENAI_API_KEY",
        ),
        ComponentStatus(
            name="web_search",
            configured=_env_present("SERPER_API_KEY"),
            detail="SERPER_API_KEY (research_agent's web_search tool)",
        ),
        ComponentStatus(
            name="semantic_embeddings",
            configured=(
                _env_present("VOYAGE_API_KEY")
                and _package_installed("voyageai")
            ),
            detail="VOYAGE_API_KEY + the voyageai package",
        ),
        ComponentStatus(
            name="safety_confidence_gate",
            configured=_env_present("OLLAMA_BASE_URL")
            or _env_present("OLLAMA_HOST"),
            detail=(
                "OLLAMA_BASE_URL/OLLAMA_HOST (Llama Guard confidence "
                "gate) -- presence of this env var is checked, NOT "
                "whether an Ollama server is actually reachable there"
            ),
        ),
        ComponentStatus(
            name="output_quality_evaluation",
            configured=_package_installed("deepeval"),
            detail="the deepeval package (OutputQualityEvaluator)",
        ),
        ComponentStatus(
            name="observability_tracing",
            configured=(
                _package_installed("langfuse")
                and _env_present("LANGFUSE_PUBLIC_KEY")
                and _env_present("LANGFUSE_SECRET_KEY")
            ),
            detail=(
                "the langfuse package + LANGFUSE_PUBLIC_KEY/"
                "LANGFUSE_SECRET_KEY"
            ),
        ),
    )
    return SystemStatus(components=components)


# ---------------------------------------------------------------------
# Running a real Kernel task and summarizing its real KernelResult.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class KernelRunSummary:
    """
    A UI-friendly, JSON-serializable summary of one real KernelResult.

    Deliberately NOT `dataclasses.asdict(result)` on the raw
    KernelResult: that would leak KernelResult's own internal,
    project-specific dataclasses (AgentLoopResult, KernelVerification,
    ExternalActionEvaluation, RetrievedContext) into an HTTP response
    wholesale, and silently change shape the next time KernelResult
    itself changes. `to_dict()` below is this dataclass's own
    explicit, hand-written, stable JSON shape.
    """

    status: str
    subject: str | None
    reason: str | None
    recovery_attempts: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "subject": self.subject,
            "reason": self.reason,
            "recovery_attempts": self.recovery_attempts,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def summarize_kernel_result(result: KernelResult) -> KernelRunSummary:
    if not isinstance(result, KernelResult):
        raise TypeError("result must be a KernelResult.")

    usage = result.token_usage

    return KernelRunSummary(
        status=result.status,
        subject=result.subject,
        reason=result.reason,
        recovery_attempts=result.recovery_attempts,
        prompt_tokens=usage.prompt_tokens if usage is not None else None,
        completion_tokens=(
            usage.completion_tokens if usage is not None else None
        ),
        total_tokens=usage.total_tokens if usage is not None else None,
    )


def run_kernel_task(kernel: Kernel, task_text: str) -> KernelRunSummary:
    """
    Run one real `kernel.run(task_text)` call and return a UI-friendly
    summary of its real KernelResult.

    Validates ITS OWN input strictly (a non-empty string), but
    deliberately does NOT catch any exception `kernel.run()` itself
    raises -- a real internal Kernel error here is a genuine bug this
    service layer must never mask behind a fabricated success, on the
    same "fail loudly, never fabricate a result" principle
    core.evaluation.output_quality.OutputQualityEvaluationError's own
    docstring already establishes elsewhere in this project. Turning
    an unexpected exception into a real HTTP 500 is `api/app.py`'s own
    job, one layer up -- never this layer's.
    """
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a Kernel.")

    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("task_text must be a non-empty string.")

    result = kernel.run(task_text)
    return summarize_kernel_result(result)


# ---------------------------------------------------------------------
# Reading this project's own real audit log (Build Phase 13's JSONL
# format) for display.
# ---------------------------------------------------------------------


def read_recent_audit_events(
    audit_log_path: str | Path,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    The most recent `limit` real events from this project's own real,
    append-only audit log -- most-recent-first.

    Returns an empty list (never raises) when the file does not exist
    yet: a fresh install genuinely has no audit history, which is not
    an error condition a dashboard should surface as one. A malformed
    or partially-written line (e.g. the process was killed mid-write)
    is silently skipped rather than raising -- one corrupt line must
    never break the entire dashboard's view of everything else that
    logged successfully. Audit logging (Build Phase 13) is documented
    elsewhere in this project as producing two events per real tool
    call -- this function does not deduplicate or otherwise
    reinterpret that; it returns exactly what is on disk.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer.")

    path = Path(audit_log_path)

    if not path.exists():
        return []

    events: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    most_recent = events[-limit:]
    most_recent.reverse()
    return most_recent
