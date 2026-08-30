"""
Tests for api/service.py (Build Phase 33, real UI Part 1): the pure
business-logic layer behind this project's first HTTP API. Everything
here runs in ANY environment -- this module has zero dependency on
`fastapi` (see api/service.py's own module docstring).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from api.service import (
    AgentSummary,
    ComponentStatus,
    KernelRunSummary,
    SystemStatus,
    get_system_status,
    list_agents,
    read_recent_audit_events,
    run_kernel_task,
    summarize_kernel_result,
)

from core.kernel.kernel import Kernel, KernelResult
from core.llm.token_usage import TokenUsage


# ---------------------------------------------------------------------
# list_agents()
# ---------------------------------------------------------------------


def test_list_agents_returns_the_three_known_agents():
    agents = list_agents()

    subjects = [agent.subject for agent in agents]
    assert subjects == ["research_agent", "writer_agent", "reviewer_agent"]

    for agent in agents:
        assert isinstance(agent, AgentSummary)
        assert agent.display_name
        assert agent.description


# ---------------------------------------------------------------------
# get_system_status()
# ---------------------------------------------------------------------


def test_get_system_status_returns_every_expected_component():
    status = get_system_status()

    names = {component.name for component in status.components}
    assert names == {
        "llm_provider",
        "web_search",
        "semantic_embeddings",
        "safety_confidence_gate",
        "output_quality_evaluation",
        "observability_tracing",
    }

    for component in status.components:
        assert isinstance(component, ComponentStatus)
        assert isinstance(component.configured, bool)
        assert component.detail


def test_get_system_status_llm_provider_configured_when_anthropic_key_present(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    status = get_system_status()

    llm = next(c for c in status.components if c.name == "llm_provider")
    assert llm.configured is True


def test_get_system_status_llm_provider_not_configured_when_no_key(
    monkeypatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    status = get_system_status()

    llm = next(c for c in status.components if c.name == "llm_provider")
    assert llm.configured is False


def test_get_system_status_web_search_reflects_serper_key(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "fake-key")
    status = get_system_status()
    web_search = next(c for c in status.components if c.name == "web_search")
    assert web_search.configured is True

    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    status = get_system_status()
    web_search = next(c for c in status.components if c.name == "web_search")
    assert web_search.configured is False


def test_system_status_all_configured_is_true_only_when_every_component_is():
    all_true = SystemStatus(
        components=(
            ComponentStatus(name="a", configured=True, detail="d"),
            ComponentStatus(name="b", configured=True, detail="d"),
        )
    )
    assert all_true.all_configured is True

    one_false = SystemStatus(
        components=(
            ComponentStatus(name="a", configured=True, detail="d"),
            ComponentStatus(name="b", configured=False, detail="d"),
        )
    )
    assert one_false.all_configured is False


# ---------------------------------------------------------------------
# summarize_kernel_result() / run_kernel_task()
# ---------------------------------------------------------------------


def test_summarize_kernel_result_rejects_non_kernel_result():
    with pytest.raises(TypeError, match="KernelResult"):
        summarize_kernel_result("not-a-kernel-result")


def test_summarize_kernel_result_maps_every_field_with_usage():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    result = KernelResult(
        status="COMPLETED",
        subject="research_agent",
        loop_result=None,
        verification=None,
        reason="Done.",
        recovery_attempts=1,
        token_usage=usage,
    )

    summary = summarize_kernel_result(result)

    assert isinstance(summary, KernelRunSummary)
    assert summary.status == "COMPLETED"
    assert summary.subject == "research_agent"
    assert summary.reason == "Done."
    assert summary.recovery_attempts == 1
    assert summary.prompt_tokens == 10
    assert summary.completion_tokens == 5
    assert summary.total_tokens == 15


def test_summarize_kernel_result_leaves_token_fields_none_without_usage():
    result = KernelResult(
        status="NO_AGENT_AVAILABLE",
        subject=None,
        loop_result=None,
        verification=None,
        reason=None,
        recovery_attempts=0,
        token_usage=None,
    )

    summary = summarize_kernel_result(result)

    assert summary.prompt_tokens is None
    assert summary.completion_tokens is None
    assert summary.total_tokens is None


def test_kernel_run_summary_to_dict_shape():
    summary = KernelRunSummary(
        status="COMPLETED",
        subject="writer_agent",
        reason="Done.",
        recovery_attempts=0,
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )

    assert summary.to_dict() == {
        "status": "COMPLETED",
        "subject": "writer_agent",
        "reason": "Done.",
        "recovery_attempts": 0,
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


class _StubKernel(Kernel):
    """A Kernel subclass whose run() is stubbed to return a fixed
    KernelResult, without exercising any real agent machinery -- this
    project's own established convention for testing a thin
    orchestration layer above Kernel (see e.g.
    tests/kernel/test_kernel_trace_recorder.py's own _StubLLMClient)."""

    def __init__(self, result: KernelResult):
        # Deliberately never call Kernel.__init__ -- this stub only
        # needs to satisfy isinstance(kernel, Kernel) and override
        # run(); it holds none of the real Kernel's own state.
        self._result = result
        self.received_task: str | None = None

    def run(self, task: str, **kwargs) -> KernelResult:  # type: ignore[override]
        self.received_task = task
        return self._result


def test_run_kernel_task_rejects_non_kernel():
    with pytest.raises(TypeError, match="Kernel"):
        run_kernel_task("not-a-kernel", "do something")


def test_run_kernel_task_rejects_empty_task_text():
    kernel = _StubKernel(
        KernelResult(
            status="COMPLETED",
            subject=None,
            loop_result=None,
            verification=None,
            reason=None,
        )
    )

    with pytest.raises(ValueError, match="task_text"):
        run_kernel_task(kernel, "   ")


def test_run_kernel_task_calls_kernel_run_and_summarizes_result():
    result = KernelResult(
        status="COMPLETED",
        subject="research_agent",
        loop_result=None,
        verification=None,
        reason="Done.",
        recovery_attempts=0,
    )
    kernel = _StubKernel(result)

    summary = run_kernel_task(kernel, "Research AI agents")

    assert kernel.received_task == "Research AI agents"
    assert summary.status == "COMPLETED"
    assert summary.subject == "research_agent"


# ---------------------------------------------------------------------
# read_recent_audit_events()
# ---------------------------------------------------------------------


def test_read_recent_audit_events_returns_empty_list_when_file_missing():
    events = read_recent_audit_events("/tmp/definitely-does-not-exist-xyz.jsonl")
    assert events == []


def test_read_recent_audit_events_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="limit"):
        read_recent_audit_events("/tmp/whatever.jsonl", limit=0)

    with pytest.raises(ValueError, match="limit"):
        read_recent_audit_events("/tmp/whatever.jsonl", limit=-5)


def test_read_recent_audit_events_returns_most_recent_first_and_limited():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "audit.jsonl"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            for i in range(5):
                handle.write(json.dumps({"event": "x", "seq": i}) + "\n")

        events = read_recent_audit_events(path, limit=3)

        assert [e["seq"] for e in events] == [4, 3, 2]


def test_read_recent_audit_events_skips_malformed_lines():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "audit.jsonl"
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps({"event": "a", "seq": 1}) + "\n")
            handle.write("{not valid json\n")
            handle.write("\n")
            handle.write(json.dumps({"event": "b", "seq": 2}) + "\n")

        events = read_recent_audit_events(path, limit=50)

        assert [e["seq"] for e in events] == [2, 1]
