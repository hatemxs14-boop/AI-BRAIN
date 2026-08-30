"""
Tests for Kernel(trace_recorder=...) -- Build Phase 32's Kernel-level
observability tracing wiring (core/observability/langfuse_trace.py,
core/kernel/kernel.py).

Uses the same lightweight `_StubLLMClient` + `build_default_kernel()`
pattern tests/kernel/test_default_kernel_optional_components_
integration.py already established for Build Phase 30, rather than the
heavier ToolGateway/permissions.json fixture style test_kernel_budget.py
and test_kernel_model_tier.py use -- this Build Phase only needs a
Kernel that reaches a real terminal status through a real run(), not a
real tool call.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from core.kernel.default_kernel import build_default_kernel
from core.kernel.kernel import Kernel

from core.llm.llm_client import LLMClient
from core.llm.llm_request import LLMRequest
from core.llm.llm_response import LLMResponse
from core.llm.token_usage import TokenUsage

from core.observability.langfuse_trace import TraceRecorder


_COMPLETE_RESPONSE_TEXT = (
    '{"action_type":"COMPLETE","tool_id":null,"inputs":null,'
    '"reason":"Done."}'
)


class _StubLLMClient(LLMClient):
    def __init__(self, *, usage: TokenUsage | None = None):
        self._usage = usage

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            content=_COMPLETE_RESPONSE_TEXT,
            model=request.model or "stub",
            usage=self._usage,
        )


class _FakeTraceRecorder(TraceRecorder):
    def __init__(self, *, raise_on_record: bool = False):
        self.calls: list[dict] = []
        self.flush_calls = 0
        self._raise_on_record = raise_on_record

    def record_run(
        self,
        *,
        name,
        input_text,
        output_text,
        status,
        metadata=None,
        usage=None,
    ):
        if self._raise_on_record:
            raise RuntimeError("simulated trace-recording failure")

        self.calls.append(
            {
                "name": name,
                "input_text": input_text,
                "output_text": output_text,
                "status": status,
                "metadata": metadata,
                "usage": usage,
            }
        )

    def flush(self) -> None:
        self.flush_calls += 1


def _make_roots():
    docs_root = tempfile.mkdtemp()
    findings_root = tempfile.mkdtemp()
    audit_dir = tempfile.mkdtemp()
    return docs_root, findings_root, audit_dir


def _cleanup(*dirs):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------
# Kernel-level validation
# ---------------------------------------------------------------------


def test_kernel_rejects_non_trace_recorder():
    with pytest.raises(TypeError, match="trace_recorder"):
        Kernel(trace_recorder="not-a-trace-recorder")


def test_kernel_defaults_trace_recorder_to_none():
    kernel = Kernel()
    assert kernel.trace_recorder is None


# ---------------------------------------------------------------------
# build_default_kernel() wiring
# ---------------------------------------------------------------------


def test_build_default_kernel_trace_recorder_reaches_the_underlying_kernel():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        recorder = _FakeTraceRecorder()

        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            trace_recorder=recorder,
        )

        assert kernel.trace_recorder is recorder
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_build_default_kernel_trace_recorder_defaults_to_none():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        assert kernel.trace_recorder is None
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


# ---------------------------------------------------------------------
# Genuine end-to-end behavioral proof
# ---------------------------------------------------------------------


def test_kernel_without_trace_recorder_is_unaffected():
    docs_root, findings_root, audit_dir = _make_roots()
    try:
        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
        )

        result = kernel.run("Research AI agents")

        assert result.status == "COMPLETED"
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_kernel_records_exactly_one_trace_per_run_call():
    """
    The strongest possible proof this wiring is real: a
    build_default_kernel()-built Kernel, with a real (fake-but-
    injectable) TraceRecorder passed straight through, must actually
    call record_run() exactly once on a real run() call -- with the
    real task text, the real final status, and the real metadata --
    not just carry the object as an inert attribute.
    """

    docs_root, findings_root, audit_dir = _make_roots()
    try:
        usage = TokenUsage(
            prompt_tokens=12, completion_tokens=8, total_tokens=20
        )
        recorder = _FakeTraceRecorder()

        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(usage=usage),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            trace_recorder=recorder,
        )

        result = kernel.run("Research AI agents")

        assert result.status == "COMPLETED"
        assert len(recorder.calls) == 1

        call = recorder.calls[0]
        assert call["name"] == "kernel_run"
        assert call["input_text"] == "Research AI agents"
        assert call["status"] == "COMPLETED"
        assert call["usage"] == usage
        assert call["metadata"]["subject"] == result.subject
        assert call["metadata"]["recovery_attempts"] == 0

        # This Build Phase deliberately never auto-flushes -- see
        # TraceRecorder.flush()'s own docstring for why.
        assert recorder.flush_calls == 0
    finally:
        _cleanup(docs_root, findings_root, audit_dir)


def test_kernel_run_survives_a_trace_recorder_that_raises():
    """
    Recording a trace must never be allowed to prevent a real
    Kernel.run() result from reaching its caller -- see Kernel.run()'s
    own inline comment for this deliberate, broad exception swallow.
    """

    docs_root, findings_root, audit_dir = _make_roots()
    try:
        recorder = _FakeTraceRecorder(raise_on_record=True)

        kernel = build_default_kernel(
            llm_client_factory=lambda: _StubLLMClient(),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="test-key",
            audit_log_path=str(Path(audit_dir) / "audit.jsonl"),
            trace_recorder=recorder,
        )

        result = kernel.run("Research AI agents")

        assert result.status == "COMPLETED"
    finally:
        _cleanup(docs_root, findings_root, audit_dir)
