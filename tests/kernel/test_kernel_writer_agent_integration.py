"""
End-to-end tests wiring the real Kernel (core.kernel.kernel) to the
real writer_agent stack (core.agents.writer_agent) through
core.kernel.default_kernel.build_default_kernel(), plus the real
multi-agent CLASSIFY behavior build_default_kernel() introduced in
Build Phase 8 (core.kernel.default_kernel's
_research_agent_handles/_writer_agent_handles keyword predicates).

Mirrors tests/kernel/test_kernel_research_agent_integration.py's
pattern (a local deterministic decision engine per scenario, isolated
temp roots, a self-contained proof of the full stack) rather than
importing from that file.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.decision_engine import AgentDecisionEngine

from core.kernel.default_kernel import build_default_kernel

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)


class _ReadDocumentThenCompleteEngine(AgentDecisionEngine):
    """research_agent-flavored engine: reads a source document."""

    def __init__(self, path: str):
        self._path = path

    def decide(self, context: AgentContext) -> AgentAction:
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="read_document",
                inputs={"path": self._path},
                reason="No document has been read yet.",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="The requested document has been read.",
        )


class _ReadFindingThenCompleteEngine(AgentDecisionEngine):
    """writer_agent-flavored engine: reads an already-persisted finding."""

    def __init__(self, filename: str):
        self._filename = filename

    def decide(self, context: AgentContext) -> AgentAction:
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="read_research_findings",
                inputs={"filename": self._filename},
                reason="No finding has been read yet.",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="The requested finding has been read.",
        )


class _WriteReportWithoutApprovalEngine(AgentDecisionEngine):
    def __init__(self, filename: str):
        self._filename = filename

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="write_report",
            inputs={
                "filename": self._filename,
                "content": "This should not be written without approval.",
            },
            reason="Attempting to publish a report without approval.",
        )


def _make_documents_root() -> str:
    root = tempfile.mkdtemp()
    Path(root, "sample.txt").write_text(
        "The sky is blue.",
        encoding="utf-8",
    )
    return root


def _make_findings_root(*, with_sample=True) -> str:
    root = tempfile.mkdtemp()
    if with_sample:
        Path(root, "finding.md").write_text(
            "The sky is blue, confirmed by two independent sources.",
            encoding="utf-8",
        )
    return root


def _make_reports_root() -> str:
    return tempfile.mkdtemp()


# ---------------------------------------------------------------------
# writer_agent, end to end through the real Kernel
# ---------------------------------------------------------------------

def test_kernel_run_completes_a_read_research_findings_task_through_writer_agent():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadFindingThenCompleteEngine(
                "finding.md"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        # "Summarize" is a _WRITER_AGENT_KEYWORDS entry and contains no
        # _RESEARCH_AGENT_KEYWORDS substring, so this must classify to
        # writer_agent.
        result = kernel.run("Summarize finding.md.", max_steps=3)

        assert result.status == "COMPLETED"
        assert result.subject == "writer_agent"
        assert result.verification.passed is True
        assert result.loop_result.last_result.status == "SUCCESS"

        (artifact,) = result.loop_result.last_result.artifacts
        assert artifact["content"] == (
            "The sky is blue, confirmed by two independent sources."
        )
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_kernel_run_surfaces_write_report_approval_required_through_writer_agent():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _WriteReportWithoutApprovalEngine(
                "report.md"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        # "Draft" and "report" are both _WRITER_AGENT_KEYWORDS entries
        # and neither is a _RESEARCH_AGENT_KEYWORDS substring, so this
        # must classify to writer_agent.
        result = kernel.run("Draft a report.", max_steps=3)

        assert result.status == "AWAITING_APPROVAL"
        assert result.subject == "writer_agent"
        assert not Path(reports_root, "report.md").exists()
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


# ---------------------------------------------------------------------
# CLASSIFY: real multi-agent routing (Build Phase 8)
# ---------------------------------------------------------------------

def test_kernel_run_reports_no_agent_available_for_an_unclassifiable_task():
    """
    A task matching neither agent's keyword vocabulary must not
    silently fall back to either agent -- KERNEL_SPEC.md's own
    NO_AGENT_AVAILABLE status exists exactly for this case, and Build
    Phase 8 is the first time this project has ever had a real chance
    to exercise it (the prior accept-everything placeholder could
    never produce it).
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadDocumentThenCompleteEngine(
                "sample.txt"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        result = kernel.run("Do the thing.", max_steps=3)

        assert result.status == "NO_AGENT_AVAILABLE"
        assert result.subject is None
        assert result.loop_result is None
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)


def test_kernel_run_prefers_research_agent_on_a_genuine_keyword_tie():
    """
    A task matching both agents' keyword vocabularies is a genuine tie
    -- Kernel._select_agent()'s own documented behavior is "first
    candidate in registration order wins", and
    build_default_kernel() registers research_agent before
    writer_agent (see that module's own docstring), so research_agent
    must be the one actually selected and run.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadDocumentThenCompleteEngine(
                "sample.txt"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        # Contains "research" (research keyword) AND "summarize"
        # (writer keyword) -- a genuine tie.
        result = kernel.run(
            "Research and summarize sample.txt.",
            max_steps=3,
        )

        assert result.subject == "research_agent"
        assert result.status == "COMPLETED"
        assert result.loop_result.last_result.status == "SUCCESS"
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)
