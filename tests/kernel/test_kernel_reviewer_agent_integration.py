"""
End-to-end tests wiring the real Kernel (core.kernel.kernel) to the
real reviewer_agent stack (core.agents.reviewer_agent) through
core.kernel.default_kernel.build_default_kernel(), plus the real
three-agent CLASSIFY behavior Build Phase 11 introduced
(core.kernel.default_kernel's _reviewer_agent_handles keyword
predicate, and the removal of the overlapping bare "report" keyword
from _WRITER_AGENT_KEYWORDS -- see that module's own docstring).

Mirrors tests/kernel/test_kernel_writer_agent_integration.py's pattern
(a local deterministic decision engine per scenario, isolated temp
roots, a self-contained proof of the full stack) rather than importing
from that file.
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


class _ReadFindingThenCompleteEngine(AgentDecisionEngine):
    """writer_agent/reviewer_agent-flavored engine: reads a finding."""

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


class _ReadReportThenCompleteEngine(AgentDecisionEngine):
    """reviewer_agent-flavored engine: reads a published report."""

    def __init__(self, filename: str):
        self._filename = filename

    def decide(self, context: AgentContext) -> AgentAction:
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="read_report",
                inputs={"filename": self._filename},
                reason="No report has been read yet.",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="The requested report has been read.",
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


def _make_reports_root(*, with_sample=True) -> str:
    root = tempfile.mkdtemp()
    if with_sample:
        # write_bytes(), not write_text(): see
        # tests/agents/test_reviewer_agent_wiring.py's own
        # _make_reports_root() for why (a multi-line fixture must not
        # depend on write_text()'s platform-default newline
        # translation to stay platform-independent).
        Path(root, "report.md").write_bytes(
            "# Report\n\nThe sky is blue, per finding.md.".encode(
                "utf-8"
            )
        )
    return root


# ---------------------------------------------------------------------
# reviewer_agent, end to end through the real Kernel
# ---------------------------------------------------------------------

def test_kernel_run_completes_a_read_report_task_through_reviewer_agent():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadReportThenCompleteEngine(
                "report.md"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        # "Verify" is a _REVIEWER_AGENT_KEYWORDS entry and contains no
        # _RESEARCH_AGENT_KEYWORDS/_WRITER_AGENT_KEYWORDS entry, so
        # this must classify to reviewer_agent.
        result = kernel.run("Verify report.md.", max_steps=3)

        assert result.status == "COMPLETED"
        assert result.subject == "reviewer_agent"
        assert result.verification.passed is True
        assert result.loop_result.last_result.status == "SUCCESS"

        (artifact,) = result.loop_result.last_result.artifacts
        assert artifact["content"] == (
            "# Report\n\nThe sky is blue, per finding.md."
        )
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_kernel_run_reviewer_agent_can_read_findings_writer_agent_also_reads():
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

        # "Audit" is a _REVIEWER_AGENT_KEYWORDS entry only.
        result = kernel.run("Audit finding.md.", max_steps=3)

        assert result.status == "COMPLETED"
        assert result.subject == "reviewer_agent"
        assert result.loop_result.last_result.status == "SUCCESS"
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


# ---------------------------------------------------------------------
# CLASSIFY: the "report" vocabulary-overlap fix (Build Phase 11)
# ---------------------------------------------------------------------

def test_kernel_run_routes_a_report_review_task_to_reviewer_agent_not_writer_agent():
    """
    Before Build Phase 11, _WRITER_AGENT_KEYWORDS included a bare
    "report" trigger -- a task like "Review the report." would have
    been a genuine tie between writer_agent and reviewer_agent that
    always resolved to writer_agent (registration order), starving
    reviewer_agent of almost every realistic phrasing of its own job.
    "report" was removed from _WRITER_AGENT_KEYWORDS specifically to
    fix this (see core/kernel/default_kernel.py's own module
    docstring); this test proves the fix, not just the absence of a
    crash.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadReportThenCompleteEngine(
                "report.md"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        result = kernel.run("Review the report.", max_steps=3)

        assert result.subject == "reviewer_agent"
        assert result.status == "COMPLETED"
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_kernel_run_still_routes_a_plain_drafting_task_to_writer_agent():
    """
    Non-regression: removing "report" from _WRITER_AGENT_KEYWORDS must
    not stop writer_agent from being reachable at all -- its remaining
    verbs ("draft"/"write"/"summarize"/"compose") still work.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadFindingThenCompleteEngine(
                "finding.md"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        result = kernel.run("Draft something.", max_steps=3)

        assert result.subject == "writer_agent"
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)


def test_kernel_run_reports_no_agent_available_for_an_unclassifiable_task():
    """
    A task matching none of the three agents' keyword vocabularies
    must still correctly report NO_AGENT_AVAILABLE with three agents
    registered, exactly as it did with two (Build Phase 8).
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadFindingThenCompleteEngine(
                "finding.md"
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
