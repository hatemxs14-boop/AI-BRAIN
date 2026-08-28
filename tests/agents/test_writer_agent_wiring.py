"""
Tests for the real writer_agent wiring (core.agents.writer_agent):
build_writer_agent() and run_writer_agent() assembling the two real
tools (read_research_findings, sandboxed to a findings-root directory;
write_report, gated on explicit approval) behind the full, unmodified
security stack.

No network access is used at all -- both tools are real, sandboxed
filesystem operations against isolated temp directories, structurally
identical in spirit to tests/agents/test_research_agent_wiring.py's
own read_document/write_research_findings coverage.
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
from core.agents.writer_agent import (
    WRITER_AGENT_SUBJECT,
    build_writer_agent,
    run_writer_agent,
)
from core.policies.policy_engine import (
    AgentPermissionAlignment,
    AgentScopeEvaluation,
    PolicyEngine,
)


class _ReadFindingThenCompleteEngine(AgentDecisionEngine):
    """
    Deterministic engine mirroring test_research_agent_wiring.py's
    _ReadDocumentThenCompleteEngine, but exercising
    read_research_findings instead.
    """

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
    """
    Attempts write_report with no approval at all, proving the
    HIGH-risk/"policy" gate pauses the agent instead of writing.
    """

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


class _WriteReportWithApprovalEngine(AgentDecisionEngine):
    """
    Deterministic engine mirroring
    test_research_agent_wiring.py's _WriteFindingsWithApprovalEngine,
    but exercising write_report with an explicit, attributed approval
    supplied up front on the AgentAction itself.
    """

    def __init__(self, filename: str, content: str):
        self._filename = filename
        self._content = content

    def decide(self, context: AgentContext) -> AgentAction:
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="write_report",
                inputs={
                    "filename": self._filename,
                    "content": self._content,
                },
                reason="Publishing an explicitly authorized report.",
                approved=True,
                approved_by="human_operator",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="The report has been published.",
        )


class _RequestUnavailableShellEngine(AgentDecisionEngine):
    """
    Attempts to invoke a "shell" tool_id that writer_agent's real
    wiring never registers -- proving writer_agent's tool surface is
    exactly its two wired tools and nothing else.
    """

    def decide(self, context: AgentContext) -> AgentAction:
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="shell",
                inputs={"command": "echo hi"},
                reason="Attempting to use the shell tool.",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Done.",
        )


class _AlwaysOutOfScopePolicyEngine(PolicyEngine):
    """
    A PolicyEngine stand-in whose evaluate_agent_scope() always reports
    within_scope=False -- mirrors
    tests/agents/test_research_agent_wiring.py's own stand-in of the
    same name, proving build_writer_agent() genuinely delegates to the
    supplied policy_engine rather than always trusting its own
    registrations.
    """

    def evaluate_agent_scope(
        self,
        *,
        subject: str,
        declared_tool_ids,
        actual_tool_ids,
    ) -> AgentScopeEvaluation:
        actual = frozenset(actual_tool_ids)
        return AgentScopeEvaluation(
            subject=subject,
            declared_tool_ids=frozenset(declared_tool_ids),
            actual_tool_ids=actual,
            unauthorized_tool_ids=actual,
            within_scope=False,
        )


class _AlwaysMisalignedPolicyEngine(PolicyEngine):
    """
    A PolicyEngine stand-in whose evaluate_agent_permission_alignment()
    always reports aligned=False -- mirrors
    tests/agents/test_research_agent_wiring.py's own stand-in of the
    same name, proving build_writer_agent() genuinely delegates to the
    supplied policy_engine's config-side alignment check too (Build
    Phase 10), not just its tool-id scope check (Build Phase 9).
    """

    def evaluate_agent_permission_alignment(
        self,
        *,
        subject: str,
        tool_grants_needed,
        security_grants_present,
    ) -> AgentPermissionAlignment:
        needed = frozenset(tool_grants_needed)
        present = frozenset(security_grants_present)
        return AgentPermissionAlignment(
            subject=subject,
            tool_grants_needed=needed,
            security_grants_present=present,
            missing_grants=needed,
            extra_grants=present,
            aligned=False,
        )


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
# build_writer_agent(): tool discovery and configuration errors
# ---------------------------------------------------------------------

def test_build_writer_agent_exposes_exactly_the_two_wired_tools():
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        agent = build_writer_agent(
            findings_root=findings_root,
            reports_root=reports_root,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        assert agent.identity.subject == WRITER_AGENT_SUBJECT

        tool_ids = {tool.id for tool in agent.discover_tools()}

        assert tool_ids == {"read_research_findings", "write_report"}
        assert "web_search" not in tool_ids
        assert "read_document" not in tool_ids
        assert "read_webpage" not in tool_ids
        assert "write_research_findings" not in tool_ids
        assert "shell" not in tool_ids
    finally:
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_build_writer_agent_raises_clear_error_for_missing_findings_root():
    reports_root = _make_reports_root()
    try:
        with pytest.raises(ValueError, match="does not exist"):
            build_writer_agent(
                findings_root="/no/such/directory/anywhere",
                reports_root=reports_root,
            )
    finally:
        shutil.rmtree(reports_root)


def test_build_writer_agent_raises_clear_error_for_missing_reports_root():
    findings_root = _make_findings_root()
    try:
        with pytest.raises(ValueError, match="does not exist"):
            build_writer_agent(
                findings_root=findings_root,
                reports_root="/no/such/directory/anywhere",
            )
    finally:
        shutil.rmtree(findings_root)


def test_build_writer_agent_raises_when_policy_engine_reports_out_of_scope():
    """
    Genuine delegation proof (Build Phase 9): mirrors
    test_research_agent_wiring.py's equivalent test. With an otherwise
    completely normal build, injecting a policy_engine whose
    evaluate_agent_scope() reports within_scope=False must still make
    build_writer_agent() raise ValueError.
    """
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        with pytest.raises(ValueError, match="silently expanded"):
            build_writer_agent(
                findings_root=findings_root,
                reports_root=reports_root,
                policy_engine=_AlwaysOutOfScopePolicyEngine(),
            )
    finally:
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)


def test_build_writer_agent_raises_when_policy_engine_reports_misaligned_permissions():
    """
    Genuine delegation proof (Build Phase 10): mirrors
    test_research_agent_wiring.py's equivalent test. With an otherwise
    completely normal build against the real, aligned permissions.json,
    injecting a policy_engine whose evaluate_agent_permission_alignment()
    reports aligned=False must still make build_writer_agent() raise
    ValueError.
    """
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        with pytest.raises(ValueError, match="drifted"):
            build_writer_agent(
                findings_root=findings_root,
                reports_root=reports_root,
                policy_engine=_AlwaysMisalignedPolicyEngine(),
            )
    finally:
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)


# ---------------------------------------------------------------------
# run_writer_agent(): full end-to-end loops
# ---------------------------------------------------------------------

def test_run_writer_agent_completes_a_read_research_findings_task():
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_writer_agent(
            "Summarize finding.md.",
            decision_engine=_ReadFindingThenCompleteEngine("finding.md"),
            findings_root=findings_root,
            reports_root=reports_root,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            max_steps=3,
        )

        assert result.status == "COMPLETED"
        assert result.last_result is not None
        assert result.last_result.status == "SUCCESS"

        (artifact,) = result.last_result.artifacts
        assert artifact["content"] == (
            "The sky is blue, confirmed by two independent sources."
        )
    finally:
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_run_writer_agent_pauses_for_approval_when_publishing_a_report():
    """
    write_report is HIGH risk / "policy" approval by design (see
    core.agents.writer_agent's module docstring and write_report_tool's
    own docstring): a decision engine that requests it with no approval
    at all must pause the agent exactly like
    tests/agents/test_agent_await_approval.py's generic HIGH-risk
    fixture does -- never write anything.
    """
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_writer_agent(
            "Publish a report.",
            decision_engine=_WriteReportWithoutApprovalEngine(
                "report.md"
            ),
            findings_root=findings_root,
            reports_root=reports_root,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            max_steps=3,
        )

        assert result.status == "APPROVAL_REQUIRED"
        assert not Path(reports_root, "report.md").exists()
    finally:
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_run_writer_agent_completes_a_write_report_task():
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_writer_agent(
            "Publish a report.",
            decision_engine=_WriteReportWithApprovalEngine(
                "report.md",
                "# Report\n\nThe sky is blue, per finding.md.",
            ),
            findings_root=findings_root,
            reports_root=reports_root,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            max_steps=3,
        )

        assert result.status == "COMPLETED"
        assert result.last_result is not None
        assert result.last_result.status == "SUCCESS"

        (artifact,) = result.last_result.artifacts
        assert artifact["path"] == "report.md"

        written = Path(reports_root, "report.md")
        assert written.exists()
        assert "The sky is blue, per finding.md." in (
            written.read_text(encoding="utf-8")
        )
    finally:
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_run_writer_agent_requires_llm_client_or_decision_engine():
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        with pytest.raises(
            ValueError,
            match="llm_client or decision_engine",
        ):
            run_writer_agent(
                "Do something.",
                findings_root=findings_root,
                reports_root=reports_root,
            )
    finally:
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)


def test_shell_permission_grants_nothing_because_no_tool_is_registered():
    """
    writer_agent never registers a ToolDefinition for resource=shell,
    so a "shell" tool_id must be inert: discovery must not expose it,
    and requesting it by tool_id must fail before reaching any
    executor.
    """
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_writer_agent(
            "Try to run a shell command.",
            decision_engine=_RequestUnavailableShellEngine(),
            findings_root=findings_root,
            reports_root=reports_root,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            max_steps=3,
        )

        # AgentToolInterface.create_invocation() raises PermissionError
        # for a tool_id the subject cannot discover; AgentExecutionLoop
        # surfaces that as EXECUTION_ERROR, not a successful shell run.
        assert result.status == "EXECUTION_ERROR"
        assert result.last_result is None
    finally:
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)
