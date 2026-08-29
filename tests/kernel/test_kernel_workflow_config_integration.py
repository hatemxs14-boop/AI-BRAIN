"""
End-to-end tests wiring the real Kernel (core.kernel.kernel) to the
real writer_agent/reviewer_agent stack through core.kernel.
default_kernel.build_default_kernel()'s new (Build Phase 16)
"write_and_review" WorkflowDefinition -- built entirely from a config
dict via core.kernel.workflow_config.build_workflow_from_config(),
rather than the hand-written can_handle/build_task functions Build
Phase 15's own "research_write_review" workflow uses.

Mirrors tests/kernel/test_kernel_workflow_integration.py's pattern (a
local deterministic decision engine, isolated temp roots, a
self-contained proof of the full stack) rather than importing from
that file. The point of this file specifically is to prove that a
workflow built from a plain config dict behaves identically, through
the real Kernel/security stack, to one built from hand-written Python
-- not to re-test build_workflow_from_config()'s own mechanics (see
tests/kernel/test_workflow_config.py for that).
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


class _WriteThenReviewEngine(AgentDecisionEngine):
    """
    Drives whichever of writer_agent/reviewer_agent is currently
    selected, by pattern-matching `context.task` -- the exact phrasing
    core/kernel/default_kernel.py's "write_and_review" config produces
    ("{original_task}" verbatim for writer_agent's own step, "Review
    {previous_artifact_path}." for reviewer_agent's).

    Unlike tests/kernel/test_kernel_workflow_integration.py's own
    _PipelineEngine, writer_agent's step here drafts and publishes
    directly from the workflow's own original instruction -- it never
    reads a research_agent finding first, since this workflow has no
    research_agent step at all (see this module's own docstring).
    """

    def decide(self, context: AgentContext) -> AgentAction:
        task = context.task

        if task.startswith("Review "):
            filename = task[len("Review "):].rstrip(".")

            if not context.tool_results:
                return AgentAction(
                    action_type=AgentActionType.INVOKE_TOOL,
                    tool_id="read_report",
                    inputs={"filename": filename},
                    reason="Read the report before reviewing it.",
                )

            return AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="Reviewed.",
            )

        # Otherwise: this is the workflow's own original task text --
        # writer_agent's own first step.
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="write_report",
                inputs={
                    "filename": "report.md",
                    "content": "# Report\n\nDrafted directly, no prior research step.",
                },
                reason="Publish the report.",
                # write_report is HIGH-risk under the REAL default
                # permissions (core/security/schemas/permissions.json)
                # -- explicit, attributed approval supplied up front,
                # mirroring tests/agents/test_writer_agent_wiring.py's
                # own _WriteReportWithApprovalEngine and
                # tests/kernel/test_kernel_workflow_integration.py's
                # own _PipelineEngine.
                approved=True,
                approved_by="human_operator",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Report published.",
        )


class _AttemptWriteWithoutApprovalEngine(AgentDecisionEngine):
    """Always attempts write_report with no approval -- used against
    the REAL default permissions (HIGH-risk, approval required) to
    prove the config-driven workflow stops at step 1 rather than
    bypassing that gate, exactly like the hand-written Build Phase 15
    workflow already does."""

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="write_report",
            inputs={
                "filename": "report.md",
                "content": "Attempting to publish without approval.",
            },
            reason="Attempt a gated write.",
        )


def _make_documents_root() -> str:
    return tempfile.mkdtemp()


def _make_findings_root() -> str:
    return tempfile.mkdtemp()


def _make_reports_root() -> str:
    return tempfile.mkdtemp()


# ---------------------------------------------------------------------
# Opt-in default, same "no behavior change unless a caller opts in"
# precedent every enable_* flag on build_default_kernel() follows.
# ---------------------------------------------------------------------

def test_build_default_kernel_registers_no_write_and_review_workflow_by_default():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _WriteThenReviewEngine(),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        assert kernel._workflows == []

        result = kernel.run_workflow("Draft a report about the topic, then review it.")
        assert result.status == "NO_WORKFLOW_AVAILABLE"
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)


def test_write_and_review_can_be_enabled_independently_of_research_write_review():
    # Both workflows can be registered on the same Kernel at once --
    # registering one never displaces or collides with the other (see
    # WorkflowDefinition's own docstring: can_handle predicates are
    # each evaluated independently, first match in registration order
    # wins).
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _WriteThenReviewEngine(),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            orchestration_engine=SequentialOrchestrationEngine(),
            enable_research_write_review_workflow=True,
            enable_write_and_review_workflow=True,
        )

        names = {workflow.name for workflow in kernel._workflows}
        assert names == {"research_write_review", "write_and_review"}
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)


# ---------------------------------------------------------------------
# Full pipeline, real stack, config-built WorkflowDefinition
# ---------------------------------------------------------------------

def test_write_and_review_workflow_completes_the_full_pipeline():
    """
    Uses the REAL default permissions.json -- write_report is
    genuinely HIGH-risk there, so _WriteThenReviewEngine supplies
    explicit, attributed approval up front (approved=True,
    approved_by="human_operator"), the same real mechanism every prior
    Build Phase's own approval-gated integration test already
    establishes -- never a weakened test-only policy.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _WriteThenReviewEngine(),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
            enable_write_and_review_workflow=True,
        )

        result = kernel.run_workflow(
            "Draft a report about the topic, then review it."
        )

        assert result.status == "COMPLETED"
        assert result.workflow_name == "write_and_review"
        assert [s.subject for s in result.completed_steps] == [
            "writer_agent",
            "reviewer_agent",
        ]
        assert all(s.loop_result.status == "COMPLETED" for s in result.completed_steps)
        assert all(s.verification.passed for s in result.completed_steps)

        # A real file actually landed where writer_agent's own tool
        # writes -- and reviewer_agent's own step never wrote anything
        # (it holds no write tool of any kind).
        assert Path(reports_root, "report.md").exists()
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_write_and_review_workflow_stops_at_the_first_approval_gate():
    """
    Against the REAL default permissions (writer_agent's write_report
    is HIGH-risk/approval-required), the workflow must stop at step 1
    with AWAITING_APPROVAL -- never silently resolving the gate, and
    never running reviewer_agent on top of an unapproved write. Proves
    Kernel.run_workflow() honors POLICY_SPEC.md's Human Approval
    requirement identically for a config-built workflow as it already
    does for a hand-written one (see tests/kernel/
    test_kernel_workflow_integration.py's own equivalent test).
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _AttemptWriteWithoutApprovalEngine(),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
            enable_write_and_review_workflow=True,
        )

        result = kernel.run_workflow(
            "Draft a report about the topic, then review it."
        )

        assert result.status == "AWAITING_APPROVAL"
        assert len(result.completed_steps) == 1
        assert result.completed_steps[0].subject == "writer_agent"
        assert not Path(reports_root, "report.md").exists()
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)
