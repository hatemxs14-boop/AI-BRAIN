"""
End-to-end tests wiring the real Kernel (core.kernel.kernel) to the
real research_agent/writer_agent/reviewer_agent stack through
core.kernel.default_kernel.build_default_kernel()'s new (Build Phase
15) "research_write_review" WorkflowDefinition -- research_agent ->
writer_agent -> reviewer_agent, chained end to end from a single
instruction via the new Kernel.run_workflow().

Mirrors tests/kernel/test_kernel_reviewer_agent_integration.py's
pattern (a local deterministic decision engine per scenario, isolated
temp roots, a self-contained proof of the full stack) rather than
importing from that file. Since build_default_kernel() shares ONE
decision-engine factory across every registered agent (see that
module's own docstring), the engine below branches on each step's own
task text -- exactly the text
core.kernel.default_kernel._research_write_review_step_*_task actually
produce -- rather than on which agent is currently running.
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

from core.kernel.default_kernel import (
    _research_write_review_handles,
    build_default_kernel,
)

from core.kernel.kernel import NormalizedTask

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)


class _PipelineEngine(AgentDecisionEngine):
    """
    Drives whichever of research_agent/writer_agent/reviewer_agent is
    currently selected, entirely by pattern-matching `context.task` --
    the exact phrasing core/kernel/default_kernel.py's
    _research_write_review_step_2_task/_step_3_task build
    ("Write a report summarizing the findings in {path}."/
    "Review {path}."), plus the workflow's own original task text for
    research_agent's own first step.
    """

    def decide(self, context: AgentContext) -> AgentAction:
        task = context.task

        if task.startswith("Write a report summarizing the findings in "):
            filename = task[
                len("Write a report summarizing the findings in "):
            ].rstrip(".")

            if not context.tool_results:
                return AgentAction(
                    action_type=AgentActionType.INVOKE_TOOL,
                    tool_id="read_research_findings",
                    inputs={"filename": filename},
                    reason="Read the findings before writing the report.",
                )

            if len(context.tool_results) == 1:
                return AgentAction(
                    action_type=AgentActionType.INVOKE_TOOL,
                    tool_id="write_report",
                    inputs={
                        "filename": "report.md",
                        "content": "# Report\n\nBased on the findings.",
                    },
                    reason="Publish the report.",
                    # write_report is HIGH-risk under the REAL default
                    # permissions (core/security/schemas/permissions.
                    # json) -- explicit, attributed approval supplied
                    # up front, mirroring tests/agents/test_writer_
                    # agent_wiring.py's own _WriteReportWithApprovalEngine,
                    # is how this test reaches a genuine COMPLETED
                    # pipeline without weakening the real security
                    # policy itself.
                    approved=True,
                    approved_by="human_operator",
                )

            return AgentAction(
                action_type=AgentActionType.COMPLETE,
                reason="Report published.",
            )

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
        # research_agent's own first step.
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="write_research_findings",
                inputs={
                    "filename": "finding.md",
                    "content": (
                        "The sky is blue, confirmed by two independent "
                        "sources."
                    ),
                },
                reason="Persist a finding before handing off to writer_agent.",
                # write_research_findings is likewise HIGH-risk under
                # the real default permissions -- see write_report's
                # own comment above for why this is approved up front
                # rather than via a weakened permissions.json.
                approved=True,
                approved_by="human_operator",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Finding persisted.",
        )


class _AttemptFindingWriteWithoutApprovalEngine(AgentDecisionEngine):
    """Always attempts write_research_findings -- used against the
    REAL default permissions (HIGH-risk, approval required) to prove
    the workflow stops at step 1 rather than bypassing that gate."""

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="write_research_findings",
            inputs={
                "filename": "finding.md",
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
# Opt-in default (Build Phase 15, mirroring Build Phase 12/14's own
# "no existing caller's behavior changes unless they opt in" precedent)
# ---------------------------------------------------------------------

def test_build_default_kernel_registers_no_workflow_by_default():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _PipelineEngine(),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        assert kernel._workflows == []

        result = kernel.run_workflow(
            "Research the topic and write a report about it."
        )
        assert result.status == "NO_WORKFLOW_AVAILABLE"
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)


# ---------------------------------------------------------------------
# can_handle: conjunctive (research AND writer signal), not either
# vocabulary alone -- see core/kernel/default_kernel.py's own docstring
# for why.
# ---------------------------------------------------------------------

def test_research_write_review_handles_requires_both_signals():
    assert _research_write_review_handles(
        NormalizedTask(text="Research the topic and write a report about it.")
    ) is True

    # Research signal only.
    assert _research_write_review_handles(
        NormalizedTask(text="Research the topic.")
    ) is False

    # Writer signal only.
    assert _research_write_review_handles(
        NormalizedTask(text="Write a report.")
    ) is False

    # Reviewer signal alone is not enough either.
    assert _research_write_review_handles(
        NormalizedTask(text="Review the report.")
    ) is False


# ---------------------------------------------------------------------
# Full pipeline, real stack, real Kernel.run_workflow()
# ---------------------------------------------------------------------

def test_research_write_review_workflow_completes_the_full_pipeline():
    """
    Uses the REAL default permissions.json (core/security/schemas/
    permissions.json) -- both write_research_findings and write_report
    are genuinely HIGH-risk there, so _PipelineEngine supplies explicit,
    attributed approval up front on each write (approved=True,
    approved_by="human_operator"), the same real mechanism tests/agents/
    test_writer_agent_wiring.py's own _WriteReportWithApprovalEngine
    already establishes -- never a weakened test-only policy. This is
    what a real caller who already has a human's go-ahead for a given
    write would do; test_research_write_review_workflow_stops_at_the_
    first_approval_gate below covers the unapproved case.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _PipelineEngine(),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
            enable_research_write_review_workflow=True,
        )

        result = kernel.run_workflow(
            "Research the topic and write a report about it."
        )

        assert result.status == "COMPLETED"
        assert result.workflow_name == "research_write_review"
        assert [s.subject for s in result.completed_steps] == [
            "research_agent",
            "writer_agent",
            "reviewer_agent",
        ]
        assert all(s.loop_result.status == "COMPLETED" for s in result.completed_steps)
        assert all(s.verification.passed for s in result.completed_steps)

        # Real files actually landed where each agent's own tools write.
        assert Path(findings_root, "finding.md").exists()
        assert Path(reports_root, "report.md").exists()
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)


def test_research_write_review_workflow_stops_at_the_first_approval_gate():
    """
    Against the REAL default permissions (research_agent's
    write_research_findings is HIGH-risk/approval-required), the
    workflow must stop at step 1 with AWAITING_APPROVAL -- never
    silently resolving the gate, and never running writer_agent/
    reviewer_agent on top of an unapproved write. Proves Kernel.
    run_workflow() honors POLICY_SPEC.md's Human Approval requirement
    exactly like Kernel.run() itself, even though this is now an
    automated, chained, multi-step run.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    reports_root = _make_reports_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _AttemptFindingWriteWithoutApprovalEngine(),
            documents_root=docs_root,
            findings_root=findings_root,
            reports_root=reports_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
            enable_research_write_review_workflow=True,
        )

        result = kernel.run_workflow(
            "Research the topic and write a report about it."
        )

        assert result.status == "AWAITING_APPROVAL"
        assert len(result.completed_steps) == 1
        assert result.completed_steps[0].subject == "research_agent"
        assert not Path(findings_root, "finding.md").exists()
        assert not Path(reports_root, "report.md").exists()
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(reports_root)
        shutil.rmtree(audit_dir)
