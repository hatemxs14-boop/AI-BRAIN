"""
End-to-end tests wiring the real Kernel (core.kernel.kernel) to the
real research_agent stack (core.agents.research_agent) through
core.kernel.default_kernel.build_default_kernel().

Mirrors the patterns already established in
tests/agents/test_research_agent_wiring.py (a local deterministic
decision engine per scenario, isolated temp roots for documents/
findings/audit) rather than importing from that file, so this file
stays a self-contained proof that the full stack -- Kernel ->
OrchestrationEngine -> AgentCore -> ToolRuntime -> Security Layer ->
real research_agent tools -- composes correctly end to end.
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


class _WriteFindingsWithoutApprovalEngine(AgentDecisionEngine):
    def __init__(self, filename: str):
        self._filename = filename

    def decide(self, context: AgentContext) -> AgentAction:
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="write_research_findings",
            inputs={
                "filename": self._filename,
                "content": "This should not be written without approval.",
            },
            reason="Attempting to write a finding without approval.",
        )


def _make_documents_root() -> str:
    root = tempfile.mkdtemp()
    Path(root, "sample.txt").write_text(
        "The sky is blue.",
        encoding="utf-8",
    )
    return root


def _make_findings_root() -> str:
    return tempfile.mkdtemp()


def test_build_default_kernel_registers_all_three_agents():
    """
    As of Build Phase 11, build_default_kernel() registers
    research_agent, writer_agent, and reviewer_agent, in that order
    (see core/kernel/default_kernel.py's own docstring for why
    registration order matters as the genuine-tie tiebreak).
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadDocumentThenCompleteEngine(
                "sample.txt"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        assert [r.subject for r in kernel._registrations] == [
            "research_agent",
            "writer_agent",
            "reviewer_agent",
        ]
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(audit_dir)


def test_kernel_run_completes_a_read_document_task_through_research_agent():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _ReadDocumentThenCompleteEngine(
                "sample.txt"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        # "Read document" (not "Summarize") deliberately: this task
        # must classify to research_agent, not writer_agent -- see
        # core/kernel/default_kernel.py's _RESEARCH_AGENT_KEYWORDS/
        # _WRITER_AGENT_KEYWORDS.
        result = kernel.run("Read document sample.txt.", max_steps=3)

        assert result.status == "COMPLETED"
        assert result.subject == "research_agent"
        assert result.verification.passed is True
        assert result.loop_result.last_result.status == "SUCCESS"

        (artifact,) = result.loop_result.last_result.artifacts
        assert artifact["content"] == "The sky is blue."
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(audit_dir)


def test_kernel_run_surfaces_write_findings_approval_required_through_research_agent():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    audit_dir = tempfile.mkdtemp()
    try:
        kernel = build_default_kernel(
            decision_engine_factory=lambda: _WriteFindingsWithoutApprovalEngine(
                "finding.md"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            orchestration_engine=SequentialOrchestrationEngine(),
        )

        # "Research and persist" (not just "Persist") deliberately:
        # this task must classify to research_agent, not
        # NO_AGENT_AVAILABLE -- see core/kernel/default_kernel.py's
        # _RESEARCH_AGENT_KEYWORDS/_WRITER_AGENT_KEYWORDS.
        result = kernel.run("Research and persist a finding.", max_steps=3)

        assert result.status == "AWAITING_APPROVAL"
        assert not Path(findings_root, "finding.md").exists()
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(audit_dir)


def test_build_default_kernel_requires_llm_client_factory_or_decision_engine_factory():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    try:
        with pytest.raises(
            ValueError,
            match="llm_client_factory or decision_engine_factory",
        ):
            build_default_kernel(
                documents_root=docs_root,
                findings_root=findings_root,
                serper_api_key="test-key",
            )
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
