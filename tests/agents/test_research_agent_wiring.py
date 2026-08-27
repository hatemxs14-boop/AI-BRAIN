"""
Tests for the real research_agent wiring (core.agents.research_agent):
build_research_agent() and run_research_agent() assembling real tools
(web_search via Serper.dev, read_document sandboxed to a workspace
directory) behind the full, unmodified security stack.

No real network access is used: the web_search path is exercised by
patching `requests.post` on the web_search_tool module, exactly as
tests/tools/implementations/test_web_search_tool.py already does for
that module directly.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.deterministic_decision_engine import (
    DeterministicDecisionEngine,
)
from core.agents.research_agent import (
    RESEARCH_AGENT_SUBJECT,
    build_research_agent,
    run_research_agent,
)


class _ReadDocumentThenCompleteEngine(AgentDecisionEngine):
    """
    Deterministic engine mirroring DeterministicDecisionEngine's shape,
    but exercising read_document instead of web_search.
    """

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


class _RequestUnavailableShellEngine(AgentDecisionEngine):
    """
    Attempts to invoke a "shell" tool_id that research_agent's real
    wiring never registers -- proving the documented-not-wired shell
    permission grants nothing in practice (see the module docstring of
    core.agents.research_agent).
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


def _make_documents_root(*, with_sample=True) -> str:
    root = tempfile.mkdtemp()
    if with_sample:
        Path(root, "sample.txt").write_text(
            "The sky is blue.",
            encoding="utf-8",
        )
    return root


# ---------------------------------------------------------------------
# build_research_agent(): tool discovery and configuration errors
# ---------------------------------------------------------------------

def test_build_research_agent_exposes_exactly_the_two_wired_tools():
    docs_root = _make_documents_root()
    audit_dir = tempfile.mkdtemp()
    try:
        agent = build_research_agent(
            documents_root=docs_root,
            serper_api_key="test-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        assert agent.identity.subject == RESEARCH_AGENT_SUBJECT

        tool_ids = {tool.id for tool in agent.discover_tools()}

        assert tool_ids == {"web_search", "read_document"}
        assert "shell" not in tool_ids
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(audit_dir)


def test_build_research_agent_raises_clear_error_without_serper_key():
    docs_root = _make_documents_root()
    original = os.environ.pop("SERPER_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="Serper.dev API key"):
            build_research_agent(documents_root=docs_root)
    finally:
        shutil.rmtree(docs_root)
        if original is not None:
            os.environ["SERPER_API_KEY"] = original


def test_build_research_agent_raises_clear_error_for_missing_documents_root():
    with pytest.raises(ValueError, match="does not exist"):
        build_research_agent(
            documents_root="/no/such/directory/anywhere",
            serper_api_key="test-key",
        )


# ---------------------------------------------------------------------
# run_research_agent(): full end-to-end loops
# ---------------------------------------------------------------------

def test_run_research_agent_completes_a_web_search_task():
    docs_root = _make_documents_root()
    audit_dir = tempfile.mkdtemp()
    try:
        organic = [
            {
                "title": "AI Agents Explained",
                "link": "https://example.com/ai-agents",
                "snippet": "An overview of AI agents.",
            }
        ]

        with patch(
            "core.tools.implementations.web_search_tool.requests.post"
        ) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "organic": organic
            }

            result = run_research_agent(
                "Research AI agent frameworks.",
                decision_engine=DeterministicDecisionEngine(),
                documents_root=docs_root,
                serper_api_key="test-key",
                audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
                max_steps=3,
            )

        assert result.status == "COMPLETED"
        assert result.last_result is not None
        assert result.last_result.status == "SUCCESS"

        (artifact,) = result.last_result.artifacts
        assert artifact["results"][0]["title"] == "AI Agents Explained"
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(audit_dir)


def test_run_research_agent_completes_a_read_document_task():
    docs_root = _make_documents_root()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_research_agent(
            "Summarize sample.txt.",
            decision_engine=_ReadDocumentThenCompleteEngine("sample.txt"),
            documents_root=docs_root,
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            max_steps=3,
        )

        assert result.status == "COMPLETED"
        assert result.last_result is not None
        assert result.last_result.status == "SUCCESS"

        (artifact,) = result.last_result.artifacts
        assert artifact["content"] == "The sky is blue."
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(audit_dir)


def test_run_research_agent_requires_llm_client_or_decision_engine():
    docs_root = _make_documents_root()
    try:
        with pytest.raises(
            ValueError,
            match="llm_client or decision_engine",
        ):
            run_research_agent(
                "Do something.",
                documents_root=docs_root,
                serper_api_key="test-key",
            )
    finally:
        shutil.rmtree(docs_root)


def test_shell_permission_grants_nothing_because_no_tool_is_registered():
    """
    permissions.json still grants research_agent a HIGH-risk
    resource=shell/action=execute/scope=workspace permission (present
    since early test fixtures, unrelated to this build phase -- see
    the module docstring of core.agents.research_agent for why it was
    deliberately left unwired rather than silently built or removed).
    A latent permission with no registered ToolDefinition behind it
    must be inert: discovery must not expose a "shell" tool, and
    requesting one by tool_id must fail with "tool not available",
    never reach any executor.
    """
    docs_root = _make_documents_root()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_research_agent(
            "Try to run a shell command.",
            decision_engine=_RequestUnavailableShellEngine(),
            documents_root=docs_root,
            serper_api_key="test-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            max_steps=3,
        )

        # AgentToolInterface.create_invocation() raises PermissionError
        # for a tool_id the subject cannot discover; AgentExecutionLoop
        # surfaces that as EXECUTION_ERROR, not a successful shell run.
        assert result.status == "EXECUTION_ERROR"
        assert result.last_result is None
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(audit_dir)
