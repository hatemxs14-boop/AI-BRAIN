"""
Tests for the real research_agent wiring (core.agents.research_agent):
build_research_agent() and run_research_agent() assembling real tools
(web_search via Serper.dev, read_document sandboxed to a workspace
directory, read_webpage for fetching a public URL's text content,
write_research_findings for persisting an explicitly-authorized
finding) behind the full, unmodified security stack.

No real network access is used: the web_search path is exercised by
patching `requests.post` on the web_search_tool module, exactly as
tests/tools/implementations/test_web_search_tool.py already does for
that module directly; the read_webpage path similarly patches
`requests.get` and `socket.getaddrinfo` on the webpage_read_tool
module, as tests/tools/implementations/test_webpage_read_tool.py does;
write_research_findings needs no network fake at all (a real, sandboxed
filesystem write), only an isolated temp findings root.
"""
from __future__ import annotations

import os
import shutil
import socket
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
from core.policies.policy_engine import (
    AgentPermissionAlignment,
    AgentScopeEvaluation,
    PolicyEngine,
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


class _ReadWebpageThenCompleteEngine(AgentDecisionEngine):
    """
    Deterministic engine mirroring _ReadDocumentThenCompleteEngine's
    shape, but exercising read_webpage instead.
    """

    def __init__(self, url: str):
        self._url = url

    def decide(self, context: AgentContext) -> AgentAction:
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="read_webpage",
                inputs={"url": self._url},
                reason="No webpage has been read yet.",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="The requested webpage has been read.",
        )


class _WriteFindingsWithoutApprovalEngine(AgentDecisionEngine):
    """
    Attempts write_research_findings with no approval at all, proving
    the HIGH-risk/"policy" gate pauses the agent instead of writing.
    """

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


class _WriteFindingsWithApprovalEngine(AgentDecisionEngine):
    """
    Deterministic engine mirroring _ReadDocumentThenCompleteEngine's
    shape, but exercising write_research_findings with an explicit,
    attributed approval supplied up front on the AgentAction itself.
    """

    def __init__(self, filename: str, content: str):
        self._filename = filename
        self._content = content

    def decide(self, context: AgentContext) -> AgentAction:
        if not context.tool_results:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="write_research_findings",
                inputs={
                    "filename": self._filename,
                    "content": self._content,
                },
                reason="Persisting an explicitly authorized finding.",
                approved=True,
                approved_by="human_operator",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="The finding has been persisted.",
        )


def _patched_public_resolution(ip: str = "93.184.216.34"):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (ip, port),
            )
        ]

    return patch(
        "core.tools.implementations.webpage_read_tool.socket.getaddrinfo",
        side_effect=fake_getaddrinfo,
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


class _AlwaysOutOfScopePolicyEngine(PolicyEngine):
    """
    A PolicyEngine stand-in whose evaluate_agent_scope() always reports
    within_scope=False, regardless of what is actually declared/
    registered -- injected to prove build_research_agent() genuinely
    delegates to the supplied policy_engine and acts on its verdict,
    rather than always trusting its own registrations (mirrors this
    project's established genuine-delegation test pattern, e.g. the
    _should_recover()/_evaluate_policy() stand-ins used elsewhere).
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
    always reports aligned=False, regardless of what is actually
    granted/needed -- injected to prove build_research_agent()
    genuinely delegates to the supplied policy_engine's config-side
    alignment check too (Build Phase 10), not just its tool-id scope
    check (Build Phase 9). evaluate_agent_scope() is left as the real,
    inherited implementation so a normal build reaches this second
    check at all.
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


def _make_documents_root(*, with_sample=True) -> str:
    root = tempfile.mkdtemp()
    if with_sample:
        Path(root, "sample.txt").write_text(
            "The sky is blue.",
            encoding="utf-8",
        )
    return root


def _make_findings_root() -> str:
    return tempfile.mkdtemp()


def _make_memory_store_dir() -> str:
    """
    An isolated, per-test directory to hold a memory.jsonl (Build
    Phase 14) -- same isolation discipline as
    _make_documents_root()/_make_findings_root() above, so no test in
    this file ever touches the real workspace/project_memory/
    memory.jsonl the default build wires research_agent to in
    production. Returns the directory (not the file path) so callers
    can shutil.rmtree it in a `finally` block exactly like every other
    tmp resource in this file.
    """
    return tempfile.mkdtemp()


def _memory_store_path(memory_dir: str) -> str:
    return str(Path(memory_dir) / "memory.jsonl")


# ---------------------------------------------------------------------
# build_research_agent(): tool discovery and configuration errors
# ---------------------------------------------------------------------

def test_build_research_agent_exposes_exactly_the_five_wired_tools():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    memory_dir = _make_memory_store_dir()
    audit_dir = tempfile.mkdtemp()
    try:
        agent = build_research_agent(
            documents_root=docs_root,
            findings_root=findings_root,
            memory_store_path=_memory_store_path(memory_dir),
            serper_api_key="test-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        assert agent.identity.subject == RESEARCH_AGENT_SUBJECT

        tool_ids = {tool.id for tool in agent.discover_tools()}

        assert tool_ids == {
            "web_search",
            "read_document",
            "read_webpage",
            "write_research_findings",
            "read_project_memory",
        }
        assert "shell" not in tool_ids
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(memory_dir)
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


def test_build_research_agent_raises_when_policy_engine_reports_out_of_scope():
    """
    Genuine delegation proof (Build Phase 9): with an otherwise
    completely normal build -- real, valid documents_root/findings_root,
    real Serper key -- injecting a policy_engine whose
    evaluate_agent_scope() reports within_scope=False must still make
    build_research_agent() raise ValueError. This proves the scope
    check actually asks the supplied policy_engine and acts on its
    answer, rather than the raise being unreachable dead code.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    try:
        with pytest.raises(ValueError, match="silently expanded"):
            build_research_agent(
                documents_root=docs_root,
                findings_root=findings_root,
                serper_api_key="test-key",
                policy_engine=_AlwaysOutOfScopePolicyEngine(),
            )
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)


def test_build_research_agent_raises_when_policy_engine_reports_misaligned_permissions():
    """
    Genuine delegation proof (Build Phase 10): with an otherwise
    completely normal build against the real, aligned permissions.json,
    injecting a policy_engine whose evaluate_agent_permission_alignment()
    reports aligned=False must still make build_research_agent() raise
    ValueError -- proving the config-side alignment check actually asks
    the supplied policy_engine and acts on its answer.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    try:
        with pytest.raises(ValueError, match="drifted"):
            build_research_agent(
                documents_root=docs_root,
                findings_root=findings_root,
                serper_api_key="test-key",
                policy_engine=_AlwaysMisalignedPolicyEngine(),
            )
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)


# ---------------------------------------------------------------------
# run_research_agent(): full end-to-end loops
# ---------------------------------------------------------------------

def test_run_research_agent_completes_a_web_search_task():
    docs_root = _make_documents_root()
    memory_dir = _make_memory_store_dir()
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
                memory_store_path=_memory_store_path(memory_dir),
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
        shutil.rmtree(memory_dir)
        shutil.rmtree(audit_dir)


def test_run_research_agent_completes_a_read_document_task():
    docs_root = _make_documents_root()
    memory_dir = _make_memory_store_dir()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_research_agent(
            "Summarize sample.txt.",
            decision_engine=_ReadDocumentThenCompleteEngine("sample.txt"),
            documents_root=docs_root,
            memory_store_path=_memory_store_path(memory_dir),
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
        shutil.rmtree(memory_dir)
        shutil.rmtree(audit_dir)


def test_run_research_agent_completes_a_read_webpage_task():
    docs_root = _make_documents_root()
    memory_dir = _make_memory_store_dir()
    audit_dir = tempfile.mkdtemp()
    try:
        html = (
            b"<html><head><title>Evidence Page</title></head>"
            b"<body><p>The sky is blue.</p></body></html>"
        )

        with patch(
            "core.tools.implementations.webpage_read_tool.requests.get"
        ) as mock_get, _patched_public_resolution():
            mock_get.return_value.status_code = 200
            mock_get.return_value.content = html
            mock_get.return_value.headers = {}

            result = run_research_agent(
                "Read https://example.com/evidence.",
                decision_engine=_ReadWebpageThenCompleteEngine(
                    "https://example.com/evidence"
                ),
                documents_root=docs_root,
                memory_store_path=_memory_store_path(memory_dir),
                serper_api_key="unused-but-required-key",
                audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
                max_steps=3,
            )

        assert result.status == "COMPLETED"
        assert result.last_result is not None
        assert result.last_result.status == "SUCCESS"

        (artifact,) = result.last_result.artifacts
        assert artifact["title"] == "Evidence Page"
        assert "The sky is blue." in artifact["content"]
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(memory_dir)
        shutil.rmtree(audit_dir)


def test_run_research_agent_pauses_for_approval_when_writing_findings():
    """
    write_research_findings is HIGH risk / "policy" approval by
    design (see core.agents.research_agent's module docstring and
    write_research_findings_tool's own docstring): a decision engine
    that requests it with no approval at all must pause the agent
    exactly like tests/agents/test_agent_await_approval.py's generic
    HIGH-risk fixture does -- never write anything.
    """
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    memory_dir = _make_memory_store_dir()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_research_agent(
            "Persist a finding.",
            decision_engine=_WriteFindingsWithoutApprovalEngine(
                "finding.md"
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            memory_store_path=_memory_store_path(memory_dir),
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            max_steps=3,
        )

        assert result.status == "APPROVAL_REQUIRED"
        assert not Path(findings_root, "finding.md").exists()
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(memory_dir)
        shutil.rmtree(audit_dir)


def test_run_research_agent_completes_a_write_research_findings_task():
    docs_root = _make_documents_root()
    findings_root = _make_findings_root()
    memory_dir = _make_memory_store_dir()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_research_agent(
            "Persist a finding.",
            decision_engine=_WriteFindingsWithApprovalEngine(
                "finding.md",
                "The sky is blue, confirmed by two independent sources.",
            ),
            documents_root=docs_root,
            findings_root=findings_root,
            memory_store_path=_memory_store_path(memory_dir),
            serper_api_key="unused-but-required-key",
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
            max_steps=3,
        )

        assert result.status == "COMPLETED"
        assert result.last_result is not None
        assert result.last_result.status == "SUCCESS"

        (artifact,) = result.last_result.artifacts
        assert artifact["path"] == "finding.md"

        written = Path(findings_root, "finding.md")
        assert written.exists()
        assert "confirmed by two independent sources" in (
            written.read_text(encoding="utf-8")
        )
    finally:
        shutil.rmtree(docs_root)
        shutil.rmtree(findings_root)
        shutil.rmtree(memory_dir)
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
    permissions.json no longer grants research_agent any shell-related
    permission at all -- the earlier HIGH-risk resource=shell/
    action=execute/scope=workspace entry (present since early test
    fixtures, unrelated to this build phase) was removed once it was
    identified as inconsistent with RESEARCH_AGENT.md's own allowed-
    tools list (see the module docstring of core.agents.research_agent
    for the full resolution). Independent of that removal, this
    module also never registers a ToolDefinition for resource=shell,
    so a "shell" tool_id must be inert either way: discovery must not
    expose it, and requesting it by tool_id must fail before reaching
    any executor.
    """
    docs_root = _make_documents_root()
    memory_dir = _make_memory_store_dir()
    audit_dir = tempfile.mkdtemp()
    try:
        result = run_research_agent(
            "Try to run a shell command.",
            decision_engine=_RequestUnavailableShellEngine(),
            documents_root=docs_root,
            memory_store_path=_memory_store_path(memory_dir),
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
        shutil.rmtree(memory_dir)
        shutil.rmtree(audit_dir)
