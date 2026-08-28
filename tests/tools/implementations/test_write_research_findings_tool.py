"""
Tests for the real write_research_findings executor (core.tools.
implementations.write_research_findings_tool), including the
path-sandboxing rules (mirroring test_document_read_tool.py) and the
HIGH-risk/"policy"-approval gate that enforces RESEARCH_AGENT.md's
"write research findings when explicitly authorized" requirement.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.implementations.write_research_findings_tool import (
    WRITE_RESEARCH_FINDINGS_TOOL,
    WRITE_RESEARCH_FINDINGS_TOOL_ID,
    create_write_research_findings_executor,
)
from core.tools.registry.tool_registry import ToolRegistry


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


class _TempWorkspace:
    """
    Minimal helper: a temp root directory plus a temp *sibling*
    directory outside it, for path-traversal tests -- same pattern as
    test_document_read_tool.py's _TempWorkspace.
    """

    def __init__(self):
        self.parent = tempfile.mkdtemp()
        self.root = os.path.join(self.parent, "findings_root")
        self.outside = os.path.join(self.parent, "outside_root")
        os.makedirs(self.root)
        os.makedirs(self.outside)

    def cleanup(self):
        shutil.rmtree(self.parent)


# ---------------------------------------------------------------------
# ToolDefinition contract
# ---------------------------------------------------------------------

def test_write_research_findings_tool_definition_registers_cleanly():
    registry = ToolRegistry()
    registry.register(WRITE_RESEARCH_FINDINGS_TOOL)

    assert registry.contains(WRITE_RESEARCH_FINDINGS_TOOL_ID)

    tool = registry.get(WRITE_RESEARCH_FINDINGS_TOOL_ID)
    assert tool.resource == "research_findings"
    assert tool.action == "write"
    assert tool.scope == "workspace"
    assert tool.risk_level == "HIGH"


# ---------------------------------------------------------------------
# Executor construction
# ---------------------------------------------------------------------

def test_create_executor_raises_when_root_missing():
    with pytest.raises(ValueError, match="does not exist"):
        create_write_research_findings_executor(
            "/no/such/directory/anywhere"
        )


def test_create_executor_raises_when_root_is_a_file():
    workspace = _TempWorkspace()
    try:
        file_path = os.path.join(workspace.parent, "not_a_dir.txt")
        Path(file_path).write_text("x", encoding="utf-8")

        with pytest.raises(ValueError, match="not a directory"):
            create_write_research_findings_executor(file_path)
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Normal writes
# ---------------------------------------------------------------------

def test_executor_writes_a_new_finding():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(workspace.root)

        result = executor(
            filename="finding_1.md",
            content="# Finding\n\nThe sky is blue.",
        )

        assert result["path"] == "finding_1.md"
        assert result["size_bytes"] == len(
            "# Finding\n\nThe sky is blue.".encode("utf-8")
        )

        written = Path(workspace.root, "finding_1.md")
        assert written.exists()
        assert written.read_text(encoding="utf-8") == (
            "# Finding\n\nThe sky is blue."
        )
    finally:
        workspace.cleanup()


def test_executor_writes_into_a_subdirectory_of_root():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(workspace.root)

        result = executor(
            filename="topic/finding.md",
            content="Nested finding.",
        )

        assert result["path"] == "topic/finding.md"
        assert Path(workspace.root, "topic", "finding.md").read_text(
            encoding="utf-8"
        ) == "Nested finding."
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Sandbox escape attempts and write-once safety.
# ---------------------------------------------------------------------

def test_executor_rejects_absolute_filename():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(workspace.root)

        secret = os.path.join(workspace.outside, "overwrite_me.md")

        with pytest.raises(PermissionError, match="absolute paths"):
            executor(filename=secret, content="malicious content")

        assert not os.path.exists(secret)
    finally:
        workspace.cleanup()


def test_executor_rejects_relative_traversal_escaping_root():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(workspace.root)

        with pytest.raises(PermissionError, match="outside the"):
            executor(
                filename="../outside_root/escape.md",
                content="malicious content",
            )

        assert not os.path.exists(
            os.path.join(workspace.outside, "escape.md")
        )
    finally:
        workspace.cleanup()


def test_executor_rejects_existing_file():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(workspace.root)

        executor(filename="finding.md", content="Original.")

        with pytest.raises(FileExistsError, match="already"):
            executor(filename="finding.md", content="Overwrite attempt.")

        assert Path(workspace.root, "finding.md").read_text(
            encoding="utf-8"
        ) == "Original."
    finally:
        workspace.cleanup()


def test_executor_rejects_disallowed_extension():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(workspace.root)

        with pytest.raises(ValueError, match="not approved"):
            executor(filename="finding.exe", content="binary-ish")
    finally:
        workspace.cleanup()


def test_executor_rejects_oversized_content():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(
            workspace.root,
            max_bytes=10,
        )

        with pytest.raises(ValueError, match="exceeding"):
            executor(filename="big.md", content="x" * 100)
    finally:
        workspace.cleanup()


def test_executor_rejects_empty_filename():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(workspace.root)

        with pytest.raises(ValueError, match="non-empty string"):
            executor(filename="   ", content="content")
    finally:
        workspace.cleanup()


def test_executor_rejects_empty_content():
    workspace = _TempWorkspace()
    try:
        executor = create_write_research_findings_executor(workspace.root)

        with pytest.raises(ValueError, match="non-empty string"):
            executor(filename="finding.md", content="   ")
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Full ToolGateway integration: the approval gate is the point.
# ---------------------------------------------------------------------

def test_full_gateway_execution_requires_approval_without_one():
    workspace = _TempWorkspace()
    audit_dir = tempfile.mkdtemp()
    try:
        registry = ToolRegistry()
        registry.register(WRITE_RESEARCH_FINDINGS_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        executed = {"called": False}

        def executor(filename, content):
            executed["called"] = True
            return {"path": filename, "size_bytes": len(content)}

        gateway.register_executor(
            tool_id=WRITE_RESEARCH_FINDINGS_TOOL_ID,
            executor=executor,
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id=WRITE_RESEARCH_FINDINGS_TOOL_ID,
            tool_kwargs={
                "filename": "finding.md",
                "content": "Unapproved finding.",
            },
        )

        assert result.status == "APPROVAL_REQUIRED"
        assert executed["called"] is False
    finally:
        workspace.cleanup()
        shutil.rmtree(audit_dir)


def test_full_gateway_execution_succeeds_with_explicit_approval():
    workspace = _TempWorkspace()
    audit_dir = tempfile.mkdtemp()
    try:
        registry = ToolRegistry()
        registry.register(WRITE_RESEARCH_FINDINGS_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        gateway.register_executor(
            tool_id=WRITE_RESEARCH_FINDINGS_TOOL_ID,
            executor=create_write_research_findings_executor(
                workspace.root
            ),
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id=WRITE_RESEARCH_FINDINGS_TOOL_ID,
            tool_kwargs={
                "filename": "finding.md",
                "content": "Approved finding.",
            },
            approved=True,
            approved_by="human_operator",
        )

        assert result.status == "SUCCESS"

        (artifact,) = result.artifacts
        assert artifact["path"] == "finding.md"
        assert Path(workspace.root, "finding.md").read_text(
            encoding="utf-8"
        ) == "Approved finding."
    finally:
        workspace.cleanup()
        shutil.rmtree(audit_dir)


def test_full_gateway_execution_surfaces_sandbox_escape_as_tool_error():
    workspace = _TempWorkspace()
    audit_dir = tempfile.mkdtemp()
    try:
        registry = ToolRegistry()
        registry.register(WRITE_RESEARCH_FINDINGS_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        gateway.register_executor(
            tool_id=WRITE_RESEARCH_FINDINGS_TOOL_ID,
            executor=create_write_research_findings_executor(
                workspace.root
            ),
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id=WRITE_RESEARCH_FINDINGS_TOOL_ID,
            tool_kwargs={
                "filename": "../outside_root/escape.md",
                "content": "malicious content",
            },
            approved=True,
            approved_by="human_operator",
        )

        assert result.status == "ERROR"
        assert "outside the" in result.artifacts[0]
    finally:
        workspace.cleanup()
        shutil.rmtree(audit_dir)
