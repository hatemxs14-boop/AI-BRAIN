"""
Tests for the real read_report executor (core.tools.implementations.
read_report_tool), including the path-sandboxing rules that keep it
from ever reading anything outside its configured root directory.
Mirrors tests/tools/implementations/test_read_research_findings_tool.py's
structure closely -- this tool's sandboxing logic is a deliberate
duplicate of that one (see the tool module's own docstring for why).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.implementations.read_report_tool import (
    READ_REPORT_TOOL,
    READ_REPORT_TOOL_ID,
    create_read_report_executor,
)
from core.tools.registry.tool_registry import ToolRegistry


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


class _TempWorkspace:
    """
    Minimal helper: a temp root directory plus a temp *sibling*
    directory outside it, for path-traversal tests. See
    test_document_read_tool.py's own docstring for why this is a
    plain class, not a pytest fixture.
    """

    def __init__(self):
        self.parent = tempfile.mkdtemp()
        self.root = os.path.join(self.parent, "reports_root")
        self.outside = os.path.join(self.parent, "outside_root")
        os.makedirs(self.root)
        os.makedirs(self.outside)

    def cleanup(self):
        shutil.rmtree(self.parent)


# ---------------------------------------------------------------------
# ToolDefinition contract
# ---------------------------------------------------------------------

def test_read_report_tool_definition_registers_cleanly():
    registry = ToolRegistry()
    registry.register(READ_REPORT_TOOL)

    assert registry.contains(READ_REPORT_TOOL_ID)

    tool = registry.get(READ_REPORT_TOOL_ID)
    assert tool.resource == "report"
    assert tool.action == "read"
    assert tool.scope == "workspace"
    assert tool.risk_level == "LOW"


# ---------------------------------------------------------------------
# Executor construction
# ---------------------------------------------------------------------

def test_create_executor_raises_when_root_missing():
    with pytest.raises(ValueError, match="does not exist"):
        create_read_report_executor(
            "/no/such/directory/anywhere"
        )


def test_create_executor_raises_when_root_is_a_file():
    workspace = _TempWorkspace()
    try:
        file_path = os.path.join(workspace.parent, "not_a_dir.md")
        Path(file_path).write_text("x", encoding="utf-8")

        with pytest.raises(ValueError, match="not a directory"):
            create_read_report_executor(file_path)
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Normal reads
# ---------------------------------------------------------------------

def test_executor_reads_an_existing_report():
    """
    Content deliberately spans multiple lines. The fixture is written
    with write_bytes() (raw UTF-8 bytes, "\\n" literally) rather than
    write_text() specifically so this test is platform-independent:
    write_text()'s default universal-newline translation would
    silently write "\\r\\n" on Windows, which -- before this same
    Build Phase 11 delivery cycle's Windows CRLF fix (see
    read_report_tool.py's own docstring) -- made this exact assertion
    fail on a real Windows pytest run (size_bytes/content mismatch).
    """
    workspace = _TempWorkspace()
    try:
        Path(workspace.root, "report.md").write_bytes(
            "# Report\n\nThe sky is blue.".encode("utf-8")
        )

        executor = create_read_report_executor(workspace.root)
        result = executor(filename="report.md")

        assert result["filename"] == "report.md"
        assert result["content"] == "# Report\n\nThe sky is blue."
        assert result["size_bytes"] == len(
            "# Report\n\nThe sky is blue.".encode("utf-8")
        )
    finally:
        workspace.cleanup()


def test_executor_reads_crlf_content_byte_exact_no_platform_translation():
    """
    Windows CRLF fix (Build Phase 11 delivery cycle): mirrors
    test_read_research_findings_tool.py's own equivalent test -- see
    its docstring for the full explanation. Deterministic on every
    platform since the fixture writes literal "\\r\\n" bytes directly.
    """
    workspace = _TempWorkspace()
    try:
        raw = b"Line one.\r\n\r\nLine two."
        Path(workspace.root, "report.md").write_bytes(raw)

        executor = create_read_report_executor(workspace.root)
        result = executor(filename="report.md")

        assert result["content"] == raw.decode("utf-8")
        assert result["size_bytes"] == len(raw)
        assert len(result["content"].encode("utf-8")) == result["size_bytes"]
    finally:
        workspace.cleanup()


def test_executor_reads_from_a_subdirectory_of_root():
    workspace = _TempWorkspace()
    try:
        subdir = os.path.join(workspace.root, "sub")
        os.makedirs(subdir)
        Path(subdir, "report.md").write_text(
            "# Heading",
            encoding="utf-8",
        )

        executor = create_read_report_executor(workspace.root)
        result = executor(filename="sub/report.md")

        assert result["content"] == "# Heading"
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Sandbox escape attempts -- the core security property of this tool.
# ---------------------------------------------------------------------

def test_executor_rejects_absolute_filename():
    workspace = _TempWorkspace()
    try:
        secret = os.path.join(workspace.outside, "secret.md")
        Path(secret).write_text("outside content", encoding="utf-8")

        executor = create_read_report_executor(workspace.root)

        with pytest.raises(PermissionError, match="absolute paths"):
            executor(filename=secret)
    finally:
        workspace.cleanup()


def test_executor_rejects_relative_traversal_escaping_root():
    workspace = _TempWorkspace()
    try:
        Path(workspace.outside, "secret.md").write_text(
            "outside content",
            encoding="utf-8",
        )

        executor = create_read_report_executor(workspace.root)

        with pytest.raises(PermissionError, match="outside the approved"):
            executor(filename="../outside_root/secret.md")
    finally:
        workspace.cleanup()


def test_executor_rejects_missing_file():
    workspace = _TempWorkspace()
    try:
        executor = create_read_report_executor(workspace.root)

        with pytest.raises(FileNotFoundError):
            executor(filename="does_not_exist.md")
    finally:
        workspace.cleanup()


def test_executor_rejects_directory_path():
    workspace = _TempWorkspace()
    try:
        os.makedirs(os.path.join(workspace.root, "a_directory.md"))

        executor = create_read_report_executor(workspace.root)

        with pytest.raises(IsADirectoryError):
            executor(filename="a_directory.md")
    finally:
        workspace.cleanup()


def test_executor_rejects_disallowed_extension():
    workspace = _TempWorkspace()
    try:
        Path(workspace.root, "script.exe").write_bytes(b"binary")

        executor = create_read_report_executor(workspace.root)

        with pytest.raises(ValueError, match="not approved"):
            executor(filename="script.exe")
    finally:
        workspace.cleanup()


def test_executor_rejects_oversized_file():
    workspace = _TempWorkspace()
    try:
        Path(workspace.root, "big.md").write_text(
            "x" * 100,
            encoding="utf-8",
        )

        executor = create_read_report_executor(
            workspace.root,
            max_bytes=10,
        )

        with pytest.raises(ValueError, match="exceeding"):
            executor(filename="big.md")
    finally:
        workspace.cleanup()


def test_executor_rejects_non_utf8_content():
    workspace = _TempWorkspace()
    try:
        Path(workspace.root, "bad_encoding.md").write_bytes(
            b"\xff\xfe not valid utf-8"
        )

        executor = create_read_report_executor(workspace.root)

        with pytest.raises(ValueError, match="not valid UTF-8"):
            executor(filename="bad_encoding.md")
    finally:
        workspace.cleanup()


def test_executor_rejects_empty_filename():
    workspace = _TempWorkspace()
    try:
        executor = create_read_report_executor(workspace.root)

        with pytest.raises(ValueError, match="non-empty string"):
            executor(filename="   ")
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Full ToolGateway integration (real security stack)
# ---------------------------------------------------------------------

def test_full_gateway_execution_succeeds_without_approval():
    workspace = _TempWorkspace()
    audit_dir = tempfile.mkdtemp()
    try:
        Path(workspace.root, "report.md").write_text(
            "Report content goes here.",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.register(READ_REPORT_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        gateway.register_executor(
            tool_id=READ_REPORT_TOOL_ID,
            executor=create_read_report_executor(
                workspace.root
            ),
        )

        result = gateway.execute(
            subject="reviewer_agent",
            tool_id=READ_REPORT_TOOL_ID,
            tool_kwargs={"filename": "report.md"},
        )

        assert result.status == "SUCCESS"
        assert result.security_decision.decision.value == "ALLOW"
        assert result.subject == "reviewer_agent"
        assert result.tool_id == READ_REPORT_TOOL_ID
        assert result.action == "read"

        (artifact,) = result.artifacts
        assert artifact["content"] == "Report content goes here."
    finally:
        workspace.cleanup()
        shutil.rmtree(audit_dir)


def test_full_gateway_execution_denies_a_subject_with_no_grant():
    """
    writer_agent has no permission for resource=report/action=read --
    only reviewer_agent does (writer_agent's own grant is
    report:write, not report:read). A distinct subject never inherits
    another subject's permission implicitly.
    """
    workspace = _TempWorkspace()
    audit_dir = tempfile.mkdtemp()
    try:
        Path(workspace.root, "report.md").write_text(
            "Report content goes here.",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.register(READ_REPORT_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        gateway.register_executor(
            tool_id=READ_REPORT_TOOL_ID,
            executor=create_read_report_executor(
                workspace.root
            ),
        )

        result = gateway.execute(
            subject="writer_agent",
            tool_id=READ_REPORT_TOOL_ID,
            tool_kwargs={"filename": "report.md"},
        )

        assert result.status == "DENIED"
    finally:
        workspace.cleanup()
        shutil.rmtree(audit_dir)


def test_full_gateway_execution_surfaces_sandbox_escape_as_tool_error():
    workspace = _TempWorkspace()
    audit_dir = tempfile.mkdtemp()
    try:
        Path(workspace.outside, "secret.md").write_text(
            "should never be readable",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.register(READ_REPORT_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        gateway.register_executor(
            tool_id=READ_REPORT_TOOL_ID,
            executor=create_read_report_executor(
                workspace.root
            ),
        )

        result = gateway.execute(
            subject="reviewer_agent",
            tool_id=READ_REPORT_TOOL_ID,
            tool_kwargs={"filename": "../outside_root/secret.md"},
        )

        assert result.status == "ERROR"
        assert "outside the approved" in result.artifacts[0]
    finally:
        workspace.cleanup()
        shutil.rmtree(audit_dir)
