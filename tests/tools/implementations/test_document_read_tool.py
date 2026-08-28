"""
Tests for the real read_document executor (core.tools.implementations.
document_read_tool), including the path-sandboxing rules that keep it
from ever reading anything outside its configured root directory.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.implementations.document_read_tool import (
    READ_DOCUMENT_TOOL,
    READ_DOCUMENT_TOOL_ID,
    create_document_read_executor,
)
from core.tools.registry.tool_registry import ToolRegistry


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


class _TempWorkspace:
    """
    Minimal helper: a temp root directory plus a temp *sibling*
    directory outside it, for path-traversal tests. Deliberately not a
    pytest fixture -- this project's sandbox test runner does not
    support fixture injection (see tests/security/
    test_effective_risk_floor.py's _build_conservative_gateway for the
    established pattern).
    """

    def __init__(self):
        self.parent = tempfile.mkdtemp()
        self.root = os.path.join(self.parent, "documents_root")
        self.outside = os.path.join(self.parent, "outside_root")
        os.makedirs(self.root)
        os.makedirs(self.outside)

    def cleanup(self):
        shutil.rmtree(self.parent)


# ---------------------------------------------------------------------
# ToolDefinition contract
# ---------------------------------------------------------------------

def test_read_document_tool_definition_registers_cleanly():
    registry = ToolRegistry()
    registry.register(READ_DOCUMENT_TOOL)

    assert registry.contains(READ_DOCUMENT_TOOL_ID)

    tool = registry.get(READ_DOCUMENT_TOOL_ID)
    assert tool.resource == "document"
    assert tool.action == "read"
    assert tool.scope == "workspace"
    assert tool.risk_level == "LOW"


# ---------------------------------------------------------------------
# Executor construction
# ---------------------------------------------------------------------

def test_create_executor_raises_when_root_missing():
    with pytest.raises(ValueError, match="does not exist"):
        create_document_read_executor("/no/such/directory/anywhere")


def test_create_executor_raises_when_root_is_a_file():
    workspace = _TempWorkspace()
    try:
        file_path = os.path.join(workspace.parent, "not_a_dir.txt")
        Path(file_path).write_text("x", encoding="utf-8")

        with pytest.raises(ValueError, match="not a directory"):
            create_document_read_executor(file_path)
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Normal reads
# ---------------------------------------------------------------------

def test_executor_reads_an_existing_text_file():
    workspace = _TempWorkspace()
    try:
        Path(workspace.root, "notes.txt").write_text(
            "The sky is blue.",
            encoding="utf-8",
        )

        executor = create_document_read_executor(workspace.root)
        result = executor(path="notes.txt")

        assert result["path"] == "notes.txt"
        assert result["content"] == "The sky is blue."
        assert result["size_bytes"] == len(
            "The sky is blue.".encode("utf-8")
        )
    finally:
        workspace.cleanup()


def test_executor_reads_crlf_content_byte_exact_no_platform_translation():
    """
    Windows CRLF fix (Build Phase 11 delivery cycle, caught by a real
    pytest -v run on the user's Windows machine while adding
    core/tools/implementations/read_report_tool.py): before this fix,
    reading via the platform default silently translated on-disk
    "\\r\\n" to "\\n", which would make the returned `content`
    shorter, in bytes, than `size_bytes` (computed from the real
    on-disk `stat()` size). The fixture here writes literal "\\r\\n"
    bytes directly (not through write_text(), whose own behavior is
    platform-dependent) so this test is deterministic on every
    platform, not just Windows.
    """
    workspace = _TempWorkspace()
    try:
        raw = b"Line one.\r\n\r\nLine two."
        Path(workspace.root, "doc.txt").write_bytes(raw)

        executor = create_document_read_executor(workspace.root)
        result = executor(path="doc.txt")

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
        Path(subdir, "notes.md").write_text(
            "# Heading",
            encoding="utf-8",
        )

        executor = create_document_read_executor(workspace.root)
        result = executor(path="sub/notes.md")

        assert result["content"] == "# Heading"
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Sandbox escape attempts -- the core security property of this tool.
# ---------------------------------------------------------------------

def test_executor_rejects_absolute_path():
    """
    Path("root") / "/etc/passwd" == Path("/etc/passwd") under pathlib's
    join semantics -- an absolute `path` would silently replace `root`
    entirely if not rejected before the join. This is the single most
    important test in this file.
    """
    workspace = _TempWorkspace()
    try:
        secret = os.path.join(workspace.outside, "secret.txt")
        Path(secret).write_text("outside content", encoding="utf-8")

        executor = create_document_read_executor(workspace.root)

        with pytest.raises(PermissionError, match="absolute paths"):
            executor(path=secret)
    finally:
        workspace.cleanup()


def test_executor_rejects_relative_traversal_escaping_root():
    workspace = _TempWorkspace()
    try:
        Path(workspace.outside, "secret.txt").write_text(
            "outside content",
            encoding="utf-8",
        )

        executor = create_document_read_executor(workspace.root)

        with pytest.raises(PermissionError, match="outside the approved"):
            executor(path="../outside_root/secret.txt")
    finally:
        workspace.cleanup()


def test_executor_rejects_missing_file():
    workspace = _TempWorkspace()
    try:
        executor = create_document_read_executor(workspace.root)

        with pytest.raises(FileNotFoundError):
            executor(path="does_not_exist.txt")
    finally:
        workspace.cleanup()


def test_executor_rejects_directory_path():
    workspace = _TempWorkspace()
    try:
        os.makedirs(os.path.join(workspace.root, "a_directory.txt"))

        executor = create_document_read_executor(workspace.root)

        with pytest.raises(IsADirectoryError):
            executor(path="a_directory.txt")
    finally:
        workspace.cleanup()


def test_executor_rejects_disallowed_extension():
    workspace = _TempWorkspace()
    try:
        Path(workspace.root, "script.exe").write_bytes(b"binary")

        executor = create_document_read_executor(workspace.root)

        with pytest.raises(ValueError, match="not approved"):
            executor(path="script.exe")
    finally:
        workspace.cleanup()


def test_executor_rejects_oversized_file():
    workspace = _TempWorkspace()
    try:
        Path(workspace.root, "big.txt").write_text(
            "x" * 100,
            encoding="utf-8",
        )

        executor = create_document_read_executor(
            workspace.root,
            max_bytes=10,
        )

        with pytest.raises(ValueError, match="exceeding"):
            executor(path="big.txt")
    finally:
        workspace.cleanup()


def test_executor_rejects_non_utf8_content():
    workspace = _TempWorkspace()
    try:
        Path(workspace.root, "bad_encoding.txt").write_bytes(
            b"\xff\xfe not valid utf-8"
        )

        executor = create_document_read_executor(workspace.root)

        with pytest.raises(ValueError, match="not valid UTF-8"):
            executor(path="bad_encoding.txt")
    finally:
        workspace.cleanup()


def test_executor_rejects_empty_path():
    workspace = _TempWorkspace()
    try:
        executor = create_document_read_executor(workspace.root)

        with pytest.raises(ValueError, match="non-empty string"):
            executor(path="   ")
    finally:
        workspace.cleanup()


# ---------------------------------------------------------------------
# Full ToolGateway integration (real security stack)
# ---------------------------------------------------------------------

def test_full_gateway_execution_succeeds_without_approval():
    workspace = _TempWorkspace()
    audit_dir = tempfile.mkdtemp()
    try:
        Path(workspace.root, "notes.txt").write_text(
            "Evidence goes here.",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.register(READ_DOCUMENT_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        gateway.register_executor(
            tool_id=READ_DOCUMENT_TOOL_ID,
            executor=create_document_read_executor(workspace.root),
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id=READ_DOCUMENT_TOOL_ID,
            tool_kwargs={"path": "notes.txt"},
        )

        assert result.status == "SUCCESS"
        assert result.security_decision.decision.value == "ALLOW"

        (artifact,) = result.artifacts
        assert artifact["content"] == "Evidence goes here."
    finally:
        workspace.cleanup()
        shutil.rmtree(audit_dir)


def test_full_gateway_execution_surfaces_sandbox_escape_as_tool_error():
    """
    Even if a decision engine were ever tricked into requesting a path
    outside the sandbox, the failure must surface as a normal ERROR
    tool result through the Gateway -- not a crash, and never a
    successful read of the escaped file.
    """
    workspace = _TempWorkspace()
    audit_dir = tempfile.mkdtemp()
    try:
        Path(workspace.outside, "secret.txt").write_text(
            "should never be readable",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.register(READ_DOCUMENT_TOOL)

        security = SecurityDecisionPoint(
            PERMISSIONS_FILE,
            audit_log_path=os.path.join(audit_dir, "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)

        gateway.register_executor(
            tool_id=READ_DOCUMENT_TOOL_ID,
            executor=create_document_read_executor(workspace.root),
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id=READ_DOCUMENT_TOOL_ID,
            tool_kwargs={"path": "../outside_root/secret.txt"},
        )

        assert result.status == "ERROR"
        assert "outside the approved" in result.artifacts[0]
    finally:
        workspace.cleanup()
        shutil.rmtree(audit_dir)
