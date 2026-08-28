from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.tools.registry.tool_registry import ToolDefinition


# ---------------------------------------------------------------------
# Real read_report tool for reviewer_agent (Build Phase 11).
#
# reviewer_agent's whole purpose is to independently verify an
# already-published report against the research findings it claims to
# be based on -- it needs a way to read what writer_agent (Build Phase
# 8's write_report tool) has actually published, without being handed
# writer_agent's own write_report permission (a distinct subject
# should never inherit another subject's permission grant implicitly;
# POLICY_SPEC.md's Agent Constraints/AGENT_REGISTRY.md's Boundaries
# both require each agent's tools/permissions to be explicitly its
# own). This is a new, separate ToolDefinition/permission
# (resource=report, action=read, scope=workspace) rather than a
# shared one, granted only to reviewer_agent -- there was previously
# no way to read a published report back at all (write_report_tool.py
# only ever writes one).
#
# The sandboxing logic below is a deliberate, self-contained duplicate
# of core.tools.implementations.read_research_findings_tool's
# create_read_research_findings_executor (itself a duplicate of
# document_read_tool.py's own), following this project's existing
# per-tool-file convention -- every tool's full security-relevant
# logic is meant to be readable/auditable in one file without chasing
# a shared base across tool boundaries.
#
# Windows CRLF fix (caught by the real pytest -v run on the user's
# Windows machine during this same Build Phase 11 delivery cycle, the
# same class of environment-only bug as Pass 1's own #5/#9): see
# document_read_tool.py's own docstring for the full explanation --
# reading with the platform default silently translates on-disk
# "\r\n" to "\n", which would otherwise leave this tool's own
# `content` and `size_bytes` (computed from the real on-disk stat()
# size) inconsistent for any report with CRLF line endings.
# `open(..., newline="")` disables that translation.
# ---------------------------------------------------------------------

READ_REPORT_TOOL_ID = "read_report"

DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (".md",)
DEFAULT_MAX_BYTES = 200_000


READ_REPORT_TOOL = ToolDefinition(
    id=READ_REPORT_TOOL_ID,
    name="Read Report",
    purpose=(
        "Read the full text content of one previously-published "
        "report from the approved reports workspace, identified by "
        "a filename relative to the approved root."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
            },
        },
        "required": ["filename"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "filename": {"type": "string"},
            "content": {"type": "string"},
            "size_bytes": {"type": "integer"},
        },
        "required": ["filename", "content", "size_bytes"],
        "additionalProperties": False,
    },
    permissions=(
        "reviewer_agent:report:read:workspace",
    ),
    resource="report",
    action="read",
    scope="workspace",
    risk_level="LOW",
    error_handling={
        "retryable": False,
        "on_failure": (
            "Surface the file-read error to the agent as a failed "
            "tool result. Do not retry automatically: a filename "
            "that is missing, outside the approved root, of a "
            "disallowed type, or oversized will not become readable "
            "by retrying the identical request."
        ),
    },
)


def create_read_report_executor(
    root: str | Path,
    *,
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Callable[[str], dict[str, Any]]:
    """
    Build a real executor for READ_REPORT_TOOL, sandboxed to `root`.

    The returned callable has the exact signature ToolGateway.execute()
    calls with (`executor(**tool_kwargs)`, i.e.
    `executor(filename=...)` for this tool's input_schema).

    Mirrors create_read_research_findings_executor's sandboxing
    exactly (see that function's own docstring for the full
    reasoning): an absolute `filename` is rejected before ever being
    joined to `root`, and a relative `filename` is resolved and
    verified to still be under `root` via `Path.relative_to` (also
    catching `..`-traversal and a symlink escape).

    Raises ValueError immediately (at executor-construction time) if
    `root` does not exist or is not a directory, so a misconfigured
    deployment fails loudly at startup rather than on the agent's
    first tool call.
    """

    root_path = Path(root).resolve()

    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(
            "Reports root does not exist or is not a directory: "
            f"{root_path}"
        )

    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")

    normalized_extensions = tuple(
        extension.lower()
        for extension in allowed_extensions
    )

    def _execute(filename: str) -> dict[str, Any]:

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string.")

        requested = Path(filename)

        if requested.is_absolute():
            raise PermissionError(
                f"Filename '{filename}' must be relative to the "
                "approved reports root; absolute paths are not "
                "allowed."
            )

        candidate = (root_path / requested).resolve()

        try:
            candidate.relative_to(root_path)
        except ValueError:
            raise PermissionError(
                f"Filename '{filename}' resolves outside the "
                f"approved reports root ({root_path}); refusing to "
                "read it."
            ) from None

        if not candidate.exists():
            raise FileNotFoundError(
                f"Report not found: '{filename}'."
            )

        if not candidate.is_file():
            raise IsADirectoryError(
                f"'{filename}' is not a file."
            )

        if candidate.suffix.lower() not in normalized_extensions:
            raise ValueError(
                f"Report extension '{candidate.suffix}' is not "
                "approved for reading. Allowed extensions: "
                f"{normalized_extensions}."
            )

        size = candidate.stat().st_size

        if size > max_bytes:
            raise ValueError(
                f"Report '{filename}' is {size} bytes, exceeding the "
                f"{max_bytes}-byte limit for a single read."
            )

        try:
            with open(candidate, "r", encoding="utf-8", newline="") as f:
                content = f.read()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Report '{filename}' is not valid UTF-8 text."
            ) from exc

        return {
            "filename": filename,
            "content": content,
            "size_bytes": size,
        }

    return _execute
