from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.tools.registry.tool_registry import ToolDefinition


# ---------------------------------------------------------------------
# Real read_research_findings tool for writer_agent (Build Phase 8).
#
# writer_agent's whole purpose is to synthesize already-persisted
# research findings into a written report -- it needs a way to read
# what research_agent (Build Phase 3's write_research_findings tool)
# has actually written, without being handed research_agent's own
# read_document/document:read:workspace permission (a distinct
# subject should never inherit another subject's permission grant
# implicitly; POLICY_SPEC.md's Agent Constraints/AGENT_REGISTRY.md's
# Boundaries both require each agent's tools/permissions to be
# explicitly its own). This is a new, separate ToolDefinition/
# permission (resource=research_findings, action=read,
# scope=workspace) rather than a shared one, granted only to
# writer_agent.
#
# The sandboxing logic below is a deliberate, self-contained
# duplicate of core.tools.implementations.document_read_tool's
# create_document_read_executor, following this project's existing
# per-tool-file convention (write_research_findings_tool.py already
# duplicates document_read_tool.py's escape-path checks rather than
# sharing a helper) -- every tool's full security-relevant logic is
# meant to be readable/auditable in one file without chasing a shared
# base across tool boundaries.
#
# Second subject, Build Phase 11: reviewer_agent independently
# verifies a published report against the same persisted findings
# writer_agent reads -- genuinely the same resource, same root, same
# trust boundary, not a distinct capability that needs its own tool
# module the way writer_agent's own read access (a different resource
# entirely from research_agent's document:read:workspace) did. This
# ToolDefinition's `permissions` tuple therefore now names BOTH
# subjects explicitly (never implicitly -- each grant below is its
# own visible, auditable string, and permissions.json separately
# grants each subject its own real authorization entry). Discovery
# filtering (ToolRuntime._subject_has_capability) checks for an exact
# "{subject}:" prefix match, so adding reviewer_agent's own grant
# string here does not change what writer_agent can discover, and a
# subject with neither grant (e.g. research_agent) still cannot
# discover or invoke this tool at all.
#
# Windows CRLF fix (Build Phase 11 delivery cycle): see
# document_read_tool.py's own docstring for the full explanation --
# the same "\r\n silently becomes \n on read, but size_bytes is
# computed from the real on-disk stat() size" inconsistency applied
# here too, and is fixed here the same way (`open(..., newline="")`).
# ---------------------------------------------------------------------

READ_RESEARCH_FINDINGS_TOOL_ID = "read_research_findings"

DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (".md", ".json")
DEFAULT_MAX_BYTES = 200_000


READ_RESEARCH_FINDINGS_TOOL = ToolDefinition(
    id=READ_RESEARCH_FINDINGS_TOOL_ID,
    name="Read Research Findings",
    purpose=(
        "Read the full text content of one previously-persisted "
        "research finding from the approved research-findings "
        "workspace, identified by a filename relative to the "
        "approved root."
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
        "writer_agent:research_findings:read:workspace",
        "reviewer_agent:research_findings:read:workspace",
    ),
    resource="research_findings",
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


def create_read_research_findings_executor(
    root: str | Path,
    *,
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Callable[[str], dict[str, Any]]:
    """
    Build a real executor for READ_RESEARCH_FINDINGS_TOOL, sandboxed
    to `root`.

    The returned callable has the exact signature ToolGateway.execute()
    calls with (`executor(**tool_kwargs)`, i.e.
    `executor(filename=...)` for this tool's input_schema).

    Mirrors create_document_read_executor's sandboxing exactly (see
    that function's own docstring for the full reasoning): an
    absolute `filename` is rejected before ever being joined to
    `root`, and a relative `filename` is resolved and verified to
    still be under `root` via `Path.relative_to` (also catching
    `..`-traversal and a symlink escape).

    Raises ValueError immediately (at executor-construction time) if
    `root` does not exist or is not a directory, so a misconfigured
    deployment fails loudly at startup rather than on the agent's
    first tool call.
    """

    root_path = Path(root).resolve()

    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(
            "Research findings root does not exist or is not a "
            f"directory: {root_path}"
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
                "approved research findings root; absolute paths "
                "are not allowed."
            )

        candidate = (root_path / requested).resolve()

        try:
            candidate.relative_to(root_path)
        except ValueError:
            raise PermissionError(
                f"Filename '{filename}' resolves outside the "
                f"approved research findings root ({root_path}); "
                "refusing to read it."
            ) from None

        if not candidate.exists():
            raise FileNotFoundError(
                f"Research finding not found: '{filename}'."
            )

        if not candidate.is_file():
            raise IsADirectoryError(
                f"'{filename}' is not a file."
            )

        if candidate.suffix.lower() not in normalized_extensions:
            raise ValueError(
                f"Research finding extension '{candidate.suffix}' is "
                "not approved for reading. Allowed extensions: "
                f"{normalized_extensions}."
            )

        size = candidate.stat().st_size

        if size > max_bytes:
            raise ValueError(
                f"Research finding '{filename}' is {size} bytes, "
                f"exceeding the {max_bytes}-byte limit for a single "
                "read."
            )

        try:
            with open(candidate, "r", encoding="utf-8", newline="") as f:
                content = f.read()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Research finding '{filename}' is not valid UTF-8 "
                "text."
            ) from exc

        return {
            "filename": filename,
            "content": content,
            "size_bytes": size,
        }

    return _execute
