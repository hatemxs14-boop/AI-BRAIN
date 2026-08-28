from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.tools.registry.tool_registry import ToolDefinition


# ---------------------------------------------------------------------
# Real write_report tool for writer_agent (Build Phase 8).
#
# WRITER_AGENT.md's "Memory Access" section allows the agent to
# "publish a written report when explicitly authorized" -- the write
# side of writer_agent's whole purpose, and structurally the same
# security posture as research_agent's write_research_findings tool
# (Build Phase 3), for the same reason: persisting a report is the
# one action this agent can take that leaves a durable, externally-
# visible trace outside its own execution. This ToolDefinition
# honestly declares risk_level=HIGH even though RiskEngine's raw
# keyword heuristic would only assess a plain write-to-workspace as
# MEDIUM (see risk_engine.py's "File writes" branch); the matching
# permissions.json entry also declares HIGH/policy, so
# AuthorizationEngine's effective-risk floor (Pass 2 finding A) raises
# the real decision to REQUIRE_APPROVAL/"policy" regardless of the raw
# MEDIUM assessment -- the same "explicitly authorized" gate write_
# research_findings_tool.py already established, reused deliberately
# rather than reinvented.
# ---------------------------------------------------------------------

WRITE_REPORT_TOOL_ID = "write_report"

DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (".md",)
DEFAULT_MAX_BYTES = 200_000


WRITE_REPORT_TOOL = ToolDefinition(
    id=WRITE_REPORT_TOOL_ID,
    name="Write Report",
    purpose=(
        "Persist one written report as a new file under the approved "
        "reports workspace, identified by a filename relative to the "
        "approved root. Requires explicit approval before executing "
        "(see this module's own docstring for why): every call "
        "returns APPROVAL_REQUIRED unless the caller supplies an "
        "explicit, attributed approval."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
            },
            "content": {
                "type": "string",
            },
        },
        "required": ["filename", "content"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "size_bytes": {"type": "integer"},
        },
        "required": ["path", "size_bytes"],
        "additionalProperties": False,
    },
    permissions=(
        "writer_agent:report:write:workspace",
    ),
    resource="report",
    action="write",
    scope="workspace",
    risk_level="HIGH",
    error_handling={
        "retryable": False,
        "on_failure": (
            "Surface the write error to the agent as a failed tool "
            "result. Do not retry automatically: a filename that "
            "already exists, is outside the approved root, uses a "
            "disallowed extension, or is oversized will not become "
            "writable by retrying the identical request."
        ),
    },
)


def create_write_report_executor(
    root: str | Path,
    *,
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Callable[[str, str], dict[str, Any]]:
    """
    Build a real executor for WRITE_REPORT_TOOL, sandboxed to `root`.

    The returned callable has the exact signature ToolGateway.execute()
    calls with (`executor(**tool_kwargs)`, i.e.
    `executor(filename=..., content=...)` for this tool's
    input_schema).

    Mirrors create_write_research_findings_executor's sandboxing and
    write-once behavior exactly (see that function's own docstring for
    the full reasoning): an absolute `filename` is rejected before
    ever being joined to `root`; a relative `filename` is resolved and
    verified to still be under `root` via `Path.relative_to` (also
    catching `..`-traversal and a symlink escape); and a `filename`
    that already exists under `root` is rejected outright -- this tool
    only ever creates a new report, never silently overwrites a prior
    one. A caller that genuinely wants to revise a report should
    choose a new filename, keeping the prior version intact.

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

    def _execute(filename: str, content: str) -> dict[str, Any]:

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string.")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string.")

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
                "write it."
            ) from None

        if candidate.suffix.lower() not in normalized_extensions:
            raise ValueError(
                f"Filename extension '{candidate.suffix}' is not "
                "approved for reports. Allowed extensions: "
                f"{normalized_extensions}."
            )

        encoded = content.encode("utf-8")

        if len(encoded) > max_bytes:
            raise ValueError(
                f"Report content is {len(encoded)} bytes, exceeding "
                f"the {max_bytes}-byte limit for a single write. "
                "Split it into smaller, separately-approved reports."
            )

        if candidate.exists():
            raise FileExistsError(
                f"Report file '{filename}' already exists; refusing "
                "to overwrite a prior report. Choose a different "
                "filename to record a revision."
            )

        candidate.parent.mkdir(parents=True, exist_ok=True)

        candidate.write_text(content, encoding="utf-8")

        return {
            "path": filename,
            "size_bytes": len(encoded),
        }

    return _execute
