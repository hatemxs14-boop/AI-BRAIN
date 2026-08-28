from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.tools.registry.tool_registry import ToolDefinition


# ---------------------------------------------------------------------
# Real write_research_findings tool for research_agent.
#
# RESEARCH_AGENT.md's "Memory Access" section allows the agent to
# "write research findings when explicitly authorized" -- until this
# build phase, that was the one line of the spec with no real tool
# behind it at all: research_agent could search, read documents, and
# read webpages, but had no way to actually persist a finding anywhere.
#
# The "when explicitly authorized" phrase is not decorative -- it is
# encoded directly into this tool's security posture, not left to the
# agent's own judgment. Every prior tool in this project (web_search,
# read_document, read_webpage) is LOW risk_level and auto-executes: a
# read has no lasting effect, so nothing about it needs a human or
# policy checkpoint. Writing a persisted "research finding" is
# different -- it is the one action this agent can take that leaves a
# durable trace outside its own execution -- so this tool's
# ToolDefinition honestly declares risk_level=HIGH even though
# RiskEngine's raw keyword heuristic would only assess a plain write-
# to-workspace as MEDIUM (see risk_engine.py's "File writes" branch).
# The matching permissions.json entry also declares HIGH/policy, so
# AuthorizationEngine's effective-risk floor (Pass 2 finding A: this
# project's own regression-tested "conservative-risk permission" fix)
# raises the real decision to REQUIRE_APPROVAL/"policy" regardless of
# the raw MEDIUM assessment -- exactly the "explicitly authorized"
# gate the spec calls for, enforced by the security stack itself
# rather than by convention. This is the first REAL (non-test) tool in
# this project to exercise that approval-required path in production
# use, rather than only in test fixtures.
# ---------------------------------------------------------------------

WRITE_RESEARCH_FINDINGS_TOOL_ID = "write_research_findings"

DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (".md", ".json")
DEFAULT_MAX_BYTES = 200_000


WRITE_RESEARCH_FINDINGS_TOOL = ToolDefinition(
    id=WRITE_RESEARCH_FINDINGS_TOOL_ID,
    name="Write Research Findings",
    purpose=(
        "Persist one research finding as a new file under the "
        "approved research-findings workspace, identified by a "
        "filename relative to the approved root. Requires explicit "
        "approval before executing (see this module's own docstring "
        "for why): every call returns APPROVAL_REQUIRED unless the "
        "caller supplies an explicit, attributed approval."
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
        "research_agent:research_findings:write:workspace",
    ),
    resource="research_findings",
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


def create_write_research_findings_executor(
    root: str | Path,
    *,
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Callable[[str, str], dict[str, Any]]:
    """
    Build a real executor for WRITE_RESEARCH_FINDINGS_TOOL, sandboxed
    to `root`.

    The returned callable has the exact signature ToolGateway.execute()
    calls with (`executor(**tool_kwargs)`, i.e.
    `executor(filename=..., content=...)` for this tool's
    input_schema).

    Path-sandboxing mirrors create_document_read_executor exactly (see
    that function's docstring for the full reasoning): an absolute
    `filename` is rejected before ever being joined to `root`, and a
    relative `filename` is resolved and verified to still be under
    `root` via `Path.relative_to` (also catching `..`-traversal and a
    symlink escape). Additionally, and unlike the read tool, a
    `filename` that already exists under `root` is rejected outright
    -- this tool only ever creates a new file, never silently
    overwrites a prior finding. A caller that genuinely wants to
    revise a finding should choose a new filename, keeping the prior
    version intact as part of the research trail.

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

    def _execute(filename: str, content: str) -> dict[str, Any]:

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string.")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string.")

        requested = Path(filename)

        if requested.is_absolute():
            raise PermissionError(
                f"Filename '{filename}' must be relative to the "
                "approved research findings root; absolute paths are "
                "not allowed."
            )

        candidate = (root_path / requested).resolve()

        try:
            candidate.relative_to(root_path)
        except ValueError:
            raise PermissionError(
                f"Filename '{filename}' resolves outside the "
                f"approved research findings root ({root_path}); "
                "refusing to write it."
            ) from None

        if candidate.suffix.lower() not in normalized_extensions:
            raise ValueError(
                f"Filename extension '{candidate.suffix}' is not "
                "approved for research findings. Allowed extensions: "
                f"{normalized_extensions}."
            )

        encoded = content.encode("utf-8")

        if len(encoded) > max_bytes:
            raise ValueError(
                f"Research finding content is {len(encoded)} bytes, "
                f"exceeding the {max_bytes}-byte limit for a single "
                "write. Split it into smaller, separately-approved "
                "findings."
            )

        if candidate.exists():
            raise FileExistsError(
                f"Research findings file '{filename}' already "
                "exists; refusing to overwrite prior research. "
                "Choose a different filename to record a revision."
            )

        candidate.parent.mkdir(parents=True, exist_ok=True)

        candidate.write_text(content, encoding="utf-8")

        return {
            "path": filename,
            "size_bytes": len(encoded),
        }

    return _execute
