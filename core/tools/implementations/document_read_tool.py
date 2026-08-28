from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.tools.registry.tool_registry import ToolDefinition


# ---------------------------------------------------------------------
# Real read_document tool for research_agent.
#
# RESEARCH_AGENT.md explicitly allows "approved document/file reading
# tools" as part of the agent's intentionally minimal initial toolset.
# This is a real, sandboxed implementation of that: it reads plain-text
# documents from underneath one approved root directory only, and
# nothing else on disk.
#
# There was no existing permission entry for this in permissions.json
# (only web_search and shell were present, both dating from earlier
# passes' test fixtures) -- a new
# research_agent:document:read:workspace / risk_level=LOW entry was
# added alongside this module.
#
# Windows CRLF fix (Build Phase 11 delivery cycle): a real `pytest -v`
# run on the user's Windows machine caught a genuine cross-platform
# bug this project's Linux sandbox structurally could not (the same
# class of environment-only bug as Pass 1's own #5/#9) -- `Path.
# read_text(encoding="utf-8")`'s default universal-newline mode
# silently translates on-disk "\r\n" to "\n" on every platform, so a
# multi-line file written with Windows line endings would read back
# shorter, in bytes, than `candidate.stat().st_size` (computed from
# the real on-disk size just above) actually reports -- an internal
# inconsistency between this tool's own `content` and `size_bytes`
# fields whenever a file happens to contain CRLF line endings.
# `open(..., newline="")` disables that translation entirely, so
# `content` always reflects exactly what is on disk, decoded as
# UTF-8, on every platform (a no-op on Linux/macOS, where "\n" was
# already unaffected by universal-newline translation -- which is why
# this sandbox's own tests could never have caught it).
# ---------------------------------------------------------------------

READ_DOCUMENT_TOOL_ID = "read_document"

DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (".txt", ".md")
DEFAULT_MAX_BYTES = 200_000


READ_DOCUMENT_TOOL = ToolDefinition(
    id=READ_DOCUMENT_TOOL_ID,
    name="Read Document",
    purpose=(
        "Read the full text content of one approved plain-text or "
        "Markdown document from the research workspace, identified by "
        "a path relative to the approved document root."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "size_bytes": {"type": "integer"},
        },
        "required": ["path", "content", "size_bytes"],
        "additionalProperties": False,
    },
    permissions=(
        "research_agent:document:read:workspace",
    ),
    resource="document",
    action="read",
    scope="workspace",
    risk_level="LOW",
    error_handling={
        "retryable": False,
        "on_failure": (
            "Surface the file-read error to the agent as a failed "
            "tool result. Do not retry automatically: a path that is "
            "missing, outside the approved root, of a disallowed "
            "type, or oversized will not become readable by retrying "
            "the identical request."
        ),
    },
)


def create_document_read_executor(
    root: str | Path,
    *,
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Callable[[str], dict[str, Any]]:
    """
    Build a real executor for READ_DOCUMENT_TOOL, sandboxed to `root`.

    The returned callable has the exact signature ToolGateway.execute()
    calls with (`executor(**tool_kwargs)`, i.e. `executor(path=...)`
    for this tool's input_schema).

    Every requested path is resolved against `root` and rejected if it
    would resolve to anything outside it -- this is the only thing
    standing between "read an approved research document" and "read
    anything on disk this process can see", so it is deliberately
    strict about every escape route:

    - an absolute `path` (which would otherwise silently replace
      `root` entirely under pathlib's `/` join semantics, e.g.
      `Path("root") / "/etc/passwd" == Path("/etc/passwd")`) is
      rejected outright, before it ever reaches the join;
    - a relative `path` containing `..` segments is still caught
      afterward by resolving the joined path and verifying it remains
      under `root` (`Path.relative_to`), which also catches a symlink
      inside `root` that points back out of it.

    Raises ValueError immediately (at executor-construction time) if
    `root` does not exist or is not a directory, so a misconfigured
    deployment fails loudly at startup rather than on the agent's
    first tool call.
    """

    root_path = Path(root).resolve()

    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(
            "Document root does not exist or is not a directory: "
            f"{root_path}"
        )

    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")

    normalized_extensions = tuple(
        extension.lower()
        for extension in allowed_extensions
    )

    def _execute(path: str) -> dict[str, Any]:

        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string.")

        requested = Path(path)

        if requested.is_absolute():
            raise PermissionError(
                f"Path '{path}' must be relative to the approved "
                "document root; absolute paths are not allowed."
            )

        candidate = (root_path / requested).resolve()

        try:
            candidate.relative_to(root_path)
        except ValueError:
            raise PermissionError(
                f"Path '{path}' resolves outside the approved "
                f"document root ({root_path}); refusing to read it."
            ) from None

        if not candidate.exists():
            raise FileNotFoundError(
                f"Document not found: '{path}'."
            )

        if not candidate.is_file():
            raise IsADirectoryError(
                f"'{path}' is not a file."
            )

        if candidate.suffix.lower() not in normalized_extensions:
            raise ValueError(
                f"Document extension '{candidate.suffix}' is not "
                "approved for reading. Allowed extensions: "
                f"{normalized_extensions}."
            )

        size = candidate.stat().st_size

        if size > max_bytes:
            raise ValueError(
                f"Document '{path}' is {size} bytes, exceeding the "
                f"{max_bytes}-byte limit for a single read. Split it "
                "into smaller approved documents."
            )

        try:
            with open(candidate, "r", encoding="utf-8", newline="") as f:
                content = f.read()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Document '{path}' is not valid UTF-8 text."
            ) from exc

        return {
            "path": path,
            "content": content,
            "size_bytes": size,
        }

    return _execute
