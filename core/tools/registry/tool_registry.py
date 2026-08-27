from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    """
    Public definition of a tool available to AI-BRAIN.

    This object intentionally contains NO executable callable.

    Agents may inspect this definition, but they cannot obtain
    an executor from it.
    """

    id: str
    name: str
    purpose: str

    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    permissions: tuple[str, ...]

    resource: str
    action: str
    scope: str

    risk_level: str

    # Required by TOOL_SPEC.md ("Every Tool Must Define ...
    # error_handling") and its own rule "Tools must never hide
    # important failures". A tool author must say explicitly whether a
    # failed call is safe to retry and what should happen when it
    # isn't -- this is the contract a future retry/timeout layer will
    # read, and forcing it at registration time means a tool can never
    # be silently registered without anyone having thought about its
    # failure behavior.
    #
    # Required keys:
    #   retryable   (bool)  -- is retrying this tool call ever safe?
    #   on_failure  (str)   -- what should happen / be surfaced when
    #                          this tool fails (non-empty).
    # Optional key:
    #   max_retries (int >= 0) -- required, and only meaningful, when
    #                             retryable is True.
    error_handling: dict[str, Any]


class ToolRegistry:
    """
    Central registry for officially available tools.

    The registry stores only trusted tool definitions.

    It does NOT store executable callables.

    Responsibilities:

    - register tool definitions
    - validate definitions
    - prevent duplicate IDs
    - retrieve public definitions
    - list public definitions

    Execution is handled exclusively by the Tool Gateway.
    """

    VALID_RISK_LEVELS = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """
        Register a public tool definition.
        """

        self._validate_tool(tool)

        if tool.id in self._tools:
            raise ValueError(
                f"Tool with id '{tool.id}' is already registered."
            )

        self._tools[tool.id] = tool

    def get(self, tool_id: str) -> ToolDefinition:
        """
        Retrieve the public definition of a registered tool.
        """

        if not isinstance(tool_id, str):
            raise TypeError("tool_id must be a string.")

        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown tool id: '{tool_id}'."
            ) from exc

    def contains(self, tool_id: str) -> bool:
        """
        Return True when the tool is registered.
        """

        return tool_id in self._tools

    def list_tools(self) -> tuple[ToolDefinition, ...]:
        """
        Return all registered public tool definitions.
        """

        return tuple(self._tools.values())

    def _validate_tool(self, tool: ToolDefinition) -> None:
        """
        Validate the complete public tool contract.
        """

        if not isinstance(tool, ToolDefinition):
            raise TypeError(
                "Only ToolDefinition instances can be registered."
            )

        if not isinstance(tool.id, str) or not tool.id.strip():
            raise ValueError("Tool id must not be empty.")

        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ValueError("Tool name must not be empty.")

        if not isinstance(tool.purpose, str) or not tool.purpose.strip():
            raise ValueError("Tool purpose must not be empty.")

        if not isinstance(tool.input_schema, dict):
            raise TypeError("input_schema must be a dictionary.")

        if not isinstance(tool.output_schema, dict):
            raise TypeError("output_schema must be a dictionary.")

        if not isinstance(tool.permissions, tuple):
            raise TypeError(
                "permissions must be a tuple of permission definitions."
            )

        if not tool.permissions:
            raise ValueError(
                f"Tool '{tool.id}' must define at least one permission."
            )

        if any(
            not isinstance(permission, str) or not permission.strip()
            for permission in tool.permissions
        ):
            raise ValueError(
                f"Tool '{tool.id}' contains an invalid permission."
            )

        if not isinstance(tool.resource, str) or not tool.resource.strip():
            raise ValueError("Tool resource must not be empty.")

        if not isinstance(tool.action, str) or not tool.action.strip():
            raise ValueError("Tool action must not be empty.")

        if not isinstance(tool.scope, str) or not tool.scope.strip():
            raise ValueError("Tool scope must not be empty.")

        if not isinstance(tool.risk_level, str):
            raise TypeError("Tool risk_level must be a string.")

        if tool.risk_level not in self.VALID_RISK_LEVELS:
            raise ValueError(
                f"Invalid risk level '{tool.risk_level}' "
                f"for tool '{tool.id}'. "
                f"Expected one of: "
                f"{sorted(self.VALID_RISK_LEVELS)}."
            )

        # ---------------------------------------------------------
        # Permission strings must actually describe THIS tool.
        #
        # ToolDefinition.permissions (subject:resource:action:scope)
        # is a separate, advisory system used only for discovery
        # filtering (ToolRuntime._subject_has_capability) -- the real
        # authorization decision is made independently by
        # AuthorizationEngine against permissions.json using
        # tool.resource/action/scope directly. Nothing previously
        # checked that the two agreed, so a copy-paste or typo in a
        # permission string could make a tool wrongly appear
        # "available" to a subject who isn't really authorized to run
        # it (its real, dangerous resource/action/scope still fully
        # exposed via discovery), or wrongly hidden from a subject who
        # genuinely is. Failing loudly here, at registration, is far
        # cheaper than that silently-wrong metadata surfacing later.
        # ---------------------------------------------------------

        for permission in tool.permissions:
            segments = permission.split(":")

            if len(segments) != 4:
                raise ValueError(
                    f"Tool '{tool.id}' has a malformed permission "
                    f"'{permission}': expected exactly 4 "
                    "colon-separated segments "
                    "(subject:resource:action:scope)."
                )

            _subject, resource, action, scope = segments

            if (resource, action, scope) != (
                tool.resource,
                tool.action,
                tool.scope,
            ):
                raise ValueError(
                    f"Tool '{tool.id}' declares permission "
                    f"'{permission}' whose resource/action/scope "
                    f"('{resource}:{action}:{scope}') does not match "
                    "the tool's own registered resource/action/scope "
                    f"('{tool.resource}:{tool.action}:{tool.scope}'). "
                    "A tool's discovery permissions must describe the "
                    "same capability the tool actually performs."
                )

        # ---------------------------------------------------------
        # error_handling contract (TOOL_SPEC.md).
        # ---------------------------------------------------------

        if not isinstance(tool.error_handling, dict):
            raise TypeError(
                f"Tool '{tool.id}' error_handling must be a dictionary."
            )

        retryable = tool.error_handling.get("retryable")

        if not isinstance(retryable, bool):
            raise ValueError(
                f"Tool '{tool.id}' error_handling must define a "
                "boolean 'retryable'."
            )

        on_failure = tool.error_handling.get("on_failure")

        if not isinstance(on_failure, str) or not on_failure.strip():
            raise ValueError(
                f"Tool '{tool.id}' error_handling must define a "
                "non-empty string 'on_failure' describing what "
                "happens when the tool fails."
            )

        if retryable:
            max_retries = tool.error_handling.get("max_retries")

            if (
                not isinstance(max_retries, int)
                or isinstance(max_retries, bool)
                or max_retries < 0
            ):
                raise ValueError(
                    f"Tool '{tool.id}' is marked retryable and must "
                    "define a non-negative integer 'max_retries'."
                )