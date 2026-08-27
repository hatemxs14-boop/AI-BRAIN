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