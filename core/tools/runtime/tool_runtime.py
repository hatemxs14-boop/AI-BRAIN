from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.tools.engine.tool_gateway import (
    ToolExecutionResult,
    ToolGateway,
)

from core.tools.registry.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)

from core.tools.runtime.tool_invocation import (
    ToolInvocation,
)


@dataclass(frozen=True)
class ToolDiscovery:
    """
    Safe public description of a registered tool.

    Executor information is intentionally excluded.
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


class ToolRuntime:
    """
    Operational runtime for AI-BRAIN tools.

    The Tool Runtime is not a security boundary.

    Responsibilities:

    - receive ToolInvocation requests
    - expose safe tool discovery information
    - filter discovered tools by explicit capabilities
    - verify tool existence
    - delegate execution to the Tool Gateway

    The Runtime never:

    - executes executors directly
    - grants permissions
    - changes risk levels
    - performs authorization
    - bypasses the Tool Gateway
    - exposes executor callables through discovery
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        gateway: ToolGateway,
    ) -> None:
        self.registry = registry
        self.gateway = gateway

    def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolExecutionResult:
        """
        Execute a ToolInvocation through the Tool Gateway.
        """

        if not isinstance(invocation, ToolInvocation):
            raise TypeError(
                "invocation must be a ToolInvocation."
            )

        return self.gateway.execute(
            subject=invocation.subject,
            tool_id=invocation.tool_id,
            tool_kwargs=invocation.inputs,
            approved=invocation.approved,
            approved_by=invocation.approved_by,
        )

    def has_tool(
        self,
        tool_id: str,
    ) -> bool:
        """
        Return True when the tool exists in the registry.
        """

        return self.registry.contains(tool_id)

    def get_tool(
        self,
        tool_id: str,
    ) -> ToolDefinition:
        """
        Return the registered tool definition.

        This method is intended for internal runtime use.
        Agents should use discover_tool() instead.
        """

        return self.registry.get(tool_id)

    def list_tools(
        self,
    ) -> tuple[ToolDefinition, ...]:
        """
        Return registered tool definitions.

        This method is intended for internal runtime use.
        """

        return self.registry.list_tools()

    def discover_tool(
        self,
        tool_id: str,
    ) -> ToolDiscovery:
        """
        Return a safe description of one registered tool.

        Executor information is never included.
        """

        tool = self.registry.get(tool_id)

        return self._to_discovery(tool)

    def discover_tools(
        self,
    ) -> tuple[ToolDiscovery, ...]:
        """
        Return safe descriptions of all registered tools.

        Executor information is never exposed.
        """

        return tuple(
            self._to_discovery(tool)
            for tool in self.registry.list_tools()
        )

    def discover_tools_for_subject(
        self,
        subject: str,
    ) -> tuple[ToolDiscovery, ...]:
        """
        Return only tools whose registered permission contract
        explicitly matches the supplied subject.

        This is capability filtering for discovery.

        It is not the final authorization decision.

        The Tool Gateway and Security Layer remain authoritative
        during actual execution.
        """

        if not isinstance(subject, str):
            raise TypeError(
                "subject must be a string."
            )

        if not subject.strip():
            raise ValueError(
                "subject must not be empty."
            )

        discoveries: list[ToolDiscovery] = []

        for tool in self.registry.list_tools():
            if self._subject_has_capability(
                subject=subject,
                tool=tool,
            ):
                discoveries.append(
                    self._to_discovery(tool)
                )

        return tuple(discoveries)

    @staticmethod
    def _subject_has_capability(
        *,
        subject: str,
        tool: ToolDefinition,
    ) -> bool:
        """
        Return True when the ToolDefinition contains at least
        one explicit permission for the supplied subject.

        Permission format:

            subject:resource:action:scope
        """

        prefix = f"{subject}:"

        return any(
            permission.startswith(prefix)
            for permission in tool.permissions
        )

    @staticmethod
    def _to_discovery(
        tool: ToolDefinition,
    ) -> ToolDiscovery:
        """
        Convert an internal ToolDefinition into a safe
        Agent-facing discovery object.

        The executor is intentionally excluded.
        """

        return ToolDiscovery(
            id=tool.id,
            name=tool.name,
            purpose=tool.purpose,
            input_schema=tool.input_schema,
            output_schema=tool.output_schema,
            permissions=tool.permissions,
            resource=tool.resource,
            action=tool.action,
            scope=tool.scope,
            risk_level=tool.risk_level,
        )