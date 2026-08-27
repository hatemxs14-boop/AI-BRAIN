from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.tools.engine.tool_gateway import ToolExecutionResult
from core.tools.runtime.tool_invocation import ToolInvocation
from core.tools.runtime.tool_runtime import (
    ToolDiscovery,
    ToolRuntime,
)


@dataclass(frozen=True)
class AgentToolInterface:
    """
    Controlled interface between an Agent and the Tool Runtime.

    The Agent interacts with tools only through this interface.

    The interface provides:

    - capability-aware tool discovery
    - safe tool lookup
    - ToolInvocation construction
    - execution through ToolRuntime

    The Agent never receives direct access to:

    - ToolRegistry
    - ToolGateway
    - SecurityDecisionPoint
    - executor callables
    """

    runtime: ToolRuntime

    def discover_tools(
        self,
        subject: str,
    ) -> tuple[ToolDiscovery, ...]:
        """
        Discover only tools available to the Agent subject.
        """

        return self.runtime.discover_tools_for_subject(
            subject
        )

    def discover_tool(
        self,
        *,
        subject: str,
        tool_id: str,
    ) -> ToolDiscovery | None:
        """
        Discover a specific tool only when the Agent has
        the corresponding capability.

        Returns None when the tool is not available to
        the supplied subject.
        """

        tools = self.runtime.discover_tools_for_subject(
            subject
        )

        for tool in tools:
            if tool.id == tool_id:
                return tool

        return None

    def create_invocation(
        self,
        *,
        subject: str,
        tool_id: str,
        inputs: dict[str, Any],
        approved: bool | None = None,
        approved_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolInvocation:
        """
        Create a standardized ToolInvocation.

        This method does not execute anything.
        """

        tool = self.discover_tool(
            subject=subject,
            tool_id=tool_id,
        )

        if tool is None:
            raise PermissionError(
                f"Tool '{tool_id}' is not available "
                f"to subject '{subject}'."
            )

        return ToolInvocation(
            subject=subject,
            tool_id=tool_id,
            inputs=inputs,
            approved=approved,
            approved_by=approved_by,
            metadata=metadata,
        )

    def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolExecutionResult:
        """
        Execute a ToolInvocation through the Tool Runtime.
        """

        if not isinstance(
            invocation,
            ToolInvocation,
        ):
            raise TypeError(
                "invocation must be a ToolInvocation."
            )

        return self.runtime.execute(
            invocation
        )

    def invoke(
        self,
        *,
        subject: str,
        tool_id: str,
        inputs: dict[str, Any],
        approved: bool | None = None,
        approved_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        """
        Convenience method that performs:

        Discovery
            ↓
        Capability Check
            ↓
        ToolInvocation
            ↓
        ToolRuntime
            ↓
        ToolGateway
        """

        invocation = self.create_invocation(
            subject=subject,
            tool_id=tool_id,
            inputs=inputs,
            approved=approved,
            approved_by=approved_by,
            metadata=metadata,
        )

        return self.execute(invocation)