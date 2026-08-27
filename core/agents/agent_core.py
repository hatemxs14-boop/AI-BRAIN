from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.tool_interface import (
    AgentToolInterface,
)

from core.tools.engine.tool_gateway import (
    ToolExecutionResult,
)

from core.tools.runtime.tool_invocation import (
    ToolInvocation,
)

from core.tools.runtime.tool_runtime import (
    ToolDiscovery,
)


@dataclass(frozen=True)
class AgentIdentity:
    """
    Immutable identity of an AI-BRAIN Agent.
    """

    subject: str
    name: str
    purpose: str

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str):
            raise TypeError(
                "subject must be a string."
            )

        if not self.subject.strip():
            raise ValueError(
                "subject must not be empty."
            )

        if not isinstance(self.name, str):
            raise TypeError(
                "name must be a string."
            )

        if not self.name.strip():
            raise ValueError(
                "name must not be empty."
            )

        if not isinstance(self.purpose, str):
            raise TypeError(
                "purpose must be a string."
            )

        if not self.purpose.strip():
            raise ValueError(
                "purpose must not be empty."
            )


@dataclass
class AgentState:
    """
    Mutable operational state of an Agent.

    The state contains no executor references and no direct
    Security Layer references.
    """

    task: str | None = None
    status: str = "IDLE"

    last_tool_id: str | None = None
    last_result: ToolExecutionResult | None = None

    history: list[ToolExecutionResult] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class AgentCore:
    """
    Secure operational core of an AI-BRAIN Agent.

    The Agent Core sits between the reasoning layer and the
    Tool Runtime.

    The Agent Core does not:

    - access ToolRegistry directly
    - access ToolGateway directly
    - access SecurityDecisionPoint directly
    - access executor callables
    - grant permissions
    - modify security decisions

    Tool operations are performed exclusively through
    AgentToolInterface.
    """

    def __init__(
        self,
        *,
        identity: AgentIdentity,
        tools: AgentToolInterface,
    ) -> None:
        self.identity = identity
        self.tools = tools
        self.state = AgentState()

    def start_task(
        self,
        task: str,
    ) -> None:
        """
        Start a new Agent task.
        """

        if not isinstance(task, str):
            raise TypeError(
                "task must be a string."
            )

        if not task.strip():
            raise ValueError(
                "task must not be empty."
            )

        self.state.task = task
        self.state.status = "RUNNING"

    def complete_task(self) -> None:
        """
        Mark the current task as completed.
        """

        self.state.status = "COMPLETED"

    def fail_task(self) -> None:
        """
        Mark the current task as failed.
        """

        self.state.status = "FAILED"

    def await_approval(self) -> None:
        """
        Mark the current task as paused pending additional approval.

        This is deliberately distinct from `fail_task()`: a HIGH/
        CRITICAL-risk tool call that comes back APPROVAL_REQUIRED is
        not a failure -- it's a recoverable pause. The task can be
        resumed (with an explicit approval decision) once approval is
        obtained. Reusing FAILED for this case made an in-progress,
        resumable task indistinguishable from a genuinely dead one to
        anything reading `agent.state.status` (e.g.
        `get_state_snapshot()`), which the richer, one-shot
        `AgentLoopResult.status` value has always distinguished.
        """

        self.state.status = "AWAITING_APPROVAL"

    def discover_tools(
        self,
    ) -> tuple[ToolDiscovery, ...]:
        """
        Discover tools available to this Agent.
        """

        return self.tools.discover_tools(
            self.identity.subject
        )

    def discover_tool(
        self,
        tool_id: str,
    ) -> ToolDiscovery | None:
        """
        Discover one tool if this Agent has access to it.
        """

        return self.tools.discover_tool(
            subject=self.identity.subject,
            tool_id=tool_id,
        )

    def create_tool_invocation(
        self,
        *,
        tool_id: str,
        inputs: dict[str, Any],
        approved: bool | None = None,
        approved_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolInvocation:
        """
        Create a ToolInvocation through the controlled
        AgentToolInterface.
        """

        return self.tools.create_invocation(
            subject=self.identity.subject,
            tool_id=tool_id,
            inputs=inputs,
            approved=approved,
            approved_by=approved_by,
            metadata=metadata,
        )

    def execute_tool(
        self,
        invocation: ToolInvocation,
    ) -> ToolExecutionResult:
        """
        Execute a ToolInvocation through the controlled
        AgentToolInterface.
        """

        result = self.tools.execute(
            invocation
        )

        self.state.last_tool_id = (
            invocation.tool_id
        )

        self.state.last_result = result

        self.state.history.append(
            result
        )

        return result

    def invoke_tool(
        self,
        *,
        tool_id: str,
        inputs: dict[str, Any],
        approved: bool | None = None,
        approved_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        """
        Create and execute a ToolInvocation.
        """

        invocation = self.create_tool_invocation(
            tool_id=tool_id,
            inputs=inputs,
            approved=approved,
            approved_by=approved_by,
            metadata=metadata,
        )

        return self.execute_tool(
            invocation
        )

    def execute_action(
        self,
        action: AgentAction,
    ) -> ToolExecutionResult | None:
        """
        Execute an AgentAction.

        INVOKE_TOOL:
            Creates and executes a ToolInvocation.

        COMPLETE:
            Completes the current task.

        FAIL:
            Marks the current task as failed.

        Unknown actions fail closed.
        """

        if not isinstance(action, AgentAction):
            raise TypeError(
                "action must be an AgentAction."
            )

        if (
            action.action_type
            == AgentActionType.INVOKE_TOOL
        ):
            result = self.invoke_tool(
                tool_id=action.tool_id,
                inputs=action.inputs,
                approved=action.approved,
                approved_by=action.approved_by,
                metadata=action.metadata,
            )

            return result

        if (
            action.action_type
            == AgentActionType.COMPLETE
        ):
            self.complete_task()
            return None

        if (
            action.action_type
            == AgentActionType.FAIL
        ):
            self.fail_task()
            return None

        raise ValueError(
            "Unknown AgentActionType; execution blocked."
        )

    def get_state_snapshot(
        self,
    ) -> dict[str, Any]:
        """
        Return a safe snapshot of the current Agent state.

        Executor and Security Layer internals are not exposed.
        """

        return {
            "subject": self.identity.subject,
            "name": self.identity.name,
            "purpose": self.identity.purpose,
            "task": self.state.task,
            "status": self.state.status,
            "last_tool_id": self.state.last_tool_id,
            "history_length": len(
                self.state.history
            ),
            "metadata": dict(
                self.state.metadata
            ),
        }