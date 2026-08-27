from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.agents.action_validator import (
    AgentActionValidator,
)

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_context import (
    AgentContext,
)

from core.agents.agent_core import (
    AgentCore,
)

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.tools.engine.tool_gateway import (
    ToolExecutionResult,
)


@dataclass(frozen=True)
class AgentLoopResult:
    """
    Final result of an Agent execution loop.
    """

    status: str
    steps: int
    last_result: ToolExecutionResult | None
    reason: str | None
    context: AgentContext


class AgentExecutionLoop:
    """
    Controlled execution loop for an AI-BRAIN Agent.

    The loop:

        AgentContext
            ↓
        Decision Engine
            ↓
        AgentAction
            ↓
        AgentActionValidator
            ↓
        AgentCore
            ↓
        AgentToolInterface
            ↓
        ToolRuntime
            ↓
        ToolGateway
            ↓
        Security Layer
            ↓
        Private Executor

    The loop never:

    - executes tools directly
    - accesses executors
    - accesses the Security Layer directly
    - grants permissions
    - bypasses AgentCore
    """

    def __init__(
        self,
        *,
        agent: AgentCore,
        decision_engine: AgentDecisionEngine | None = None,
        action_provider: Callable[
            [AgentCore],
            AgentAction,
        ] | None = None,
        max_steps: int = 10,
        action_validator: AgentActionValidator | None = None,
    ) -> None:

        if not isinstance(
            agent,
            AgentCore,
        ):
            raise TypeError(
                "agent must be an AgentCore."
            )

        if not isinstance(
            max_steps,
            int,
        ):
            raise TypeError(
                "max_steps must be an integer."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        if decision_engine is None:

            if action_provider is None:
                raise ValueError(
                    "Either decision_engine or "
                    "action_provider must be provided."
                )

            if not callable(action_provider):
                raise TypeError(
                    "action_provider must be callable."
                )

            self.decision_engine = None
            self.action_provider = action_provider

        else:

            if not isinstance(
                decision_engine,
                AgentDecisionEngine,
            ):
                raise TypeError(
                    "decision_engine must implement "
                    "AgentDecisionEngine."
                )

            self.decision_engine = decision_engine
            self.action_provider = None

        self.agent = agent
        self.max_steps = max_steps

        self.action_validator = (
            action_validator
            if action_validator is not None
            else AgentActionValidator()
        )

        self.context: AgentContext | None = None

    def run(
        self,
    ) -> AgentLoopResult:
        """
        Run the Agent execution loop.
        """

        task = getattr(
            self.agent.state,
            "task",
            None,
        )

        if not isinstance(
            task,
            str,
        ) or not task.strip():

            raise ValueError(
                "Agent must have an active task before execution."
            )

        self.context = AgentContext(
            task=task,
        )

        self._refresh_available_tools()

        steps = 0

        last_result: ToolExecutionResult | None = None

        while steps < self.max_steps:

            self._refresh_available_tools()

            try:

                action = self._get_next_action()

            except Exception as exc:

                self.agent.fail_task()

                if self.action_provider is not None:

                    return AgentLoopResult(
                        status="INVALID_ACTION",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Action provider failed to produce "
                            f"a valid AgentAction: {exc}"
                        ),
                        context=self.context,
                    )

                return AgentLoopResult(
                    status="DECISION_ERROR",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        f"Decision engine failed: {exc}"
                    ),
                    context=self.context,
                )

            if not isinstance(
                action,
                AgentAction,
            ):

                self.agent.fail_task()

                return AgentLoopResult(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "Decision engine did not return "
                        "an AgentAction."
                    ),
                    context=self.context,
                )

            try:

                validation = (
                    self.action_validator.validate(
                        action
                    )
                )

            except Exception as exc:

                self.agent.fail_task()

                return AgentLoopResult(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "AgentActionValidator raised an "
                        f"exception: {exc}"
                    ),
                    context=self.context,
                )

            if not validation.valid:

                self.agent.fail_task()

                validation_details = getattr(
                    validation,
                    "errors",
                    None,
                )

                return AgentLoopResult(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "Agent produced an invalid action. "
                        f"errors={validation_details!r}; "
                        f"action={action!r}"
                    ),
                    context=self.context,
                )

            steps += 1

            self.context.record_step()

            if (
                action.action_type
                == AgentActionType.COMPLETE
            ):

                self.agent.execute_action(
                    action
                )

                return AgentLoopResult(
                    status="COMPLETED",
                    steps=steps,
                    last_result=last_result,
                    reason=action.reason,
                    context=self.context,
                )

            if (
                action.action_type
                == AgentActionType.FAIL
            ):

                self.agent.execute_action(
                    action
                )

                return AgentLoopResult(
                    status="FAILED",
                    steps=steps,
                    last_result=last_result,
                    reason=action.reason,
                    context=self.context,
                )

            if (
                action.action_type
                == AgentActionType.INVOKE_TOOL
            ):

                try:

                    last_result = (
                        self.agent.execute_action(
                            action
                        )
                    )

                except Exception as exc:

                    self.agent.fail_task()

                    return AgentLoopResult(
                        status="FAILED",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Tool action execution failed: "
                            f"{exc}"
                        ),
                        context=self.context,
                    )

                self.context.record_tool_result(
                    last_result
                )

                self._refresh_available_tools()

                continue

            self.agent.fail_task()

            return AgentLoopResult(
                status="INVALID_ACTION",
                steps=steps,
                last_result=last_result,
                reason=(
                    "Unsupported AgentActionType."
                ),
                context=self.context,
            )

        self.agent.fail_task()

        return AgentLoopResult(
            status="MAX_STEPS_EXCEEDED",
            steps=steps,
            last_result=last_result,
            reason=(
                "Agent execution stopped because "
                "max_steps was exceeded."
            ),
            context=self.context,
        )

    def _refresh_available_tools(
        self,
    ) -> None:
        """
        Refresh the tools exposed to the decision engine.

        Discovery is performed through AgentToolInterface.

        This does not authorize execution.
        """

        if self.context is None:
            raise RuntimeError(
                "AgentContext has not been initialized."
            )

        subject = self.agent.identity.subject

        discoveries = (
            self.agent.tools.discover_tools(
                subject
            )
        )

        available_tools = []

        for tool in discoveries:

            available_tools.append(
                {
                    "id": tool.id,
                    "name": tool.name,
                    "purpose": tool.purpose,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "resource": tool.resource,
                    "action": tool.action,
                    "scope": tool.scope,
                    "risk_level": tool.risk_level,
                }
            )

        self.context.set_metadata(
            "available_tools",
            available_tools,
        )

        self.context.set_metadata(
            "available_tool_ids",
            tuple(
                tool["id"]
                for tool in available_tools
            ),
        )

    def _get_next_action(
        self,
    ) -> AgentAction:

        if self.context is None:
            raise RuntimeError(
                "AgentContext has not been initialized."
            )

        if self.decision_engine is not None:

            return self.decision_engine.decide(
                self.context
            )

        if self.action_provider is None:
            raise RuntimeError(
                "No action provider is configured."
            )

        return self.action_provider(
            self.agent
        )