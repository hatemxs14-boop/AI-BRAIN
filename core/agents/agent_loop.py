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

from core.llm.token_usage import (
    TokenUsage,
)

from core.tools.engine.tool_gateway import (
    ToolExecutionResult,
)


@dataclass(frozen=True)
class AgentLoopResult:
    """
    Final result of an Agent execution loop.

    `token_usage` (Build Phase 19) is the real, normalized token cost
    of every LLM call this one loop run made (a single run can call
    `decide()` more than once before COMPLETE/FAIL) -- see
    AgentExecutionLoop._build_result's own docstring for exactly how
    it is read, and TokenUsage's own docstring for what "real" means
    here. `None` when the decision engine in use doesn't expose usage
    at all (e.g. DeterministicDecisionEngine, a test double, or an
    `action_provider`-driven loop with no decision engine) -- never a
    fabricated zero.
    """

    status: str

    steps: int

    last_result: ToolExecutionResult | None

    reason: str | None

    context: AgentContext

    token_usage: TokenUsage | None = None


class AgentExecutionLoop:
    """
    Controlled execution loop for an AI-BRAIN Agent.

    Responsibilities:

    - obtain the current AgentContext
    - discover available tools
    - request the next AgentAction
    - validate the AgentAction
    - execute valid actions through AgentCore
    - record tool results
    - stop on COMPLETE
    - stop on FAIL
    - handle decision errors
    - handle validation errors
    - handle tool execution errors
    - handle approval-required results
    - enforce max_steps

    The loop does not:

    - execute tools directly
    - access executors
    - access the Security Layer directly
    - grant permissions
    - approve security requests
    - bypass AgentCore
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

                    return self._build_result(
                        status="INVALID_ACTION",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Action provider failed to produce "
                            f"a valid AgentAction: {exc}"
                        ),
                    )

                return self._build_result(
                    status="DECISION_ERROR",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        f"Decision engine failed: {exc}"
                    ),
                )

            if not isinstance(
                action,
                AgentAction,
            ):

                self.agent.fail_task()

                return self._build_result(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "Decision engine did not return "
                        "an AgentAction."
                    ),
                )

            try:

                validation = (
                    self.action_validator.validate(
                        action
                    )
                )

            except Exception as exc:

                self.agent.fail_task()

                return self._build_result(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "AgentActionValidator raised an "
                        f"exception: {exc}"
                    ),
                )

            if not validation.valid:

                self.agent.fail_task()

                validation_details = getattr(
                    validation,
                    "errors",
                    None,
                )

                return self._build_result(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "Agent produced an invalid action. "
                        f"errors={validation_details!r}; "
                        f"action={action!r}"
                    ),
                )

            steps += 1

            self.context.record_step()

            if (
                action.action_type
                == AgentActionType.COMPLETE
            ):

                try:

                    self.agent.execute_action(
                        action
                    )

                except Exception as exc:

                    self.agent.fail_task()

                    return self._build_result(
                        status="EXECUTION_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Completion action failed: "
                            f"{exc}"
                        ),
                    )

                return self._build_result(
                    status="COMPLETED",
                    steps=steps,
                    last_result=last_result,
                    reason=action.reason,
                )

            if (
                action.action_type
                == AgentActionType.FAIL
            ):

                try:

                    self.agent.execute_action(
                        action
                    )

                except Exception as exc:

                    self.agent.fail_task()

                    return self._build_result(
                        status="EXECUTION_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Failure action failed: "
                            f"{exc}"
                        ),
                    )

                return self._build_result(
                    status="FAILED",
                    steps=steps,
                    last_result=last_result,
                    reason=action.reason,
                )

            if (
                action.action_type
                == AgentActionType.INVOKE_TOOL
            ):

                try:

                    execution_result = (
                        self.agent.execute_action(
                            action
                        )
                    )

                except Exception as exc:

                    self.agent.fail_task()

                    return self._build_result(
                        status="EXECUTION_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Tool action execution failed: "
                            f"{exc}"
                        ),
                    )

                if not isinstance(
                    execution_result,
                    ToolExecutionResult,
                ):

                    self.agent.fail_task()

                    return self._build_result(
                        status="EXECUTION_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "AgentCore returned an invalid "
                            "tool execution result."
                        ),
                    )

                last_result = execution_result

                self.context.record_tool_result(
                    execution_result
                )

                result_status = getattr(
                    execution_result,
                    "status",
                    None,
                )

                if result_status == "APPROVAL_REQUIRED":

                    self.agent.await_approval()

                    return self._build_result(
                        status="APPROVAL_REQUIRED",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            getattr(
                                execution_result,
                                "summary",
                                None,
                            )
                            or (
                                "Tool execution requires "
                                "additional approval."
                            )
                        ),
                    )

                if result_status != "SUCCESS":

                    self.agent.fail_task()

                    return self._build_result(
                        status="TOOL_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            getattr(
                                execution_result,
                                "summary",
                                None,
                            )
                            or "Tool execution failed."
                        ),
                    )

                continue

            self.agent.fail_task()

            return self._build_result(
                status="INVALID_ACTION",
                steps=steps,
                last_result=last_result,
                reason=(
                    "Unsupported AgentActionType."
                ),
            )

        self.agent.fail_task()

        return self._build_result(
            status="MAX_STEPS_EXCEEDED",
            steps=steps,
            last_result=last_result,
            reason=(
                "Agent execution stopped because "
                "max_steps was exceeded."
            ),
        )

    def _build_result(
        self,
        *,
        status: str,
        steps: int,
        last_result: ToolExecutionResult | None,
        reason: str | None,
    ) -> AgentLoopResult:
        """
        Build an AgentLoopResult for one of `run()`'s many exit
        points, filling in `context` and `token_usage` the same way
        every single time (Build Phase 19; the same "funnel every
        return point through one helper" pattern SecurityDecisionPoint.
        record_execution_outcome()/ToolGateway._finalize() already
        established for a return-point count problem, Build Phase 13).

        `token_usage` is read from `self.decision_engine.total_usage`
        via `getattr(..., None)` -- deliberately duck-typed, not an
        AgentDecisionEngine interface requirement: LLMDecisionEngine
        exposes it (see that class's own docstring), an
        `action_provider`-driven loop has no decision engine at all
        (`self.decision_engine is None`), and any other decision
        engine (DeterministicDecisionEngine, a test double) simply
        doesn't have the attribute -- both cases resolve to `None`
        here, never an error.
        """

        return AgentLoopResult(
            status=status,
            steps=steps,
            last_result=last_result,
            reason=reason,
            context=self.context,
            token_usage=getattr(
                self.decision_engine,
                "total_usage",
                None,
            ),
        )

    def _refresh_available_tools(
        self,
    ) -> None:
        """
        Refresh the tools exposed to the decision engine.

        Tool discovery is performed through
        AgentToolInterface.

        Discovery does not authorize execution.
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

        self.context.set_available_tools(
            available_tools
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