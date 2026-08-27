from __future__ import annotations

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_context import (
    AgentContext,
)

from core.agents.decision_engine import (
    AgentDecisionEngine,
)


class DeterministicDecisionEngine(
    AgentDecisionEngine
):
    """
    Deterministic decision engine used for testing
    the Agent architecture before an LLM is introduced.

    Behavior:

    - if no tool result exists, invoke web_search
    - otherwise, complete the task
    """

    def decide(
        self,
        context: AgentContext,
    ) -> AgentAction:

        if not context.tool_results:

            return AgentAction(
                action_type=(
                    AgentActionType.INVOKE_TOOL
                ),
                tool_id="web_search",
                inputs={
                    "query": context.task,
                },
                reason=(
                    "No research result exists yet."
                ),
            )

        return AgentAction(
            action_type=(
                AgentActionType.COMPLETE
            ),
            reason=(
                "The required research step is complete."
            ),
        )