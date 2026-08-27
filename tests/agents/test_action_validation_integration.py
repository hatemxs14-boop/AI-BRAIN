from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_loop import (
    AgentExecutionLoop,
)

from tests.agents.test_agent_loop import (
    build_agent,
)


def test_execution_loop_rejects_invalid_action():

    agent = build_agent()

    agent.start_task(
        "Research AI agents"
    )

    def action_provider(current_agent):

        return AgentAction(
            action_type=(
                AgentActionType.INVOKE_TOOL
            ),
            tool_id=None,
            inputs={
                "query": "AI agents",
            },
            reason="Invalid tool action.",
        )

    loop = AgentExecutionLoop(
        agent=agent,
        action_provider=action_provider,
        max_steps=5,
    )

    result = loop.run()

    assert result.status == (
        "INVALID_ACTION"
    )

    assert result.steps == 0

    assert agent.state.status == (
        "FAILED"
    )


def test_valid_actions_still_execute():

    agent = build_agent()

    agent.start_task(
        "Research AI agents"
    )

    actions = [
        AgentAction(
            action_type=(
                AgentActionType.INVOKE_TOOL
            ),
            tool_id="web_search",
            inputs={
                "query": "AI agents",
            },
            reason="Search for information.",
        ),
        AgentAction(
            action_type=(
                AgentActionType.COMPLETE
            ),
            reason="Research completed.",
        ),
    ]

    index = 0

    def action_provider(current_agent):

        nonlocal index

        action = actions[index]

        index += 1

        return action

    loop = AgentExecutionLoop(
        agent=agent,
        action_provider=action_provider,
        max_steps=5,
    )

    result = loop.run()

    assert result.status == (
        "COMPLETED"
    )

    assert result.steps == 2

    assert len(
        result.context.tool_results
    ) == 1