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

        # AgentAction's own constructor already enforces the
        # INVOKE_TOOL/tool_id/inputs contract (see
        # test_action_validator.py's "*_rejected_by_action_contract"
        # tests), so a malformed *AgentAction* can never be
        # constructed here. What this test needs to exercise is the
        # execution loop's handling of a decision engine that hands
        # back something that isn't a valid AgentAction at all.
        return "NOT_AN_AGENT_ACTION"

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