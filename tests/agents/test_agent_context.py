from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_context import (
    AgentContext,
)

from core.agents.agent_loop import (
    AgentExecutionLoop,
)

from tests.agents.test_agent_loop import (
    build_agent,
)


def test_agent_context_records_execution():

    context = AgentContext(
        task="Research AI agents"
    )

    assert context.task == (
        "Research AI agents"
    )

    assert context.step_count == 0

    assert context.tool_results == []


def test_agent_context_records_steps_and_results():

    context = AgentContext(
        task="Research AI agents"
    )

    context.record_step()

    context.record_tool_result(
        "SEARCH RESULT"
    )

    context.record_step()

    assert context.step_count == 2

    assert context.tool_results == [
        "SEARCH RESULT"
    ]


def test_agent_context_stores_metadata():

    context = AgentContext(
        task="Research AI agents"
    )

    context.set_metadata(
        "source",
        "web",
    )

    assert (
        context.get_metadata(
            "source"
        )
        == "web"
    )

    assert (
        context.get_metadata(
            "missing"
        )
        is None
    )


def test_execution_loop_populates_context():

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

    action_index = 0

    def action_provider(current_agent):

        nonlocal action_index

        action = actions[action_index]

        action_index += 1

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

    assert result.context.task == (
        "Research AI agents"
    )

    assert result.context.step_count == 2

    assert len(
        result.context.tool_results
    ) == 1

    assert (
        result.context.tool_results[0]
        .artifacts
        == (
            "LOOP TOOL RESULT: AI agents",
        )
    )


def test_context_does_not_execute_tools():

    context = AgentContext(
        task="Research AI agents"
    )

    assert not hasattr(
        context,
        "execute",
    )

    assert not hasattr(
        context,
        "executor",
    )