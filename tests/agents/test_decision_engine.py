from core.agents.agent_action import (
    AgentActionType,
)

from core.agents.agent_context import (
    AgentContext,
)

from core.agents.deterministic_decision_engine import (
    DeterministicDecisionEngine,
)

from core.agents.agent_loop import (
    AgentExecutionLoop,
)

from tests.agents.test_agent_loop import (
    build_agent,
)


def test_decision_engine_invokes_search_first():

    engine = DeterministicDecisionEngine()

    context = AgentContext(
        task="Research AI agents"
    )

    action = engine.decide(
        context
    )

    assert (
        action.action_type
        == AgentActionType.INVOKE_TOOL
    )

    assert action.tool_id == (
        "web_search"
    )

    assert action.inputs == {
        "query": "Research AI agents",
    }


def test_decision_engine_completes_after_result():

    engine = DeterministicDecisionEngine()

    context = AgentContext(
        task="Research AI agents"
    )

    context.record_tool_result(
        "SEARCH RESULT"
    )

    action = engine.decide(
        context
    )

    assert (
        action.action_type
        == AgentActionType.COMPLETE
    )


def test_execution_loop_uses_decision_engine():

    agent = build_agent()

    agent.start_task(
        "Research AI agents"
    )

    engine = DeterministicDecisionEngine()

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        max_steps=5,
    )

    result = loop.run()

    assert result.status == (
        "COMPLETED"
    )

    assert result.steps == 2

    assert result.context.step_count == 2

    assert len(
        result.context.tool_results
    ) == 1

    assert (
        result.context.tool_results[0]
        .artifacts
        == (
            "LOOP TOOL RESULT: Research AI agents",
        )
    )

    assert agent.state.status == (
        "COMPLETED"
    )