from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_core import (
    AgentCore,
    AgentIdentity,
)

from core.agents.agent_loop import (
    AgentExecutionLoop,
)

from core.agents.tool_interface import (
    AgentToolInterface,
)

from core.tools.engine.tool_gateway import (
    ToolGateway,
)

from core.tools.registry.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)

from core.tools.runtime.tool_runtime import (
    ToolRuntime,
)

from core.security.engine.security_decision import (
    SecurityDecisionPoint,
)


PERMISSIONS_FILE = (
    r".\core\security\schemas\permissions.json"
)


def build_agent() -> AgentCore:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="web_search",
            name="Web Search",
            purpose="Search the public web.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "string",
            },
            permissions=(
                "research_agent:web_search:search:public_web",
            ),
            resource="web_search",
            action="search",
            scope="public_web",
            risk_level="LOW",
        )
    )

    security = SecurityDecisionPoint(
        PERMISSIONS_FILE
    )

    gateway = ToolGateway(
        security=security,
        registry=registry,
    )

    gateway.register_executor(
        tool_id="web_search",
        executor=lambda query: (
            "LOOP TOOL RESULT: " + query
        ),
    )

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    interface = AgentToolInterface(
        runtime=runtime,
    )

    identity = AgentIdentity(
        subject="research_agent",
        name="Research Agent",
        purpose="Research public information.",
    )

    return AgentCore(
        identity=identity,
        tools=interface,
    )


def test_agent_execution_loop():

    agent = build_agent()

    agent.start_task(
        "Research AI agents"
    )

    actions = [
        AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="web_search",
            inputs={
                "query": "AI agents",
            },
            reason="Search for information.",
        ),
        AgentAction(
            action_type=AgentActionType.COMPLETE,
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

    assert result.status == "COMPLETED"
    assert result.steps == 2

    assert agent.state.status == "COMPLETED"

    assert len(agent.state.history) == 1

    assert (
        agent.state.last_result.artifacts
        == ("LOOP TOOL RESULT: AI agents",)
    )


def test_agent_execution_loop_respects_max_steps():

    agent = build_agent()

    agent.start_task(
        "Research AI agents"
    )

    def action_provider(current_agent):
        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="web_search",
            inputs={
                "query": "AI agents",
            },
        )

    loop = AgentExecutionLoop(
        agent=agent,
        action_provider=action_provider,
        max_steps=2,
    )

    result = loop.run()

    assert result.status == "MAX_STEPS_EXCEEDED"
    assert result.steps == 2

    assert agent.state.status == "FAILED"


def test_agent_execution_loop_handles_fail_action():

    agent = build_agent()

    agent.start_task(
        "Research AI agents"
    )

    def action_provider(current_agent):
        return AgentAction(
            action_type=AgentActionType.FAIL,
            reason="Research failed.",
        )

    loop = AgentExecutionLoop(
        agent=agent,
        action_provider=action_provider,
        max_steps=5,
    )

    result = loop.run()

    assert result.status == "FAILED"
    assert result.steps == 1

    assert agent.state.status == "FAILED"

    assert result.reason == "Research failed."