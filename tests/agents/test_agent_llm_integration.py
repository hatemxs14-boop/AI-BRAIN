from __future__ import annotations

import pytest

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_context import (
    AgentContext,
)

from core.agents.agent_core import (
    AgentCore,
    AgentIdentity,
)

from core.agents.agent_loop import (
    AgentExecutionLoop,
)

from core.agents.llm_decision_engine import (
    LLMDecisionEngine,
)

from core.agents.tool_interface import (
    AgentToolInterface,
)

from core.llm.llm_client import (
    LLMClient,
)

from core.llm.llm_request import (
    LLMRequest,
)

from core.llm.llm_response import (
    LLMResponse,
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


class MockToolRuntime:

    def __init__(self):
        self.executions = []

    def discover_tools_for_subject(
        self,
        subject,
    ):

        if subject != "research_agent":
            return ()

        return (
            ToolDiscovery(
                id="web_search",
                name="Web Search",
                purpose="Search the public web.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": ["query"],
                },
                output_schema={
                    "type": "string"
                },
                permissions=(
                    "research_agent:web:search:public",
                ),
                resource="web",
                action="search",
                scope="public",
                risk_level="LOW",
            ),
        )

    def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolExecutionResult:

        self.executions.append(invocation)

        return ToolExecutionResult(
            status="SUCCESS",
            summary="Tool executed successfully.",
            next_actions=(),
            artifacts=("SEARCH RESULT",),
            security_decision=None,
            subject=invocation.subject,
            tool_id=invocation.tool_id,
            action="search",
        )


class MockLLMClient(LLMClient):

    def __init__(
        self,
        responses,
    ):
        self.responses = list(responses)
        self.requests = []

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:

        self.requests.append(request)

        return LLMResponse(
            content=self.responses.pop(0),
            model="mock-model",
            finish_reason="stop",
        )


def build_agent(
    runtime,
):

    tools = AgentToolInterface(
        runtime=runtime,
    )

    identity = AgentIdentity(
        subject="research_agent",
        name="Research Agent",
        purpose="Research information using authorized tools.",
    )

    return AgentCore(
        identity=identity,
        tools=tools,
    )


def test_llm_drives_agent_from_tool_to_complete():

    runtime = MockToolRuntime()

    client = MockLLMClient(
        responses=[
            (
                '{"action_type":"INVOKE_TOOL",'
                '"tool_id":"web_search",'
                '"inputs":{"query":"Research AI agents"},'
                '"reason":"Research is required."}'
            ),
            (
                '{"action_type":"COMPLETE",'
                '"reason":"Research completed."}'
            ),
        ]
    )

    engine = LLMDecisionEngine(
        client,
        model="integration-model",
    )

    agent = build_agent(
        runtime
    )

    agent.start_task(
        "Research AI agents"
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        max_steps=5,
    )

    result = loop.run()

    assert result is not None
    assert agent.state.status == "COMPLETED"
    assert len(runtime.executions) == 1
    assert runtime.executions[0].tool_id == "web_search"


def test_llm_receives_tool_result_on_next_decision():

    runtime = MockToolRuntime()

    client = MockLLMClient(
        responses=[
            (
                '{"action_type":"INVOKE_TOOL",'
                '"tool_id":"web_search",'
                '"inputs":{"query":"AI"},'
                '"reason":"Search required."}'
            ),
            (
                '{"action_type":"COMPLETE",'
                '"reason":"Result received."}'
            ),
        ]
    )

    engine = LLMDecisionEngine(
        client
    )

    agent = build_agent(
        runtime
    )

    agent.start_task(
        "AI"
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        max_steps=5,
    )

    result = loop.run()

    assert result is not None
    assert agent.state.status == "COMPLETED"

    assert len(client.requests) == 2

    second_request = client.requests[1]

    assert "SEARCH RESULT" in (
        second_request.messages[1].content
    )


def test_llm_complete_action_stops_without_tool_execution():

    runtime = MockToolRuntime()

    client = MockLLMClient(
        responses=[
            (
                '{"action_type":"COMPLETE",'
                '"reason":"Already complete."}'
            ),
        ]
    )

    engine = LLMDecisionEngine(
        client
    )

    agent = build_agent(
        runtime
    )

    agent.start_task(
        "Already complete"
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        max_steps=5,
    )

    result = loop.run()

    assert result is not None
    assert agent.state.status == "COMPLETED"
    assert len(runtime.executions) == 0


def test_llm_fail_action_stops_agent():

    runtime = MockToolRuntime()

    client = MockLLMClient(
        responses=[
            (
                '{"action_type":"FAIL",'
                '"reason":"Unable to complete task."}'
            ),
        ]
    )

    engine = LLMDecisionEngine(
        client
    )

    agent = build_agent(
        runtime
    )

    agent.start_task(
        "Impossible task"
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=engine,
        max_steps=5,
    )

    result = loop.run()

    assert result is not None
    assert agent.state.status == "FAILED"
    assert len(runtime.executions) == 0
