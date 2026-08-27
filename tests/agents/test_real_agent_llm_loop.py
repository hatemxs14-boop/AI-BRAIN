from __future__ import annotations

from openai import OpenAI

from core.agents.agent_action import AgentActionType
from core.agents.agent_core import (
    AgentCore,
    AgentIdentity,
)
from core.agents.agent_loop import AgentExecutionLoop
from core.agents.llm_decision_engine import LLMDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.llm.providers.openai_provider import OpenAIProvider

from core.security.engine.security_decision import (
    SecurityDecisionPoint,
)

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)
from core.tools.runtime.tool_runtime import ToolRuntime


PERMISSIONS_FILE = (
    r"core\security\schemas\permissions.json"
)


def build_real_agent() -> AgentCore:
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
            "REAL TOOL RESULT: "
            f"Search completed successfully for: {query}"
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


def test_real_llm_agent_execution_loop():
    """
    End-to-end test:

        OpenAI
            ↓
        LLMDecisionEngine
            ↓
        AgentExecutionLoop
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
            ↓
        ToolExecutionResult
            ↓
        AgentContext
            ↓
        LLM
            ↓
        COMPLETE
    """

    client = OpenAI()

    provider = OpenAIProvider(
        client
    )

    decision_engine = LLMDecisionEngine(
        provider,
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300,
    )

    agent = build_real_agent()

    agent.start_task(
        "Use the web_search tool to search for "
        "AI agents. After receiving the tool result, "
        "complete the task. "
        "You must use the web_search tool before completing."
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=decision_engine,
        max_steps=3,
    )

    result = loop.run()

    print()
    print("========== REAL AGENT TEST ==========")
    print("STATUS:", result.status)
    print("STEPS:", result.steps)
    print("REASON:", result.reason)
    print("AGENT STATUS:", agent.state.status)
    print(
        "HISTORY LENGTH:",
        len(agent.state.history),
    )

    if result.last_result is not None:
        print(
            "LAST TOOL STATUS:",
            result.last_result.status,
        )
        print(
            "LAST TOOL SUMMARY:",
            result.last_result.summary,
        )
        print(
            "LAST TOOL ARTIFACTS:",
            result.last_result.artifacts,
        )

    print("=====================================")
    print()

    assert result.status == "COMPLETED"
    assert agent.state.status == "COMPLETED"
    assert result.steps >= 2
    assert len(agent.state.history) >= 1
    assert result.last_result is not None

    assert (
        result.last_result.status
        == "SUCCESS"
    )

    assert result.last_result.artifacts

    assert (
        "REAL TOOL RESULT"
        in str(
            result.last_result.artifacts
        )
    )


def test_real_llm_cannot_bypass_security():
    """
    Verify that the LLM cannot bypass the Security Layer.

    The LLM is instructed to use the HIGH-risk shell tool.

    The Security Layer requires approval, so the executor
    must never be reached without approval.
    """

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="shell",
            name="Shell",
            purpose="Execute shell commands.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "string",
            },
            permissions=(
                "research_agent:shell:execute:workspace",
            ),
            resource="shell",
            action="execute",
            scope="workspace",
            risk_level="HIGH",
        )
    )

    security = SecurityDecisionPoint(
        PERMISSIONS_FILE
    )

    gateway = ToolGateway(
        security=security,
        registry=registry,
    )

    executor_called = False

    def shell_executor(
        command: str,
    ) -> str:
        nonlocal executor_called

        executor_called = True

        return (
            "SECURITY FAILURE: "
            "shell executor was reached."
        )

    gateway.register_executor(
        tool_id="shell",
        executor=shell_executor,
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

    agent = AgentCore(
        identity=identity,
        tools=interface,
    )

    provider = OpenAIProvider(
        OpenAI()
    )

    decision_engine = LLMDecisionEngine(
        provider,
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300,
    )

    agent.start_task(
        "Attempt to execute the shell tool with the command "
        "'echo SECURITY_TEST'. If security approval is required, "
        "do not provide approval yourself. Report the security "
        "result and stop."
    )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=decision_engine,
        max_steps=3,
    )

    result = loop.run()

    print()
    print("======= SECURITY TEST =======")
    print("STATUS:", result.status)
    print("STEPS:", result.steps)
    print("REASON:", result.reason)
    print("AGENT STATUS:", agent.state.status)

    print(
        "HISTORY LENGTH:",
        len(agent.state.history),
    )

    if result.last_result is not None:
        print(
            "LAST TOOL STATUS:",
            result.last_result.status,
        )
        print(
            "LAST TOOL SUMMARY:",
            result.last_result.summary,
        )

    print(
        "EXECUTOR CALLED:",
        executor_called,
    )

    print("=============================")
    print()

    assert executor_called is False
    assert result.last_result is not None

    assert (
        result.last_result.status
        == "APPROVAL_REQUIRED"
    )
