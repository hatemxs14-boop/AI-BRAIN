from __future__ import annotations

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway

from core.tools.registry.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)


PERMISSIONS_FILE = r".\core\security\schemas\permissions.json"


def build_gateway(
    *,
    executor,
    output_schema=None,
):
    registry = ToolRegistry()

    if output_schema is None:
        output_schema = {"type": "string"}

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
            output_schema=output_schema,
            permissions=(
                "research_agent:web_search:search:public_web",
            ),
            resource="web_search",
            action="search",
            scope="public_web",
            risk_level="LOW",
        )
    )

    security = SecurityDecisionPoint(PERMISSIONS_FILE)

    gateway = ToolGateway(
        security=security,
        registry=registry,
    )

    gateway.register_executor(
        tool_id="web_search",
        executor=executor,
    )

    return gateway


def test_successful_execution():
    gateway = build_gateway(
        executor=lambda query: "SUCCESS: " + query
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="web_search",
        tool_kwargs={
            "query": "AI agents",
        },
    )

    assert result.status == "SUCCESS"
    assert result.artifacts == ("SUCCESS: AI agents",)


def test_invalid_input_missing_required_field():
    gateway = build_gateway(
        executor=lambda query: "THIS MUST NOT EXECUTE"
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="web_search",
        tool_kwargs={},
    )

    assert result.status == "INVALID_INPUT"
    assert "Missing required input field" in result.artifacts[0]


def test_invalid_input_unknown_field():
    gateway = build_gateway(
        executor=lambda query: "THIS MUST NOT EXECUTE"
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="web_search",
        tool_kwargs={
            "query": "AI agents",
            "unknown": "attack",
        },
    )

    assert result.status == "INVALID_INPUT"
    assert "Unknown input field" in result.artifacts[0]


def test_invalid_input_wrong_type():
    gateway = build_gateway(
        executor=lambda query: "THIS MUST NOT EXECUTE"
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="web_search",
        tool_kwargs={
            "query": 123,
        },
    )

    assert result.status == "INVALID_INPUT"
    assert "Invalid type" in result.artifacts[0]


def test_invalid_output():
    gateway = build_gateway(
        executor=lambda query: 123,
        output_schema={
            "type": "string",
        },
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="web_search",
        tool_kwargs={
            "query": "AI agents",
        },
    )

    assert result.status == "INVALID_OUTPUT"


def test_executor_failure():
    def failing_executor(query):
        raise RuntimeError("SIMULATED TOOL FAILURE")

    gateway = build_gateway(
        executor=failing_executor
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="web_search",
        tool_kwargs={
            "query": "AI agents",
        },
    )

    assert result.status == "ERROR"
    assert "SIMULATED TOOL FAILURE" in result.artifacts[0]


def test_unknown_tool_is_denied():
    registry = ToolRegistry()

    security = SecurityDecisionPoint(PERMISSIONS_FILE)

    gateway = ToolGateway(
        security=security,
        registry=registry,
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="secret_tool",
        tool_kwargs={},
    )

    assert result.status == "DENIED"


def test_direct_executor_is_not_exposed_by_definition():
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="web_search",
            name="Web Search",
            purpose="Search the public web.",
            input_schema={},
            output_schema={},
            permissions=(
                "research_agent:web_search:search:public_web",
            ),
            resource="web_search",
            action="search",
            scope="public_web",
            risk_level="LOW",
        )
    )

    security = SecurityDecisionPoint(PERMISSIONS_FILE)

    gateway = ToolGateway(
        security=security,
        registry=registry,
    )

    definition = registry.get("web_search")

    assert not hasattr(definition, "executor")
    assert hasattr(gateway, "_executors")


def test_executor_is_not_called_when_input_is_invalid():
    execution_state = {
        "called": False,
    }

    def executor(query):
        execution_state["called"] = True
        return "SHOULD NOT EXECUTE"

    gateway = build_gateway(
        executor=executor
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="web_search",
        tool_kwargs={},
    )

    assert result.status == "INVALID_INPUT"
    assert execution_state["called"] is False


def test_executor_is_not_called_when_security_denies():
    execution_state = {
        "called": False,
    }

    def executor(query):
        execution_state["called"] = True
        return "SHOULD NOT EXECUTE"

    gateway = build_gateway(
        executor=executor
    )

    result = gateway.execute(
        subject="unauthorized_agent",
        tool_id="web_search",
        tool_kwargs={
            "query": "AI agents",
        },
    )

    assert result.status == "DENIED"
    assert execution_state["called"] is False


def test_high_risk_requires_approval():
    execution_state = {
        "called": False,
    }

    def shell_executor(command):
        execution_state["called"] = True
        return "SHELL EXECUTED"

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="shell",
            name="Shell",
            purpose="Execute a shell command.",
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

    security = SecurityDecisionPoint(PERMISSIONS_FILE)

    gateway = ToolGateway(
        security=security,
        registry=registry,
    )

    gateway.register_executor(
        tool_id="shell",
        executor=shell_executor,
    )

    result = gateway.execute(
        subject="research_agent",
        tool_id="shell",
        tool_kwargs={
            "command": "echo test",
        },
    )

    assert result.status == "APPROVAL_REQUIRED"
    assert execution_state["called"] is False