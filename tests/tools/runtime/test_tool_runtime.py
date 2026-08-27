from __future__ import annotations

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway

from core.tools.registry.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)

from core.tools.runtime.tool_invocation import (
    ToolInvocation,
)

from core.tools.runtime.tool_runtime import (
    ToolDiscovery,
    ToolRuntime,
)


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


def build_gateway():
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
                "admin_agent:shell:execute:workspace",
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

    return registry, gateway


def test_runtime_executes_tool_through_gateway():
    registry, gateway = build_gateway()

    gateway.register_executor(
        tool_id="web_search",
        executor=lambda query: "RUNTIME TEST: " + query,
    )

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    invocation = ToolInvocation(
        subject="research_agent",
        tool_id="web_search",
        inputs={
            "query": "AI agents",
        },
    )

    result = runtime.execute(invocation)

    assert result.status == "SUCCESS"
    assert result.artifacts == (
        "RUNTIME TEST: AI agents",
    )


def test_runtime_rejects_invalid_invocation_type():
    registry, gateway = build_gateway()

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    try:
        runtime.execute("invalid")
    except TypeError as exc:
        assert "ToolInvocation" in str(exc)
    else:
        raise AssertionError(
            "Runtime accepted an invalid invocation type."
        )


def test_runtime_discovery_hides_executor():
    registry, gateway = build_gateway()

    gateway.register_executor(
        tool_id="web_search",
        executor=lambda query: "SECRET EXECUTOR",
    )

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    discovery = runtime.discover_tool(
        "web_search"
    )

    assert isinstance(
        discovery,
        ToolDiscovery,
    )

    assert discovery.id == "web_search"
    assert discovery.name == "Web Search"
    assert discovery.risk_level == "LOW"

    assert not hasattr(
        discovery,
        "executor",
    )


def test_runtime_filters_tools_by_subject():
    registry, gateway = build_gateway()

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    research_tools = (
        runtime.discover_tools_for_subject(
            "research_agent"
        )
    )

    admin_tools = (
        runtime.discover_tools_for_subject(
            "admin_agent"
        )
    )

    research_ids = tuple(
        tool.id
        for tool in research_tools
    )

    admin_ids = tuple(
        tool.id
        for tool in admin_tools
    )

    assert research_ids == (
        "web_search",
    )

    assert admin_ids == (
        "shell",
    )


def test_runtime_does_not_execute_invalid_input():
    execution_state = {
        "called": False,
    }

    registry, gateway = build_gateway()

    gateway.register_executor(
        tool_id="web_search",
        executor=lambda query: (
            execution_state.__setitem__(
                "called",
                True,
            ),
            "SHOULD NOT EXECUTE",
        )[1],
    )

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    invocation = ToolInvocation(
        subject="research_agent",
        tool_id="web_search",
        inputs={},
    )

    result = runtime.execute(invocation)

    assert result.status == "INVALID_INPUT"
    assert execution_state["called"] is False


def test_runtime_blocks_unknown_tool():
    registry, gateway = build_gateway()

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    invocation = ToolInvocation(
        subject="research_agent",
        tool_id="unknown_tool",
        inputs={},
    )

    result = runtime.execute(invocation)

    assert result.status == "DENIED"


def test_runtime_preserves_high_risk_approval():
    execution_state = {
        "called": False,
    }

    registry, gateway = build_gateway()

    gateway.register_executor(
        tool_id="shell",
        executor=lambda command: (
            execution_state.__setitem__(
                "called",
                True,
            ),
            "SHELL EXECUTED",
        )[1],
    )

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    invocation = ToolInvocation(
        subject="research_agent",
        tool_id="shell",
        inputs={
            "command": "echo test",
        },
    )

    result = runtime.execute(invocation)

    assert result.status == "APPROVAL_REQUIRED"
    assert execution_state["called"] is False


def test_runtime_preserves_explicit_approval():
    registry, gateway = build_gateway()

    gateway.register_executor(
        tool_id="shell",
        executor=lambda command: (
            "APPROVED SHELL: " + command
        ),
    )

    runtime = ToolRuntime(
        registry=registry,
        gateway=gateway,
    )

    invocation = ToolInvocation(
        subject="research_agent",
        tool_id="shell",
        inputs={
            "command": "echo approved",
        },
        approved=True,
        approved_by="human_operator",
    )

    result = runtime.execute(invocation)

    assert result.status == "SUCCESS"
    assert result.artifacts == (
        "APPROVED SHELL: echo approved",
    )