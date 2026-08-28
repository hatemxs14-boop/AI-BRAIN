from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway

from core.tools.registry.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


def _write_isolated_shell_policy(tmp_dir: Path) -> Path:
    """
    A minimal, self-contained permissions.json granting exactly the
    HIGH-risk shell permission test_high_risk_requires_approval needs,
    isolated from the real project policy (which no longer grants
    research_agent any shell-related permission -- see core.agents.
    research_agent's module docstring). Every other test in this file
    uses the real PERMISSIONS_FILE via build_gateway() and is
    unaffected.
    """

    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": "research_agent",
                "resource": "shell",
                "action": "execute",
                "scope": "workspace",
                "risk_level": "HIGH",
                "approval": "policy",
            }
        ],
        "defaults": {
            "unknown_risk": "DENY",
            "unknown_permission": "DENY",
            "unknown_scope": "DENY",
            "authorization_failure": "DENY",
        },
    }

    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path


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
            error_handling={
                "retryable": True,
                "max_retries": 2,
                "on_failure": "Surface the search error to the agent.",
            },
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
            error_handling={
                "retryable": True,
                "max_retries": 2,
                "on_failure": "Surface the search error to the agent.",
            },
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

    tmp_dir = Path(tempfile.mkdtemp())
    try:
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
                error_handling={
                    "retryable": False,
                    "on_failure": (
                        "Do not retry a shell command automatically; "
                        "surface the failure for human review."
                    ),
                },
            )
        )

        policy_path = _write_isolated_shell_policy(tmp_dir)

        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# ToolExecutionResult.subject/tool_id/action (Build Phase 7: added so
# a Kernel-level consumer of AgentLoopResult.last_result can answer
# PolicyEngine.evaluate_external_action()'s six questions without any
# caller having to separately track which tool_id/action produced a
# given result -- see ToolExecutionResult's own docstring in
# core/tools/engine/tool_gateway.py).
# ---------------------------------------------------------------------

def test_successful_execution_reports_subject_tool_id_and_action():
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

    assert result.subject == "research_agent"
    assert result.tool_id == "web_search"
    assert result.action == "search"


def test_unknown_tool_reports_subject_and_tool_id_with_execute_action():
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
    assert result.subject == "research_agent"
    assert result.tool_id == "secret_tool"
    # No ToolDefinition was ever resolved -- "execute" is the
    # documented default for exactly this case.
    assert result.action == "execute"


def test_denied_execution_reports_subject_tool_id_and_action():
    gateway = build_gateway(
        executor=lambda query: "SHOULD NOT EXECUTE"
    )

    result = gateway.execute(
        subject="unauthorized_agent",
        tool_id="web_search",
        tool_kwargs={
            "query": "AI agents",
        },
    )

    assert result.status == "DENIED"
    assert result.subject == "unauthorized_agent"
    assert result.tool_id == "web_search"
    assert result.action == "search"


def test_approval_required_execution_reports_subject_tool_id_and_action():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
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
                error_handling={
                    "retryable": False,
                    "on_failure": (
                        "Do not retry a shell command automatically; "
                        "surface the failure for human review."
                    ),
                },
            )
        )

        policy_path = _write_isolated_shell_policy(tmp_dir)

        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )

        gateway = ToolGateway(
            security=security,
            registry=registry,
        )

        gateway.register_executor(
            tool_id="shell",
            executor=lambda command: "SHELL EXECUTED",
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id="shell",
            tool_kwargs={
                "command": "echo test",
            },
        )

        assert result.status == "APPROVAL_REQUIRED"
        assert result.subject == "research_agent"
        assert result.tool_id == "shell"
        assert result.action == "execute"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)