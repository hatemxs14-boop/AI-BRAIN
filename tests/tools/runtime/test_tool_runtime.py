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

from core.tools.runtime.tool_invocation import (
    ToolInvocation,
)

from core.tools.runtime.tool_runtime import (
    ToolDiscovery,
    ToolRuntime,
)


def _write_isolated_policy(tmp_dir: Path) -> Path:
    """
    A minimal, self-contained permissions.json granting both
    permissions this file's tests need: research_agent's real
    web_search grant (kept in sync with the production LOW-risk
    entry), plus a HIGH-risk research_agent shell grant that the
    real production policy no longer has (see core.agents.
    research_agent's module docstring) but the approval-preservation
    tests below still need as fixture data. Isolating this file from
    the real policy decouples these generic plumbing tests from the
    evolving real-world security policy content -- the same pattern
    already used in tests/security/test_effective_risk_floor.py.
    """

    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": "research_agent",
                "resource": "web_search",
                "action": "search",
                "scope": "public_web",
                "risk_level": "LOW",
                "approval": "none",
            },
            {
                "subject": "research_agent",
                "resource": "shell",
                "action": "execute",
                "scope": "workspace",
                "risk_level": "HIGH",
                "approval": "policy",
            },
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


def build_gateway(tmp_dir: Path):
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
            error_handling={
                "retryable": True,
                "max_retries": 2,
                "on_failure": "Surface the search error to the agent.",
            },
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
            error_handling={
                "retryable": False,
                "on_failure": (
                    "Do not retry a shell command automatically; "
                    "surface the failure for human review."
                ),
            },
        )
    )

    policy_path = _write_isolated_policy(tmp_dir)

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

    gateway = ToolGateway(
        security=security,
        registry=registry,
    )

    return registry, gateway


def test_runtime_executes_tool_through_gateway():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        registry, gateway = build_gateway(tmp_dir)

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_runtime_rejects_invalid_invocation_type():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        registry, gateway = build_gateway(tmp_dir)

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_runtime_discovery_hides_executor():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        registry, gateway = build_gateway(tmp_dir)

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_runtime_filters_tools_by_subject():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        registry, gateway = build_gateway(tmp_dir)

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_runtime_does_not_execute_invalid_input():
    execution_state = {
        "called": False,
    }

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        registry, gateway = build_gateway(tmp_dir)

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_runtime_blocks_unknown_tool():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        registry, gateway = build_gateway(tmp_dir)

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_runtime_preserves_high_risk_approval():
    execution_state = {
        "called": False,
    }

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        registry, gateway = build_gateway(tmp_dir)

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_runtime_preserves_explicit_approval():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        registry, gateway = build_gateway(tmp_dir)

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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)