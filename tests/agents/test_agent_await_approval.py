"""
Regression test for AgentCore.await_approval()/AgentExecutionLoop.

A tool call that comes back APPROVAL_REQUIRED is a recoverable pause,
not a failure: the task can be resumed once an explicit approval
decision is supplied. The loop used to call `agent.fail_task()` for
this case, which set `agent.state.status = "FAILED"` -- making a
resumable, in-progress task indistinguishable from a genuinely dead
one to anything reading `agent.state.status` directly (e.g.
`get_state_snapshot()`), even though `AgentLoopResult.status` already
reported the richer "APPROVAL_REQUIRED" value.

This test exercises the real AgentCore/AgentExecutionLoop/ToolGateway/
SecurityDecisionPoint stack (no mocks) against a HIGH-risk tool that
requires approval, and asserts the agent's own state reflects the
pause rather than a failure.

The HIGH-risk "shell" tool + permission used here is a synthetic,
self-contained fixture built against an isolated tempfile-based
policy (see _build_agent_with_shell_tool) rather than the project's
real core/security/schemas/permissions.json. This test only needs
*some* HIGH-risk permission to exist to exercise the generic
approval-pause mechanism; it must not depend on the real production
policy still granting research_agent shell access, which it no
longer does (see core.agents.research_agent's module docstring).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.agent_loop import AgentExecutionLoop
from core.agents.tool_interface import AgentToolInterface

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


def _write_isolated_shell_policy(tmp_dir: Path) -> Path:
    """
    A minimal, self-contained permissions.json granting exactly the
    one HIGH-risk permission these tests need, isolated from the
    real project policy (see tests/security/test_effective_risk_floor.
    py's _build_conservative_gateway for the established pattern).
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


def _build_agent_with_shell_tool(tmp_dir: Path) -> AgentCore:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="shell",
            name="Shell",
            purpose="Execute shell commands.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            output_schema={"type": "string"},
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

    gateway = ToolGateway(security=security, registry=registry)

    gateway.register_executor(
        tool_id="shell",
        executor=lambda command: "SHOULD NOT EXECUTE WITHOUT APPROVAL",
    )

    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject="research_agent",
        name="Research Agent",
        purpose="Research public information.",
    )

    return AgentCore(identity=identity, tools=interface)


def test_approval_required_pauses_the_agent_instead_of_failing_it():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        agent = _build_agent_with_shell_tool(tmp_dir)

        agent.start_task("Run a shell command that needs approval.")

        def action_provider(current_agent):
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="shell",
                inputs={"command": "echo test"},
                reason="Attempt a HIGH-risk operation without approval.",
            )

        loop = AgentExecutionLoop(
            agent=agent,
            action_provider=action_provider,
            max_steps=3,
        )

        result = loop.run()

        assert result.status == "APPROVAL_REQUIRED"

        # The regression: this must be the distinct "paused" state,
        # never the same "FAILED" state a genuinely dead task would
        # report.
        assert agent.state.status == "AWAITING_APPROVAL"
        assert agent.state.status != "FAILED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_agent_core_await_approval_sets_the_expected_status_directly():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        agent = _build_agent_with_shell_tool(tmp_dir)

        agent.start_task("Run a shell command that needs approval.")

        agent.await_approval()

        assert agent.state.status == "AWAITING_APPROVAL"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fail_task_is_unaffected_and_still_distinct():
    """
    Non-regression guard: await_approval() must not have collapsed
    into fail_task(), and fail_task() must still behave as before.
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        agent = _build_agent_with_shell_tool(tmp_dir)

        agent.start_task("Run a shell command that needs approval.")

        agent.fail_task()

        assert agent.state.status == "FAILED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
