"""
Tests for core.orchestration.orchestration_engine and
core.orchestration.engine_factory.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.orchestration.engine_factory import (
    create_default_orchestration_engine,
)

from core.orchestration.orchestration_engine import (
    SequentialOrchestrationEngine,
)

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


class _ImmediateCompleteEngine(AgentDecisionEngine):
    def decide(self, context):
        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Nothing to do.",
        )


def _build_zero_tool_agent(tmp_dir: Path) -> AgentCore:
    registry = ToolRegistry()

    policy = {
        "version": "1.0",
        "permissions": [],
        "defaults": {
            "unknown_risk": "DENY",
            "unknown_permission": "DENY",
            "unknown_scope": "DENY",
            "authorization_failure": "DENY",
        },
    }
    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)
    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject="test_agent",
        name="Test Agent",
        purpose="A minimal agent used only to exercise the orchestration layer.",
    )

    return AgentCore(identity=identity, tools=interface)


def test_sequential_orchestration_engine_runs_agent_to_completion():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        agent = _build_zero_tool_agent(tmp_dir)
        agent.start_task("Do something trivial.")

        engine = SequentialOrchestrationEngine()

        result = engine.run(
            agent=agent,
            decision_engine=_ImmediateCompleteEngine(),
            max_steps=5,
        )

        assert result.status == "COMPLETED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_create_default_orchestration_engine_returns_sequential_when_not_preferred():
    engine = create_default_orchestration_engine(prefer_langgraph=False)

    assert isinstance(engine, SequentialOrchestrationEngine)


def test_create_default_orchestration_engine_falls_back_when_langgraph_missing():
    """
    Real, not simulated: this sandbox genuinely has no `langgraph`
    installed (no package-index access at all -- see
    core/orchestration/langgraph_orchestration_engine.py's own
    docstring), so this exercises the actual ImportError fallback
    path, not a mocked one. On a machine where `langgraph` *is*
    installed (e.g. after `pip install -r requirements.txt`), this
    assertion would need `prefer_langgraph=False` to still observe the
    Sequential engine -- which the test above already covers.
    """

    engine = create_default_orchestration_engine()

    try:
        import langgraph  # noqa: F401

        langgraph_installed = True
    except ImportError:
        langgraph_installed = False

    if not langgraph_installed:
        assert isinstance(engine, SequentialOrchestrationEngine)
    else:
        # langgraph is available in this environment -- the factory is
        # expected to have preferred the real LangGraphOrchestrationEngine
        # instead of falling back. Verified structurally rather than by
        # isinstance import here to avoid this file itself depending on
        # langgraph being installed.
        assert engine.__class__.__name__ == "LangGraphOrchestrationEngine"
