"""
Tests for core.orchestration.langgraph_orchestration_engine.

Two tiers, deliberately:

1. `test_langgraph_orchestration_engine_raises_clear_import_error_
   when_missing` runs for real in every environment that doesn't have
   `langgraph` installed (this sandbox included -- there is no
   package-index access here at all, confirmed separately). It proves
   the *fallback* behavior is real, not just documented.

2. `test_langgraph_orchestration_engine_runs_agent_to_completion` is
   skip-guarded with `pytest.importorskip("langgraph")` and only runs
   where `langgraph` is actually installed -- i.e. never in this
   sandbox, but for real on the user's machine after `pip install -r
   requirements.txt`. This is the same skip-guard pattern already
   established in this project for the live-OpenAI tests in
   tests/agents/test_real_agent_llm_loop.py: cleanly skipped where the
   dependency is unavailable, and a real, unmocked exercise of the
   actual library everywhere it is.

Per this module's own docstring in core/orchestration/
langgraph_orchestration_engine.py, tier 2 is the first real
verification that this project's StateGraph usage actually matches
the installed langgraph API -- until it has run and passed once on a
real machine, this engine should be treated as written-but-unverified,
not done.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

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
        purpose="A minimal agent used only to exercise the LangGraph engine.",
    )

    return AgentCore(identity=identity, tools=interface)


def test_langgraph_orchestration_engine_raises_clear_import_error_when_missing():
    try:
        import langgraph  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip(
            "langgraph is installed in this environment; the missing-"
            "dependency path this test checks doesn't apply here -- "
            "see the next test for the real-integration verification "
            "instead."
        )

    from core.orchestration.langgraph_orchestration_engine import (
        LangGraphOrchestrationEngine,
    )

    with pytest.raises(ImportError, match="langgraph is not installed"):
        LangGraphOrchestrationEngine()


def test_langgraph_orchestration_engine_runs_agent_to_completion():
    pytest.importorskip("langgraph")

    from core.orchestration.langgraph_orchestration_engine import (
        LangGraphOrchestrationEngine,
    )

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        agent = _build_zero_tool_agent(tmp_dir)
        agent.start_task("Do something trivial.")

        engine = LangGraphOrchestrationEngine()

        result = engine.run(
            agent=agent,
            decision_engine=_ImmediateCompleteEngine(),
            max_steps=5,
        )

        assert result.status == "COMPLETED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
