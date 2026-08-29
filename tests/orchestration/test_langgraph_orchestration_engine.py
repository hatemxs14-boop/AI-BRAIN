"""
Tests for core.orchestration.langgraph_orchestration_engine.

Two tiers:

1. `test_langgraph_orchestration_engine_raises_clear_import_error_
   when_missing` simulates "langgraph is not installed" deterministically
   in ANY environment via `monkeypatch.setitem(sys.modules, "langgraph",
   None)` -- Python's own import machinery raises `ModuleNotFoundError`
   for `import langgraph`/`from langgraph... import ...` while that
   sys.modules entry is `None`, independent of whether the real package
   is actually installed on disk. This used to instead skip itself
   whenever `langgraph` genuinely was installed ("the missing-dependency
   path this test checks doesn't apply here") -- correct at the time,
   but once this project's own reference machine had `langgraph`
   installed for real (Build Phase 24), that left no environment in
   which this fallback behavior was ever actually exercised. The fix
   makes it run for real, everywhere, always -- proving the *fallback*
   behavior is real, not just documented, regardless of the host
   environment.

2. `test_langgraph_orchestration_engine_runs_agent_to_completion` is
   skip-guarded with `pytest.importorskip("langgraph")` and only runs
   where `langgraph` is actually installed -- i.e. never in this
   sandbox, but for real on the user's machine (confirmed passing,
   Build Phase 24's own investigation). This is the same skip-guard
   pattern already established in this project for the live-OpenAI
   tests in tests/agents/test_real_agent_llm_loop.py: cleanly skipped
   where the dependency is unavailable, and a real, unmocked exercise
   of the actual library everywhere it is -- this one genuinely cannot
   be made to run without a real `langgraph` installed, unlike tier 1
   above, since it exercises the real compiled graph's `invoke()`, not
   an import-failure branch.

Per this module's own docstring in core/orchestration/
langgraph_orchestration_engine.py, tier 2 was the first real
verification that this project's StateGraph usage actually matches the
installed langgraph API -- confirmed passing for real, see that
module's docstring.
"""
from __future__ import annotations

import json
import shutil
import sys
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


# Every exact dotted name core/orchestration/langgraph_orchestration_
# engine.py's __init__ imports. Blanking only "langgraph" itself is NOT
# sufficient: if "langgraph.graph" happens to already be cached in
# sys.modules (e.g. from pytest-langsmith or another plugin importing
# it eagerly at session start, or from a different test that already
# ran for real in this same process), Python's import machinery
# returns that cached submodule directly without ever re-checking the
# (blanked) parent -- confirmed as a real machine failure ("DID NOT
# RAISE ImportError") the first time this test used only
# `sys.modules["langgraph"] = None`. Blanking every literal dotted
# name actually imported, regardless of whether it happens to be
# cached already, makes the very first cache lookup for that exact
# name hit `None` and raise immediately, in any environment.
_LANGGRAPH_IMPORT_PATHS = ("langgraph", "langgraph.graph")


def test_langgraph_orchestration_engine_raises_clear_import_error_when_missing(
    monkeypatch,
):
    for name in _LANGGRAPH_IMPORT_PATHS:
        monkeypatch.setitem(sys.modules, name, None)

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
