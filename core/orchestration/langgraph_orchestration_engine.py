from __future__ import annotations

from typing import Any, TypedDict

from core.agents.agent_core import (
    AgentCore,
)

from core.agents.agent_loop import (
    AgentExecutionLoop,
    AgentLoopResult,
)

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.orchestration.orchestration_engine import (
    OrchestrationEngine,
)


# ---------------------------------------------------------------------
# LangGraph-backed OrchestrationEngine.
#
# core/orchestration/EXECUTION_ENGINE.md and ARCHITECTURE.md both
# state plainly that LangGraph is AI-BRAIN's execution/orchestration
# engine. This module is the real integration: a `langgraph.graph.
# StateGraph` with one node that runs the existing, unmodified
# AgentExecutionLoop, compiled and invoked through LangGraph's own
# runtime rather than called directly.
#
# IMPORTANT -- READ BEFORE TRUSTING THIS FILE:
#
# `langgraph` (pinned to 1.2.11 in requirements.txt) cannot be
# installed in the sandbox this module was written in -- there is no
# package-index access there at all (`pip download langgraph==1.2.11`
# returns "No matching distribution found", not a version-resolution
# error). That means the actual `StateGraph`/`compile()`/`invoke()`
# calls below have never been executed anywhere. They are written
# against LangGraph's public graph-construction API as documented as
# of this project's knowledge cutoff (StateGraph + a typed state +
# add_node/add_edge/START/END + compile().invoke()), which has been
# stable across that library's releases, but the exact pinned version
# has not been checked against this code.
#
# What IS verified in-sandbox (see tests/orchestration/
# test_langgraph_orchestration_engine.py): that constructing this
# class raises a clear ImportError when `langgraph` isn't installed
# (exercised for real here, since it genuinely isn't), and that
# `create_default_orchestration_engine()` (engine_factory.py) falls
# back to SequentialOrchestrationEngine when that happens -- so the
# Kernel never becomes unusable because of this file.
#
# What is NOT yet verified: that `run()` below actually produces a
# correct AgentLoopResult when a real `langgraph` is installed. That
# is the very next thing to check on the user's machine (`pip install
# -r requirements.txt`, then real `pytest -v`) before this class is
# considered done, not merely written -- consistent with every other
# piece of this project. If the installed API differs from what's
# used here, fix this one isolated file; nothing else in the Kernel or
# Orchestration layer depends on LangGraph's API directly.
# ---------------------------------------------------------------------


class _KernelGraphState(TypedDict):
    agent: AgentCore
    decision_engine: AgentDecisionEngine
    max_steps: int
    result: AgentLoopResult | None


def _run_agent_node(
    state: _KernelGraphState,
) -> dict[str, Any]:
    """
    The graph's single node: runs the unmodified
    AgentExecutionLoop and returns its result as a state update.
    """

    loop = AgentExecutionLoop(
        agent=state["agent"],
        decision_engine=state["decision_engine"],
        max_steps=state["max_steps"],
    )

    return {
        "result": loop.run(),
    }


class LangGraphOrchestrationEngine(OrchestrationEngine):
    """
    OrchestrationEngine backed by a real, compiled LangGraph
    StateGraph.

    Raises ImportError at construction time if `langgraph` is not
    installed -- callers that want automatic, silent fallback to
    SequentialOrchestrationEngine should use
    engine_factory.create_default_orchestration_engine() instead of
    constructing this class directly.
    """

    def __init__(self) -> None:

        try:
            from langgraph.graph import (
                END,
                START,
                StateGraph,
            )
        except ImportError as exc:
            raise ImportError(
                "langgraph is not installed in this environment, so "
                "LangGraphOrchestrationEngine cannot be constructed. "
                "Install it with `pip install -r requirements.txt` "
                "(it is pinned there), or use "
                "SequentialOrchestrationEngine / "
                "create_default_orchestration_engine() instead, which "
                "runs the same AgentExecutionLoop without LangGraph."
            ) from exc

        graph = StateGraph(_KernelGraphState)

        graph.add_node(
            "run_agent",
            _run_agent_node,
        )

        graph.add_edge(START, "run_agent")
        graph.add_edge("run_agent", END)

        self._compiled = graph.compile()

    def run(
        self,
        *,
        agent: AgentCore,
        decision_engine: AgentDecisionEngine,
        max_steps: int,
    ) -> AgentLoopResult:

        final_state = self._compiled.invoke(
            {
                "agent": agent,
                "decision_engine": decision_engine,
                "max_steps": max_steps,
                "result": None,
            }
        )

        result = final_state.get("result")

        if not isinstance(result, AgentLoopResult):
            raise RuntimeError(
                "LangGraph execution finished without producing an "
                f"AgentLoopResult (got {result!r}). This means the "
                "compiled graph's state handling does not match what "
                "this engine expects -- check the installed langgraph "
                "version against requirements.txt's pin and this "
                "module's own docstring before assuming the Kernel "
                "itself is at fault."
            )

        return result
