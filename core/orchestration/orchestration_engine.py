from __future__ import annotations

from abc import ABC, abstractmethod

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


# ---------------------------------------------------------------------
# Real implementation of core/orchestration/ARCHITECTURE.md and
# EXECUTION_ENGINE.md's Orchestration layer.
#
# Both spec docs describe the same core execution model regardless of
# which concrete engine runs it:
#
#   Task -> Kernel -> Plan -> Agent Selection -> Agent Execution ->
#   Tool Execution -> Observation -> Verification -> Decision ->
#   Next Step / Completion
#
# For a single agent running a single plan to completion (the only
# case Kernel v1, core/kernel/kernel.py, currently constructs), "Agent
# Execution -> Tool Execution -> Observation" is exactly what
# AgentExecutionLoop already implements and this project has tested
# exhaustively since Pass 1. This module does not reimplement that --
# it defines OrchestrationEngine as the seam the Kernel calls through,
# so the Kernel's own logic never depends on which concrete engine
# (this project's own sequential runner, or a LangGraph graph -- see
# langgraph_orchestration_engine.py) actually drives that loop.
#
# EXECUTION_ENGINE.md states "LangGraph is the execution and
# orchestration engine of AI-BRAIN." SequentialOrchestrationEngine
# below is deliberately NOT that -- it is a plain, dependency-free
# engine that calls AgentExecutionLoop directly. It exists because
# `langgraph` (pinned in requirements.txt) cannot be installed or
# exercised in this sandbox (no package-index access -- confirmed:
# `pip download langgraph==1.2.11` returns "No matching distribution
# found", the same network isolation this project has hit before with
# Serper.dev/OpenAI/Anthropic). Shipping Kernel v1 with a hard,
# unconditional dependency on code that has never once been executed
# would contradict this project's own standing rule: nothing is
# considered done until a real `pytest` run confirms it. Instead,
# create_default_orchestration_engine() (engine_factory.py) tries the
# real LangGraphOrchestrationEngine first and falls back to this
# engine only if `langgraph` genuinely is not importable -- so on the
# user's machine, where `pip install -r requirements.txt` makes
# `langgraph` available, the Kernel really does run through it, and
# this fallback engine is exactly what keeps the Kernel able to run at
# all in an environment (like this sandbox, or any environment where
# that heavy dependency isn't installed yet) where it can't -- the
# same "must remain executable, never over-strict" principle behind
# every fail-open design choice already made in this project (e.g.
# Pass 4 finding K's HIGH-not-UNKNOWN fallback).
# ---------------------------------------------------------------------


class OrchestrationEngine(ABC):
    """
    Abstract orchestration engine.

    Drives one AgentCore through one AgentDecisionEngine to a
    terminal AgentLoopResult.

    An OrchestrationEngine does not:

    - select agents (that is the Kernel's job)
    - classify or normalize tasks
    - perform verification or approval handling (the Kernel does
      this with the AgentLoopResult this engine returns)
    - know anything about `core.kernel` -- this module has no
      import of it, by design, so the orchestration layer stays
      usable and testable independently of the Kernel.
    """

    @abstractmethod
    def run(
        self,
        *,
        agent: AgentCore,
        decision_engine: AgentDecisionEngine,
        max_steps: int,
    ) -> AgentLoopResult:
        """
        Run `agent` to completion using `decision_engine`, honoring
        `max_steps`, and return the terminal AgentLoopResult.
        """
        raise NotImplementedError


class SequentialOrchestrationEngine(OrchestrationEngine):
    """
    Default, dependency-free OrchestrationEngine.

    Runs the agent through a single, direct AgentExecutionLoop --
    exactly the same execution path every test in this project has
    exercised since Pass 1. No graph, no extra dependency, always
    available.
    """

    def run(
        self,
        *,
        agent: AgentCore,
        decision_engine: AgentDecisionEngine,
        max_steps: int,
    ) -> AgentLoopResult:

        loop = AgentExecutionLoop(
            agent=agent,
            decision_engine=decision_engine,
            max_steps=max_steps,
        )

        return loop.run()
