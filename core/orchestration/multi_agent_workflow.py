from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypedDict

from core.agents.agent_core import AgentCore

from core.agents.agent_loop import (
    AgentExecutionLoop,
    AgentLoopResult,
)

from core.agents.decision_engine import AgentDecisionEngine

from core.agents.guardrails import OutputGuardrailEngine

from core.llm.budget import TokenBudget


# ---------------------------------------------------------------------
# Build Phase 24 -- real multi-agent workflows on top of LangGraph.
#
# core/orchestration/langgraph_orchestration_engine.py (Build Phase
# 23-adjacent work, verified for real on the user's machine on
# 2026-08-29 -- see tests/orchestration/test_langgraph_orchestration_engine.py)
# wraps a SINGLE agent's existing, unmodified AgentExecutionLoop in a
# one-node graph. That was deliberately minimal: it only proved the
# dependency wiring works.
#
# This module is the real use of LangGraph: a graph with ONE NODE PER
# AGENT, chained together (e.g. Research -> Writer -> Reviewer,
# matching the three agents already shown in the published "AI-BRAIN
# Core" architecture diagram), with an optional native human-approval
# gate (`langgraph.types.interrupt`) between any two stages.
#
# WHAT THIS DELIBERATELY DOES NOT DO: it does not break
# AgentExecutionLoop's own internal step-by-step control flow (decide
# -> validate -> guardrail check -> execute -> record) into separate
# graph nodes. That control flow already works, is already covered by
# 60+ tests, and nothing inside a single agent's own task execution
# actually branches in a way that benefits from being a graph instead
# of a Python loop. Doing that would only duplicate well-tested logic
# and risk subtle behavioral drift for zero real benefit. LangGraph's
# real value here is coordinating BETWEEN whole agents, each of which
# still runs through the exact same, unmodified AgentExecutionLoop.
#
# THE ONE SUBTLE CORRECTNESS POINT THIS DESIGN EXISTS TO GET RIGHT:
#
# LangGraph's `interrupt()` pauses a node and, on resume, RE-RUNS THAT
# NODE'S FUNCTION FROM THE TOP -- any code before the `interrupt()`
# call inside that same node runs again. If a stage's expensive,
# side-effecting work (running a whole AgentExecutionLoop -- real tool
# calls, real LLM calls) lived in the SAME node as the `interrupt()`
# call, approving or rejecting a stage would silently re-run that
# stage's agent a second time. To avoid that, every
# `requires_human_approval=True` stage is split into TWO graph nodes:
# one that does the actual agent work (never calls `interrupt`), and a
# second, cheap "gate" node that ONLY calls `interrupt()` and reads the
# decision -- so a resume only ever re-runs the cheap gate, never the
# agent's own work. See `test_workflow_resume_does_not_rerun_the_
# approved_stages_agent` for the test that exists specifically to
# catch a regression of this.
#
# IMPORTANT -- READ BEFORE TRUSTING THE INTERRUPT/RESUME PATH SPECIFICALLY:
#
# Everything in this file that mirrors langgraph_orchestration_engine.py
# (StateGraph, add_node, add_conditional_edges, START/END, compile(),
# invoke()) rests on the same already-verified ground as that file.
# The one genuinely NEW surface here -- `langgraph.checkpoint.memory.
# MemorySaver`, `langgraph.types.interrupt`, `langgraph.types.Command`,
# and reading a paused invoke() result's `"__interrupt__"` key -- has
# NOT yet been run against a real installed `langgraph` anywhere. It is
# written against that library's documented human-in-the-loop API as of
# this project's knowledge cutoff. If the installed version's exact
# shape differs (e.g. the interrupt payload lives somewhere other than
# `result["__interrupt__"][0].value`), the fix is isolated to
# `MultiAgentWorkflowEngine._build_result` and the `_make_approval_node`
# closure below -- nothing else in this file, and nothing in the Kernel
# or Agent layers, depends on that specific shape. Until the
# approval-gated tests in tests/orchestration/test_multi_agent_workflow.py
# have passed for real once, treat only that piece (not the whole
# module) as unverified.
#
# SCOPE, updated: this engine started as a new, standalone, opt-in
# capability, not wired into Kernel. It is now wired in --
# Kernel.run_multi_agent_workflow()/resume_multi_agent_workflow()
# (core/kernel/kernel.py) build a chain of WorkflowStage from already-
# registered AgentRegistrations by subject, and thread the Kernel's own
# `guardrail_engine` (Build Phase 23) into every stage it builds via
# WorkflowStage.guardrail_engine (see that field's own docstring) --
# exactly mirroring how Kernel._execute_once() threads the same
# guardrail_engine into a single agent's own AgentExecutionLoop.
# Build Phase 26 threads the Kernel's own `token_budget` the same way,
# via WorkflowStage.token_budget -- same mirroring, same call site.
# Kernel still does NOT thread checkpoint_store into a multi-agent
# workflow's own stages -- a deliberately narrower scope boundary than
# even guardrail_engine's/token_budget's, left for a future phase.
# ---------------------------------------------------------------------


class _WorkflowState(TypedDict):
    original_task: str
    stage_results: list[AgentLoopResult]
    halted: bool
    halt_reason: str | None


def _default_task_template(
    original_task: str,
    prior_results: tuple[AgentLoopResult, ...],
) -> str:
    """
    Default `WorkflowStage.task_template`: the first stage gets the
    original task verbatim; every later stage gets the original task
    plus a short summary of the immediately preceding stage's outcome,
    so each agent has real context about what came before it without
    needing to re-derive it itself.
    """

    if not prior_results:
        return original_task

    previous = prior_results[-1]

    return (
        f"{original_task}\n\n"
        f"Context from the previous stage (status={previous.status}): "
        f"{previous.reason or '(no summary provided)'}"
    )


@dataclass(frozen=True)
class WorkflowStage:
    """
    One agent's place in a multi-agent workflow.

    `build_agent`/`build_decision_engine` are zero-argument factories
    (same pattern Kernel's own AgentRegistration already uses) so a
    fresh AgentCore/AgentDecisionEngine is built for every real
    invocation of this stage's node -- never shared/reused mutable
    state across workflow runs.

    `task_template(original_task, prior_results) -> str` builds the
    exact task text this stage's agent receives; defaults to
    `_default_task_template` above. `prior_results` is every stage's
    AgentLoopResult so far, in order -- not just the immediately
    preceding one -- so a later stage can look further back if it
    needs to.

    `requires_human_approval`: when True, this stage's output must be
    explicitly approved (via `MultiAgentWorkflowEngine.resume`) before
    the workflow advances past it. See this module's own docstring for
    exactly how that is implemented without re-running the stage's
    agent on resume.
    """

    name: str
    build_agent: Callable[[], AgentCore]
    build_decision_engine: Callable[[], AgentDecisionEngine]
    task_template: Callable[
        [str, tuple[AgentLoopResult, ...]], str
    ] = _default_task_template
    max_steps: int = 10
    requires_human_approval: bool = False
    guardrail_engine: OutputGuardrailEngine | None = None
    """
    Optional, per-stage (Kernel.run_multi_agent_workflow() passes the
    Kernel's own `self.guardrail_engine` here for every stage it
    builds, exactly mirroring `_execute_once`'s identical treatment of
    a single agent -- see kernel.py's own docstring). `None` by
    default: a stage built directly (not through the Kernel) has no
    guardrail checking unless explicitly given one, the same opt-in
    convention every other optional component in this project follows.
    Deliberately does NOT cover a triggered independent-verification
    run any more than Kernel's own guardrail_engine does -- there is
    no such concept in this engine yet.
    """

    token_budget: TokenBudget | None = None
    """
    Optional, per-stage (Build Phase 26; Kernel.run_multi_agent_
    workflow() passes the Kernel's own `self.token_budget` here for
    every stage it builds, exactly mirroring how it already passes
    `self.guardrail_engine` above). `None` by default: a stage built
    directly (not through the Kernel) has no token budget unless
    explicitly given one, the same opt-in convention every other
    optional component in this project follows. Each stage's own
    AgentExecutionLoop tracks its own usage independently -- a budget
    here caps ONE stage's own spend, not the workflow's cumulative
    total across every stage that has already run.
    """

    def __post_init__(self) -> None:

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "WorkflowStage.name must be a non-empty string."
            )

        if not callable(self.build_agent):
            raise TypeError(
                "WorkflowStage.build_agent must be callable."
            )

        if not callable(self.build_decision_engine):
            raise TypeError(
                "WorkflowStage.build_decision_engine must be callable."
            )

        if not callable(self.task_template):
            raise TypeError(
                "WorkflowStage.task_template must be callable."
            )

        if (
            not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
        ):
            raise TypeError(
                "WorkflowStage.max_steps must be an integer."
            )

        if self.max_steps <= 0:
            raise ValueError(
                "WorkflowStage.max_steps must be greater than zero."
            )

        if not isinstance(self.requires_human_approval, bool):
            raise TypeError(
                "WorkflowStage.requires_human_approval must be a bool."
            )

        if self.guardrail_engine is not None and not isinstance(
            self.guardrail_engine, OutputGuardrailEngine
        ):
            raise TypeError(
                "WorkflowStage.guardrail_engine must be an "
                "OutputGuardrailEngine or None."
            )

        if self.token_budget is not None and not isinstance(
            self.token_budget, TokenBudget
        ):
            raise TypeError(
                "WorkflowStage.token_budget must be a TokenBudget "
                "or None."
            )


@dataclass(frozen=True)
class MultiAgentWorkflowResult:
    """
    Final (or paused) outcome of one `MultiAgentWorkflowEngine.run`/
    `.resume` call.

    `status`:
      - "COMPLETED": every stage ran and finished COMPLETED, in order.
      - "HALTED": some stage did not finish COMPLETED (see
        `halt_reason`), or a human reviewer rejected a stage's output
        -- the workflow stopped there and later stages never ran.
      - "AWAITING_APPROVAL": a `requires_human_approval=True` stage
        finished and is now paused waiting for a decision -- call
        `MultiAgentWorkflowEngine.resume(thread_id=..., approval=...)`
        with the SAME `thread_id` to continue it. `pending_interrupt`
        carries whatever payload the paused stage's gate published.

    `stage_results` is every stage's own AgentLoopResult that actually
    ran so far, in order -- exactly like AgentLoopResult.guardrail_
    findings, this is a full, honest record of what happened, not just
    the last thing.
    """

    status: str
    stage_results: tuple[AgentLoopResult, ...]
    halt_reason: str | None
    pending_interrupt: Any | None
    thread_id: str


def _approval_node_name(stage_name: str) -> str:
    return f"{stage_name}__approval_gate"


def _route_after_stage(state: _WorkflowState) -> str:
    return "halt" if state.get("halted") else "continue"


def _make_stage_node(stage: WorkflowStage):
    """
    Builds the node that actually runs `stage`'s agent. Never calls
    `interrupt()` -- see this module's docstring for why that matters.
    """

    def node(state: _WorkflowState) -> dict:

        task_text = stage.task_template(
            state["original_task"],
            tuple(state["stage_results"]),
        )

        agent = stage.build_agent()
        agent.start_task(task_text)

        decision_engine = stage.build_decision_engine()

        loop = AgentExecutionLoop(
            agent=agent,
            decision_engine=decision_engine,
            max_steps=stage.max_steps,
            guardrail_engine=stage.guardrail_engine,
            token_budget=stage.token_budget,
        )

        result = loop.run()

        new_results = list(state["stage_results"]) + [result]

        if result.status != "COMPLETED":
            return {
                "stage_results": new_results,
                "halted": True,
                "halt_reason": (
                    f"Stage '{stage.name}' ended with status "
                    f"'{result.status}': {result.reason}"
                ),
            }

        return {
            "stage_results": new_results,
            "halted": False,
            "halt_reason": None,
        }

    return node


def _make_approval_node(stage: WorkflowStage):
    """
    Builds the cheap gate node for a `requires_human_approval=True`
    stage. Only ever reads state that `_make_stage_node`'s node already
    wrote and calls `interrupt()` -- performs no agent work itself, so
    re-running it on resume (which is exactly what LangGraph does) is
    always safe and cheap.
    """

    from langgraph.types import interrupt

    def node(state: _WorkflowState) -> dict:

        stage_results = state["stage_results"]
        last_result = stage_results[-1] if stage_results else None

        decision = interrupt(
            {
                "stage": stage.name,
                "status": last_result.status if last_result else None,
                "reason": last_result.reason if last_result else None,
            }
        )

        if decision is True or decision in ("approve", "approved"):
            return {"halted": False, "halt_reason": None}

        return {
            "halted": True,
            "halt_reason": (
                f"Stage '{stage.name}' output was not approved "
                f"(reviewer decision: {decision!r})."
            ),
        }

    return node


class MultiAgentWorkflowEngine:
    """
    A real, compiled LangGraph StateGraph chaining multiple agents
    together, one graph node per agent (plus one cheap gate node per
    approval-gated stage). See this module's own docstring for the
    full design rationale and the one specific piece
    (interrupt/resume) that is not yet verified against a real
    installed `langgraph`.

    Raises ImportError at construction time if `langgraph` is not
    installed -- there is no silent fallback here (unlike
    engine_factory.create_default_orchestration_engine()) because a
    multi-agent workflow has no single-agent equivalent to fall back
    to.
    """

    def __init__(self, stages: Sequence[WorkflowStage]) -> None:

        if isinstance(stages, (str, bytes)) or not isinstance(
            stages, Sequence
        ):
            raise TypeError(
                "stages must be a sequence of WorkflowStage."
            )

        stages = tuple(stages)

        if not stages:
            raise ValueError("stages must not be empty.")

        for stage in stages:
            if not isinstance(stage, WorkflowStage):
                raise TypeError(
                    "Every item in stages must be a WorkflowStage."
                )

        names = [stage.name for stage in stages]

        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique.")

        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise ImportError(
                "langgraph is not installed in this environment, so "
                "MultiAgentWorkflowEngine cannot be constructed. "
                "Install it with `pip install -r requirements.txt` "
                "(it is pinned there)."
            ) from exc

        self._stages = stages

        graph = StateGraph(_WorkflowState)

        for stage in stages:
            graph.add_node(stage.name, _make_stage_node(stage))
            if stage.requires_human_approval:
                graph.add_node(
                    _approval_node_name(stage.name),
                    _make_approval_node(stage),
                )

        graph.add_edge(START, stages[0].name)

        for index, stage in enumerate(stages):

            is_last = index == len(stages) - 1
            next_name = stages[index + 1].name if not is_last else END

            if stage.requires_human_approval:
                approval_name = _approval_node_name(stage.name)

                graph.add_conditional_edges(
                    stage.name,
                    _route_after_stage,
                    {"continue": approval_name, "halt": END},
                )

                graph.add_conditional_edges(
                    approval_name,
                    _route_after_stage,
                    {"continue": next_name, "halt": END},
                )

            else:

                graph.add_conditional_edges(
                    stage.name,
                    _route_after_stage,
                    {"continue": next_name, "halt": END},
                )

        self._compiled = graph.compile(checkpointer=MemorySaver())

    def run(self, *, task: str, thread_id: str) -> MultiAgentWorkflowResult:
        """
        Start a fresh workflow run. `thread_id` identifies this run's
        checkpointed conversation -- it must be reused in `resume()` to
        continue this exact run past an approval gate; a new run should
        always use a new, unique `thread_id`.
        """

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string.")

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string.")

        config = {"configurable": {"thread_id": thread_id}}

        final_state = self._compiled.invoke(
            {
                "original_task": task,
                "stage_results": [],
                "halted": False,
                "halt_reason": None,
            },
            config=config,
        )

        return self._build_result(final_state, thread_id)

    def resume(
        self,
        *,
        thread_id: str,
        approval: Any,
    ) -> MultiAgentWorkflowResult:
        """
        Continue a run that `run()` (or an earlier `resume()`) left in
        "AWAITING_APPROVAL" for the given `thread_id`. `approval` is
        handed straight to the paused stage's gate; pass `True` (or the
        string "approve"/"approved") to let the workflow continue, or
        anything else to reject it and halt the workflow at that stage.
        """

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string.")

        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}

        final_state = self._compiled.invoke(
            Command(resume=approval),
            config=config,
        )

        return self._build_result(final_state, thread_id)

    def _build_result(
        self,
        final_state: dict,
        thread_id: str,
    ) -> MultiAgentWorkflowResult:

        pending = final_state.get("__interrupt__")

        if pending:
            raw = pending[0]
            payload = getattr(raw, "value", raw)

            return MultiAgentWorkflowResult(
                status="AWAITING_APPROVAL",
                stage_results=tuple(
                    final_state.get("stage_results", ())
                ),
                halt_reason=None,
                pending_interrupt=payload,
                thread_id=thread_id,
            )

        stage_results = tuple(final_state.get("stage_results", ()))
        halt_reason = final_state.get("halt_reason")

        return MultiAgentWorkflowResult(
            status="HALTED" if halt_reason else "COMPLETED",
            stage_results=stage_results,
            halt_reason=halt_reason,
            pending_interrupt=None,
            thread_id=thread_id,
        )
