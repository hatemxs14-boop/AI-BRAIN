"""
Integration tests for AgentExecutionLoop's Build Phase 22 checkpoint/
resume wiring (core/agents/checkpoint.py, core/agents/agent_loop.py).

Uses the same real-tool-agent fixture style tests/agents/test_agent_loop.py
already established (a real, LOW-risk, auto-allowed "web_search" tool
behind the real Security Layer), so these tests exercise genuine
ToolExecutionResult objects flowing into a checkpoint, not stand-ins.

"Process interruption" is simulated with `_SimulatedProcessDeath`, a
BaseException (deliberately NOT an Exception) raised by a test-only
decision engine -- AgentExecutionLoop.run()'s own try/except blocks
only catch `Exception`, so this genuinely escapes `run()` uncaught,
exactly the way a real process being killed would never reach
`_build_result()` (and therefore never delete the checkpoint) either.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from core.agents.agent_action import AgentAction, AgentActionType
from core.agents.agent_context import AgentContext
from core.agents.agent_core import AgentCore, AgentIdentity
from core.agents.agent_loop import AgentExecutionLoop
from core.agents.checkpoint import FileCheckpointStore, TaskCheckpoint
from core.agents.decision_engine import AgentDecisionEngine
from core.agents.tool_interface import AgentToolInterface

from core.security.engine.security_decision import SecurityDecisionPoint

from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry
from core.tools.runtime.tool_runtime import ToolRuntime


PERMISSIONS_FILE = "core/security/schemas/permissions.json"


def _build_agent(subject: str = "research_agent") -> AgentCore:
    """
    Same real, LOW-risk, auto-allowed "web_search" tool
    tests/agents/test_agent_loop.py already uses -- reused here rather
    than duplicated with different names, so these tests exercise a
    real ToolExecutionResult (with a real SecurityDecision) the same
    way the rest of this project's own agent-loop tests do.
    """

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="web_search",
            name="Web Search",
            purpose="Search the public web.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "string"},
            permissions=(f"{subject}:web_search:search:public_web",),
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
    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(
        tool_id="web_search",
        executor=lambda query: f"RESULT: {query}",
    )

    runtime = ToolRuntime(registry=registry, gateway=gateway)
    interface = AgentToolInterface(runtime=runtime)

    identity = AgentIdentity(
        subject=subject,
        name="Research Agent",
        purpose="Research public information.",
    )

    return AgentCore(identity=identity, tools=interface)


class _SimulatedProcessDeath(BaseException):
    """
    Deliberately a BaseException, not an Exception -- see this file's
    own module docstring for why.
    """


class _ToolCallsThenDieEngine(AgentDecisionEngine):
    """
    Invokes web_search once per decision until `calls_before_death`
    real tool calls have succeeded, then raises `_SimulatedProcessDeath`
    instead of ever returning COMPLETE -- simulating the process
    itself dying right after the last already-checkpointed step.
    """

    def __init__(self, calls_before_death: int) -> None:
        self.calls_before_death = calls_before_death

    def decide(self, context: AgentContext) -> AgentAction:
        completed = len(context.tool_results)

        if completed >= self.calls_before_death:
            raise _SimulatedProcessDeath(
                "Simulated process interruption."
            )

        return AgentAction(
            action_type=AgentActionType.INVOKE_TOOL,
            tool_id="web_search",
            inputs={"query": f"{context.task} #{completed}"},
            reason="Need more research.",
        )


class _CompleteAfterEngine(AgentDecisionEngine):
    """
    Invokes web_search until `total_calls_needed` tool calls have
    succeeded (counting both restored-from-checkpoint and newly-made
    calls), then completes -- the "new process resuming a checkpoint"
    counterpart to _ToolCallsThenDieEngine.
    """

    def __init__(self, total_calls_needed: int) -> None:
        self.total_calls_needed = total_calls_needed

    def decide(self, context: AgentContext) -> AgentAction:
        completed = len(context.tool_results)

        if completed < self.total_calls_needed:
            return AgentAction(
                action_type=AgentActionType.INVOKE_TOOL,
                tool_id="web_search",
                inputs={"query": f"{context.task} #{completed}"},
                reason="Need more research.",
            )

        return AgentAction(
            action_type=AgentActionType.COMPLETE,
            reason="Research completed.",
        )


@pytest.fixture()
def checkpoint_dir():
    directory = Path(tempfile.mkdtemp())
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------


def test_checkpoint_id_required_when_checkpoint_store_given(checkpoint_dir):
    agent = _build_agent()
    agent.start_task("Research AI agents")
    store = FileCheckpointStore(checkpoint_dir)

    with pytest.raises(ValueError):
        AgentExecutionLoop(
            agent=agent,
            decision_engine=_CompleteAfterEngine(total_calls_needed=0),
            checkpoint_store=store,
        )


def test_rejects_a_non_checkpoint_store():
    agent = _build_agent()
    agent.start_task("Research AI agents")

    with pytest.raises(TypeError):
        AgentExecutionLoop(
            agent=agent,
            decision_engine=_CompleteAfterEngine(total_calls_needed=0),
            checkpoint_store="not-a-store",
            checkpoint_id="task-1",
        )


def test_rejects_a_non_task_checkpoint_resume_from():
    agent = _build_agent()
    agent.start_task("Research AI agents")

    with pytest.raises(TypeError):
        AgentExecutionLoop(
            agent=agent,
            decision_engine=_CompleteAfterEngine(total_calls_needed=0),
            resume_from={"not": "a checkpoint"},
        )


# ---------------------------------------------------------------------
# Saving progress
# ---------------------------------------------------------------------


def test_saves_a_checkpoint_after_each_successful_tool_step(checkpoint_dir):
    agent = _build_agent()
    agent.start_task("Research AI agents")
    store = FileCheckpointStore(checkpoint_dir)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_CompleteAfterEngine(total_calls_needed=2),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-1",
    )

    result = loop.run()

    assert result.status == "COMPLETED"
    # The checkpoint is deleted once the loop reaches ANY terminal
    # status -- see AgentExecutionLoop._build_result's own docstring.
    assert store.load("task-1") is None


def test_checkpoint_is_deleted_on_a_failed_run(checkpoint_dir):
    agent = _build_agent()
    agent.start_task("Research AI agents")
    store = FileCheckpointStore(checkpoint_dir)

    class _FailImmediatelyEngine(AgentDecisionEngine):
        def decide(self, context):
            return AgentAction(
                action_type=AgentActionType.FAIL,
                reason="Deliberate failure.",
            )

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_FailImmediatelyEngine(),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-1",
    )

    result = loop.run()

    assert result.status == "FAILED"
    assert store.load("task-1") is None


def test_checkpoint_survives_a_simulated_process_death(checkpoint_dir):
    agent = _build_agent()
    agent.start_task("Research AI agents")
    store = FileCheckpointStore(checkpoint_dir)

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_ToolCallsThenDieEngine(calls_before_death=2),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-1",
    )

    with pytest.raises(_SimulatedProcessDeath):
        loop.run()

    # Unlike every normal exit path, a genuine process death never
    # reaches _build_result() -- the checkpoint from the last
    # successful step must still be there.
    checkpoint = store.load("task-1")

    assert checkpoint is not None
    assert checkpoint.step_count == 2
    assert checkpoint.subject == "research_agent"
    assert checkpoint.task == "Research AI agents"
    assert len(checkpoint.tool_results) == 2
    assert all(
        entry["status"] == "SUCCESS" for entry in checkpoint.tool_results
    )
    assert checkpoint.tool_results[0]["artifacts"] == [
        "RESULT: Research AI agents #0"
    ]


# ---------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------


def test_resume_continues_to_completion_with_no_new_tool_calls_needed(
    checkpoint_dir,
):
    # First "process": dies after 2 successful tool calls.
    dying_agent = _build_agent()
    dying_agent.start_task("Research AI agents")
    store = FileCheckpointStore(checkpoint_dir)

    dying_loop = AgentExecutionLoop(
        agent=dying_agent,
        decision_engine=_ToolCallsThenDieEngine(calls_before_death=2),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-1",
    )

    with pytest.raises(_SimulatedProcessDeath):
        dying_loop.run()

    checkpoint = store.load("task-1")
    assert checkpoint is not None

    # Second "process": a brand-new agent, resuming the checkpoint,
    # already satisfied by the 2 restored tool calls.
    resumed_agent = _build_agent()
    resumed_agent.start_task("Research AI agents")

    resumed_loop = AgentExecutionLoop(
        agent=resumed_agent,
        decision_engine=_CompleteAfterEngine(total_calls_needed=2),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-1",
        resume_from=checkpoint,
    )

    result = resumed_loop.run()

    assert result.status == "COMPLETED"
    # 2 restored steps + 1 new COMPLETE decision.
    assert result.steps == 3
    # No NEW tool call happened after resume -- a documented,
    # deliberate limitation (see checkpoint.py's own module docstring),
    # not a fabricated stand-in for the real prior result.
    assert result.last_result is None
    # The resumed AgentCore's own history reflects only what happened
    # since ITS OWN start_task() -- restored progress lives in the
    # checkpoint/context, not replayed into a fresh AgentCore.state.
    assert len(resumed_agent.state.history) == 0
    assert resumed_agent.state.status == "COMPLETED"
    # Checkpoint is gone once the resumed run reaches COMPLETED.
    assert store.load("task-1") is None


def test_resume_continues_with_one_more_real_tool_call(checkpoint_dir):
    dying_agent = _build_agent()
    dying_agent.start_task("Research AI agents")
    store = FileCheckpointStore(checkpoint_dir)

    dying_loop = AgentExecutionLoop(
        agent=dying_agent,
        decision_engine=_ToolCallsThenDieEngine(calls_before_death=2),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-2",
    )

    with pytest.raises(_SimulatedProcessDeath):
        dying_loop.run()

    checkpoint = store.load("task-2")

    resumed_agent = _build_agent()
    resumed_agent.start_task("Research AI agents")

    resumed_loop = AgentExecutionLoop(
        agent=resumed_agent,
        decision_engine=_CompleteAfterEngine(total_calls_needed=3),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-2",
        resume_from=checkpoint,
    )

    result = resumed_loop.run()

    assert result.status == "COMPLETED"
    # 2 restored + 1 new tool call + 1 COMPLETE decision.
    assert result.steps == 4
    # A real NEW tool call happened after resume -- last_result is a
    # real ToolExecutionResult again, exactly like any normal run.
    assert result.last_result is not None
    assert result.last_result.status == "SUCCESS"
    assert len(resumed_agent.state.history) == 1
    assert store.load("task-2") is None


def test_resume_rejects_a_mismatched_task(checkpoint_dir):
    store = FileCheckpointStore(checkpoint_dir)
    checkpoint = TaskCheckpoint(
        checkpoint_id="task-1",
        subject="research_agent",
        task="A completely different task",
        step_count=1,
        tool_results=(
            {"status": "SUCCESS", "summary": "ok", "artifacts": []},
        ),
        last_tool_id="web_search",
    )

    agent = _build_agent()
    agent.start_task("Research AI agents")

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_CompleteAfterEngine(total_calls_needed=1),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-1",
        resume_from=checkpoint,
    )

    with pytest.raises(ValueError):
        loop.run()


def test_resume_rejects_a_mismatched_subject(checkpoint_dir):
    store = FileCheckpointStore(checkpoint_dir)
    checkpoint = TaskCheckpoint(
        checkpoint_id="task-1",
        subject="a_completely_different_agent",
        task="Research AI agents",
        step_count=1,
        tool_results=(
            {"status": "SUCCESS", "summary": "ok", "artifacts": []},
        ),
        last_tool_id="web_search",
    )

    agent = _build_agent(subject="research_agent")
    agent.start_task("Research AI agents")

    loop = AgentExecutionLoop(
        agent=agent,
        decision_engine=_CompleteAfterEngine(total_calls_needed=1),
        max_steps=10,
        checkpoint_store=store,
        checkpoint_id="task-1",
        resume_from=checkpoint,
    )

    with pytest.raises(ValueError):
        loop.run()


def test_resumed_llm_prompt_shows_the_real_restored_status():
    """
    A resumed run's `context.tool_results` is a mix of restored
    checkpoint dicts and (once new tool calls happen) real
    ToolExecutionResult objects -- LLMDecisionEngine._serialize_tool_
    result must render both into the same honest {status, summary,
    artifacts} shape an LLM would see. This is what actually makes a
    resumed run's LLM prompt trustworthy (see llm_decision_engine.py's
    own dict-passthrough addition, and tests/agents/test_checkpoint.py
    for the underlying unit test of that method itself).
    """

    from core.agents.llm_decision_engine import LLMDecisionEngine

    checkpoint = TaskCheckpoint(
        checkpoint_id="task-1",
        subject="research_agent",
        task="Research AI agents",
        step_count=1,
        tool_results=(
            {
                "status": "SUCCESS",
                "summary": "Found relevant results.",
                "artifacts": ["RESULT: Research AI agents #0"],
            },
        ),
        last_tool_id="web_search",
    )

    context = AgentContext(task=checkpoint.task)
    context.step_count = checkpoint.step_count
    context.tool_results = list(checkpoint.tool_results)

    serialized = [
        LLMDecisionEngine._serialize_tool_result(result)
        for result in context.tool_results
    ]

    assert serialized == [
        {
            "status": "SUCCESS",
            "summary": "Found relevant results.",
            "artifacts": ["RESULT: Research AI agents #0"],
        }
    ]
