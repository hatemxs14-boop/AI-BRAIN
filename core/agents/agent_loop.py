from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.agents.action_validator import (
    AgentActionValidator,
)

from core.agents.agent_action import (
    AgentAction,
    AgentActionType,
)

from core.agents.agent_context import (
    AgentContext,
)

from core.agents.agent_core import (
    AgentCore,
)

from core.agents.checkpoint import (
    CheckpointStore,
    TaskCheckpoint,
)

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.agents.guardrails import (
    GuardrailFinding,
    OutputGuardrailEngine,
)

from core.llm.budget import (
    TokenBudget,
)

from core.llm.token_usage import (
    TokenUsage,
)

from core.tools.engine.tool_gateway import (
    ToolExecutionResult,
)


@dataclass(frozen=True)
class AgentLoopResult:
    """
    Final result of an Agent execution loop.

    `token_usage` (Build Phase 19) is the real, normalized token cost
    of every LLM call this one loop run made (a single run can call
    `decide()` more than once before COMPLETE/FAIL) -- see
    AgentExecutionLoop._build_result's own docstring for exactly how
    it is read, and TokenUsage's own docstring for what "real" means
    here. `None` when the decision engine in use doesn't expose usage
    at all (e.g. DeterministicDecisionEngine, a test double, or an
    `action_provider`-driven loop with no decision engine) -- never a
    fabricated zero.
    """

    status: str

    steps: int

    last_result: ToolExecutionResult | None

    reason: str | None

    context: AgentContext

    token_usage: TokenUsage | None = None

    guardrail_findings: tuple[GuardrailFinding, ...] = ()
    """
    Every Build Phase 23 GuardrailFinding produced across the whole
    run, in the order they were found -- empty whenever no
    `guardrail_engine` was configured for this loop, exactly like
    `token_usage` being `None` for a decision engine that doesn't
    expose it. Populated even for a run that was never blocked (a
    non-enforcing/flagging engine, or an enforcing engine that never
    saw a HIGH-severity finding) -- this is a full, honest audit
    trail, not just a record of the one finding (if any) that actually
    stopped the run. See core/agents/guardrails.py's own module
    docstring for what each finding does and does not claim.
    """


class AgentExecutionLoop:
    """
    Controlled execution loop for an AI-BRAIN Agent.

    Responsibilities:

    - obtain the current AgentContext
    - discover available tools
    - request the next AgentAction
    - validate the AgentAction
    - execute valid actions through AgentCore
    - record tool results
    - stop on COMPLETE
    - stop on FAIL
    - handle decision errors
    - handle validation errors
    - handle tool execution errors
    - handle approval-required results
    - enforce max_steps

    The loop does not:

    - execute tools directly
    - access executors
    - access the Security Layer directly
    - grant permissions
    - approve security requests
    - bypass AgentCore
    """

    def __init__(
        self,
        *,
        agent: AgentCore,
        decision_engine: AgentDecisionEngine | None = None,
        action_provider: Callable[
            [AgentCore],
            AgentAction,
        ] | None = None,
        max_steps: int = 10,
        action_validator: AgentActionValidator | None = None,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_id: str | None = None,
        resume_from: TaskCheckpoint | None = None,
        guardrail_engine: OutputGuardrailEngine | None = None,
        token_budget: TokenBudget | None = None,
    ) -> None:
        """
        `checkpoint_store`/`checkpoint_id`/`resume_from` (Build Phase
        22) are entirely optional and independent of one another in
        principle, but in practice: `checkpoint_id` is required
        whenever `checkpoint_store` is given (there is no default
        identity for a checkpoint), and `resume_from`, when given,
        seeds this run's starting AgentContext instead of starting
        from an empty one -- see `run()`'s own docstring for exactly
        how, and core/agents/checkpoint.py's own module docstring for
        the full design rationale and its honestly-scoped limitations.

        `guardrail_engine` (Build Phase 23) is entirely optional and
        independent of the checkpoint/resume parameters above -- see
        `run()`'s own docstring for exactly where it is consulted, and
        core/agents/guardrails.py's own module docstring for what it
        checks and why it defaults to flagging rather than blocking.

        `token_budget` (Build Phase 26) is entirely optional and
        independent of every other parameter above -- see `run()`'s
        own docstring for exactly where it is consulted, and
        core/llm/budget.py's own module docstring for what it checks,
        why (unlike `guardrail_engine`) it always enforces once
        configured, and why it can only ever be a reactive check.
        """

        if not isinstance(
            agent,
            AgentCore,
        ):
            raise TypeError(
                "agent must be an AgentCore."
            )

        if not isinstance(
            max_steps,
            int,
        ):
            raise TypeError(
                "max_steps must be an integer."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        if decision_engine is None:

            if action_provider is None:
                raise ValueError(
                    "Either decision_engine or "
                    "action_provider must be provided."
                )

            if not callable(action_provider):
                raise TypeError(
                    "action_provider must be callable."
                )

            self.decision_engine = None
            self.action_provider = action_provider

        else:

            if not isinstance(
                decision_engine,
                AgentDecisionEngine,
            ):
                raise TypeError(
                    "decision_engine must implement "
                    "AgentDecisionEngine."
                )

            self.decision_engine = decision_engine
            self.action_provider = None

        self.agent = agent
        self.max_steps = max_steps

        self.action_validator = (
            action_validator
            if action_validator is not None
            else AgentActionValidator()
        )

        if checkpoint_store is not None and not isinstance(
            checkpoint_store, CheckpointStore
        ):
            raise TypeError(
                "checkpoint_store must be a CheckpointStore."
            )

        if checkpoint_store is not None and (
            not isinstance(checkpoint_id, str) or not checkpoint_id.strip()
        ):
            raise ValueError(
                "checkpoint_id must be a non-empty string when "
                "checkpoint_store is provided."
            )

        if resume_from is not None and not isinstance(
            resume_from, TaskCheckpoint
        ):
            raise TypeError(
                "resume_from must be a TaskCheckpoint."
            )

        self.checkpoint_store = checkpoint_store
        self.checkpoint_id = checkpoint_id
        self.resume_from = resume_from

        if guardrail_engine is not None and not isinstance(
            guardrail_engine, OutputGuardrailEngine
        ):
            raise TypeError(
                "guardrail_engine must be an OutputGuardrailEngine."
            )

        self.guardrail_engine = guardrail_engine
        self._guardrail_findings: list[GuardrailFinding] = []

        if token_budget is not None and not isinstance(
            token_budget, TokenBudget
        ):
            raise TypeError(
                "token_budget must be a TokenBudget."
            )

        self.token_budget = token_budget

        self.context: AgentContext | None = None

    def run(
        self,
    ) -> AgentLoopResult:
        """
        Run the Agent execution loop.

        Build Phase 23: when `self.guardrail_engine` is configured, it
        is consulted once per step, right after the decided
        AgentAction has already passed `self.action_validator` and
        before that action is ever acted on (COMPLETE/FAIL executed,
        or a tool invoked) -- see core/agents/guardrails.py's own
        module docstring for exactly what it checks. Every finding is
        recorded on this run's `AgentLoopResult.guardrail_findings`
        regardless of outcome; only a HIGH-severity finding from an
        engine configured with `enforce=True` stops the step, via a
        new terminal status ("GUARDRAIL_BLOCKED") that mirrors
        INVALID_ACTION's own shape: `self.agent.fail_task()` is called,
        the step is never counted (this check runs before `steps` is
        incremented), and the action is never executed.

        Build Phase 26: when `self.token_budget` is configured, it is
        checked immediately after the guardrail check above (same
        "after validation, before execution, before `steps` is
        incremented" position) against
        `self.decision_engine.total_usage` -- the SAME real,
        cumulative usage `_build_result` itself reads, duck-typed via
        `getattr(..., None)` so an action-provider-driven loop or a
        decision engine that exposes no usage never raises, it simply
        never trips this check (see TokenBudget.exceeded_by's own
        docstring: `usage=None` always means "not exceeded," never a
        fabricated violation). This can only be a REACTIVE check --
        the tokens for the LLM call that just happened are already
        spent and already billed by the time `total_usage` reflects
        them -- so a configured budget stops every FURTHER step once
        the ceiling is reached, but honestly cannot prevent the one
        call that crosses it. Once tripped, this returns a new
        terminal status ("BUDGET_EXCEEDED") via the exact same shape
        as GUARDRAIL_BLOCKED: `self.agent.fail_task()`, the step
        uncounted, the action never executed.
        """

        task = getattr(
            self.agent.state,
            "task",
            None,
        )

        if not isinstance(
            task,
            str,
        ) or not task.strip():

            raise ValueError(
                "Agent must have an active task before execution."
            )

        self.context = AgentContext(
            task=task,
        )

        if self.resume_from is not None:
            self._apply_resume(task)

        self._refresh_available_tools()

        steps = self.context.step_count

        # A resumed run's `last_result` starts `None` even when real
        # tool calls happened before the interruption this checkpoint
        # is recovering from -- see core/agents/checkpoint.py's own
        # module docstring for why this is a deliberate, documented
        # limitation rather than a fabricated stand-in, and why it is
        # not a new, unverified code path.
        last_result: ToolExecutionResult | None = None

        while steps < self.max_steps:

            self._refresh_available_tools()

            try:

                action = self._get_next_action()

            except Exception as exc:

                self.agent.fail_task()

                if self.action_provider is not None:

                    return self._build_result(
                        status="INVALID_ACTION",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Action provider failed to produce "
                            f"a valid AgentAction: {exc}"
                        ),
                    )

                return self._build_result(
                    status="DECISION_ERROR",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        f"Decision engine failed: {exc}"
                    ),
                )

            if not isinstance(
                action,
                AgentAction,
            ):

                self.agent.fail_task()

                return self._build_result(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "Decision engine did not return "
                        "an AgentAction."
                    ),
                )

            try:

                validation = (
                    self.action_validator.validate(
                        action
                    )
                )

            except Exception as exc:

                self.agent.fail_task()

                return self._build_result(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "AgentActionValidator raised an "
                        f"exception: {exc}"
                    ),
                )

            if not validation.valid:

                self.agent.fail_task()

                validation_details = getattr(
                    validation,
                    "errors",
                    None,
                )

                return self._build_result(
                    status="INVALID_ACTION",
                    steps=steps,
                    last_result=last_result,
                    reason=(
                        "Agent produced an invalid action. "
                        f"errors={validation_details!r}; "
                        f"action={action!r}"
                    ),
                )

            if self.guardrail_engine is not None:

                verdict = self.guardrail_engine.evaluate(
                    action=action,
                    context=self.context,
                )

                self._guardrail_findings.extend(verdict.findings)

                if verdict.blocked:

                    self.agent.fail_task()

                    blocking_findings = "; ".join(
                        f"[{finding.severity}/{finding.rule}] "
                        f"{finding.detail}"
                        for finding in verdict.findings
                        if finding.severity == "HIGH"
                    )

                    return self._build_result(
                        status="GUARDRAIL_BLOCKED",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Blocked by output guardrails before "
                            f"execution: {blocking_findings}"
                        ),
                    )

            if self.token_budget is not None:

                current_usage = getattr(
                    self.decision_engine,
                    "total_usage",
                    None,
                )

                if self.token_budget.exceeded_by(current_usage):

                    self.agent.fail_task()

                    return self._build_result(
                        status="BUDGET_EXCEEDED",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Blocked before execution: token budget "
                            f"of {self.token_budget.max_total_tokens} "
                            "total tokens has been reached or "
                            f"exceeded (current: "
                            f"{current_usage.total_tokens if current_usage is not None else 'unknown'})."
                        ),
                    )

            steps += 1

            self.context.record_step()

            if (
                action.action_type
                == AgentActionType.COMPLETE
            ):

                try:

                    self.agent.execute_action(
                        action
                    )

                except Exception as exc:

                    self.agent.fail_task()

                    return self._build_result(
                        status="EXECUTION_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Completion action failed: "
                            f"{exc}"
                        ),
                    )

                return self._build_result(
                    status="COMPLETED",
                    steps=steps,
                    last_result=last_result,
                    reason=action.reason,
                )

            if (
                action.action_type
                == AgentActionType.FAIL
            ):

                try:

                    self.agent.execute_action(
                        action
                    )

                except Exception as exc:

                    self.agent.fail_task()

                    return self._build_result(
                        status="EXECUTION_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Failure action failed: "
                            f"{exc}"
                        ),
                    )

                return self._build_result(
                    status="FAILED",
                    steps=steps,
                    last_result=last_result,
                    reason=action.reason,
                )

            if (
                action.action_type
                == AgentActionType.INVOKE_TOOL
            ):

                try:

                    execution_result = (
                        self.agent.execute_action(
                            action
                        )
                    )

                except Exception as exc:

                    self.agent.fail_task()

                    return self._build_result(
                        status="EXECUTION_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "Tool action execution failed: "
                            f"{exc}"
                        ),
                    )

                if not isinstance(
                    execution_result,
                    ToolExecutionResult,
                ):

                    self.agent.fail_task()

                    return self._build_result(
                        status="EXECUTION_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            "AgentCore returned an invalid "
                            "tool execution result."
                        ),
                    )

                last_result = execution_result

                self.context.record_tool_result(
                    execution_result
                )

                result_status = getattr(
                    execution_result,
                    "status",
                    None,
                )

                if result_status == "APPROVAL_REQUIRED":

                    self.agent.await_approval()

                    return self._build_result(
                        status="APPROVAL_REQUIRED",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            getattr(
                                execution_result,
                                "summary",
                                None,
                            )
                            or (
                                "Tool execution requires "
                                "additional approval."
                            )
                        ),
                    )

                if result_status != "SUCCESS":

                    self.agent.fail_task()

                    return self._build_result(
                        status="TOOL_ERROR",
                        steps=steps,
                        last_result=last_result,
                        reason=(
                            getattr(
                                execution_result,
                                "summary",
                                None,
                            )
                            or "Tool execution failed."
                        ),
                    )

                self._save_checkpoint(
                    steps=steps,
                )

                continue

            self.agent.fail_task()

            return self._build_result(
                status="INVALID_ACTION",
                steps=steps,
                last_result=last_result,
                reason=(
                    "Unsupported AgentActionType."
                ),
            )

        self.agent.fail_task()

        return self._build_result(
            status="MAX_STEPS_EXCEEDED",
            steps=steps,
            last_result=last_result,
            reason=(
                "Agent execution stopped because "
                "max_steps was exceeded."
            ),
        )

    def _build_result(
        self,
        *,
        status: str,
        steps: int,
        last_result: ToolExecutionResult | None,
        reason: str | None,
    ) -> AgentLoopResult:
        """
        Build an AgentLoopResult for one of `run()`'s many exit
        points, filling in `context` and `token_usage` the same way
        every single time (Build Phase 19; the same "funnel every
        return point through one helper" pattern SecurityDecisionPoint.
        record_execution_outcome()/ToolGateway._finalize() already
        established for a return-point count problem, Build Phase 13).

        `token_usage` is read from `self.decision_engine.total_usage`
        via `getattr(..., None)` -- deliberately duck-typed, not an
        AgentDecisionEngine interface requirement: LLMDecisionEngine
        exposes it (see that class's own docstring), an
        `action_provider`-driven loop has no decision engine at all
        (`self.decision_engine is None`), and any other decision
        engine (DeterministicDecisionEngine, a test double) simply
        doesn't have the attribute -- both cases resolve to `None`
        here, never an error.

        Also deletes any Build Phase 22 checkpoint for this loop
        (no-op when checkpointing isn't configured) -- every one of
        `run()`'s exit points funnels through here, so this is the one
        place that reliably knows the loop is about to stop, for
        whatever reason. A checkpoint's only job is to survive an
        interruption WHILE the loop is still running; once `run()`
        actually returns (COMPLETED, FAILED, APPROVAL_REQUIRED, or any
        other terminal status), that status itself is the record of
        what happened -- see core/agents/checkpoint.py's own
        FileCheckpointStore docstring for why a leftover checkpoint
        file past this point would be stale, not helpful.
        """

        self._delete_checkpoint()

        return AgentLoopResult(
            status=status,
            steps=steps,
            last_result=last_result,
            reason=reason,
            context=self.context,
            token_usage=getattr(
                self.decision_engine,
                "total_usage",
                None,
            ),
            guardrail_findings=tuple(self._guardrail_findings),
        )

    def _apply_resume(
        self,
        task: str,
    ) -> None:
        """
        Seed `self.context` from `self.resume_from` (Build Phase 22)
        before the execution loop starts, so it picks up exactly where
        an earlier, interrupted run of the SAME task left off instead
        of starting from an empty AgentContext.

        Deliberately does not restore `available_tools`: the loop
        already refreshes those fresh on every single iteration
        (`_refresh_available_tools`), so trusting a possibly-stale
        checkpoint copy instead would be strictly worse, never better.
        """

        checkpoint = self.resume_from

        if checkpoint.task != task:
            raise ValueError(
                "resume_from.task does not match the task this loop "
                "was started with; refusing to resume a checkpoint "
                "for a different task."
            )

        if checkpoint.subject != self.agent.identity.subject:
            raise ValueError(
                "resume_from.subject does not match this loop's own "
                "agent; refusing to resume a checkpoint captured for "
                "a different agent."
            )

        self.context.step_count = checkpoint.step_count
        self.context.tool_results = list(checkpoint.tool_results)

    def _save_checkpoint(
        self,
        *,
        steps: int,
    ) -> None:
        """
        Persist a Build Phase 22 TaskCheckpoint after a step's tool
        call has already succeeded -- no-op when `self.checkpoint_store`
        isn't configured. Called only from the SUCCESS path of an
        INVOKE_TOOL action, right before the loop continues to its
        next iteration: that is exactly "already-completed, already-
        billed work" worth protecting, and nothing more -- an
        in-flight LLM call that hasn't returned yet has nothing to
        checkpoint.
        """

        if self.checkpoint_store is None:
            return

        checkpoint = TaskCheckpoint.from_tool_results(
            checkpoint_id=self.checkpoint_id,
            subject=self.agent.identity.subject,
            task=self.context.task,
            step_count=steps,
            tool_results=self.context.tool_results,
            last_tool_id=self.agent.state.last_tool_id,
        )

        self.checkpoint_store.save(checkpoint)

    def _delete_checkpoint(
        self,
    ) -> None:
        """
        Remove this loop's own Build Phase 22 checkpoint, if any --
        no-op when checkpointing isn't configured, and safe to call
        even when no checkpoint was ever actually saved (e.g. the loop
        completed on its very first step, or `run()` raised before any
        SUCCESS step occurred).
        """

        if self.checkpoint_store is None:
            return

        self.checkpoint_store.delete(self.checkpoint_id)

    def _refresh_available_tools(
        self,
    ) -> None:
        """
        Refresh the tools exposed to the decision engine.

        Tool discovery is performed through
        AgentToolInterface.

        Discovery does not authorize execution.
        """

        if self.context is None:
            raise RuntimeError(
                "AgentContext has not been initialized."
            )

        subject = self.agent.identity.subject

        discoveries = (
            self.agent.tools.discover_tools(
                subject
            )
        )

        available_tools = []

        for tool in discoveries:

            available_tools.append(
                {
                    "id": tool.id,
                    "name": tool.name,
                    "purpose": tool.purpose,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "resource": tool.resource,
                    "action": tool.action,
                    "scope": tool.scope,
                    "risk_level": tool.risk_level,
                }
            )

        self.context.set_available_tools(
            available_tools
        )

        self.context.set_metadata(
            "available_tools",
            available_tools,
        )

        self.context.set_metadata(
            "available_tool_ids",
            tuple(
                tool["id"]
                for tool in available_tools
            ),
        )

    def _get_next_action(
        self,
    ) -> AgentAction:

        if self.context is None:
            raise RuntimeError(
                "AgentContext has not been initialized."
            )

        if self.decision_engine is not None:

            return self.decision_engine.decide(
                self.context
            )

        if self.action_provider is None:
            raise RuntimeError(
                "No action provider is configured."
            )

        return self.action_provider(
            self.agent
        )