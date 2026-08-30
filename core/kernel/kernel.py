from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from core.agents.agent_core import (
    AgentCore,
)

from core.agents.agent_loop import (
    AgentExecutionLoop,
    AgentLoopResult,
)

from core.agents.checkpoint import (
    CheckpointStore,
)

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.agents.guardrails import (
    OutputGuardrailEngine,
)

from core.llm.budget import (
    TokenBudget,
)

from core.llm.token_usage import (
    TokenUsage,
    combine_token_usage,
)

from core.memory.memory_store import (
    MemoryRecord,
    MemoryStore,
)

from core.orchestration.engine_factory import (
    create_default_orchestration_engine,
)

from core.orchestration.multi_agent_workflow import (
    MultiAgentWorkflowEngine,
    MultiAgentWorkflowResult,
    WorkflowStage,
)

from core.orchestration.orchestration_engine import (
    OrchestrationEngine,
)

from core.policies.policy_engine import (
    ExternalActionEvaluation,
    PolicyEngine,
)


# ---------------------------------------------------------------------
# Real (v1) implementation of core/kernel/KERNEL_SPEC.md.
#
# KERNEL_SPEC.md describes an 18-step lifecycle (normalize, classify,
# context retrieval, strategy selection, agent selection, model
# selection, tool selection, plan, execute, observe, recover, verify,
# human approval, persist, evaluate, learn, final result) sitting on
# top of subsystems this project doesn't fully have yet -- there is no
# memory layer and no dedicated verification/learning subsystem in
# `core/` today, only the agent/tool/security stack Passes 1-5 and
# Build Phases 1-3 hardened.
#
# Rather than fabricate hollow implementations of steps this project
# has nothing real to back yet (that would hide the actual gap behind
# code that looks done), this Kernel v1 implements every step for
# which a real foundation already exists, and represents each
# remaining step as an explicit, documented, no-op passthrough with
# its own method -- so a future phase can fill in memory/verification/
# learning without restructuring the pipeline, and so nothing here
# silently claims to do more than it does. Concretely:
#
#   NORMALIZE              real (_normalize)
#   CLASSIFY               real (_classify) -- capability-predicate
#                           matching against registered agents
#   CONTEXT RETRIEVAL       real, but opt-in (_retrieve_context): as of
#                           Build Phase 14, a real core.memory.
#                           memory_store.MemoryStore now exists (see
#                           core/memory/MEMORY_SPEC.md). When this
#                           Kernel is configured with one (the
#                           `memory_store` constructor argument,
#                           default None), _retrieve_context runs a
#                           real keyword search against it for every
#                           task and surfaces the result as
#                           KernelResult.retrieved_context -- purely
#                           additive, inspectable diagnostic data, the
#                           same shape Build Phase 7's
#                           policy_evaluation and Build Phase 12's
#                           independent_verification already
#                           established. It is never woven into the
#                           normalized task text or fed to the agent
#                           automatically: POLICY_SPEC.md's Memory
#                           Constraints ("recalled memory is untrusted
#                           context") and this module's own Sec.3
#                           ("must not treat memory as authoritative
#                           without verification") both rule that out.
#                           An unconfigured Kernel (memory_store=None,
#                           the default) behaves exactly as it always
#                           has -- this step is a no-op for every
#                           existing caller that doesn't opt in.
#   STRATEGY SELECTION      real but simple (_select_strategy) --
#                           still exactly "run one matching agent" for
#                           an ordinary Kernel.run() call; ranking
#                           multiple candidates for a single task
#                           remains future work. As of Build Phase 15,
#                           a caller who wants a real multi-agent plan
#                           instead calls the new, separate
#                           Kernel.run_workflow() -- a declarative,
#                           hand-registered WorkflowDefinition of
#                           ordered WorkflowStep entries, each reusing
#                           an already-registered agent -- rather than
#                           this step growing that choice itself. See
#                           WorkflowDefinition's own docstring below for
#                           why a separate method, not a change to
#                           _select_strategy/_select_agent, is how this
#                           project answers AGENT_REGISTRY.md's
#                           Collaboration section and this docstring's
#                           own long-standing "future phase" note.
#   AGENT SELECTION         real (_select_agent)
#   MODEL SELECTION         delegated: each AgentRegistration's
#                           build_decision_engine() factory already
#                           encapsulates which model/client is used
#                           (see build_default_kernel() below)
#   TOOL SELECTION          delegated to the Agent/Tool layer -- the
#                           Kernel must not "perform specialized
#                           domain work when an agent exists for it"
#                           (KERNEL_SPEC.md Sec.3), and tool discovery/
#                           authorization is exactly that
#   PLAN                    real (KernelPlan)
#   EXECUTE                 real: delegated to an OrchestrationEngine
#                           (core/orchestration/), never called
#                           directly -- the Kernel must not "replace
#                           the orchestration layer" (Sec.3)
#   OBSERVE                 real: the returned AgentLoopResult
#   RECOVER IF NEEDED       real but deliberately narrow (_should_
#                           recover): a bounded number of full,
#                           fresh re-attempts (new agent + new
#                           decision engine from the same
#                           registration, via Kernel.__init__'s
#                           max_recovery_attempts, default 1) ONLY for
#                           AgentLoopResult statuses that indicate
#                           something crashed unexpectedly rather than
#                           a considered outcome -- DECISION_ERROR (the
#                           decision engine raised, e.g. a transient
#                           LLM API hiccup) and EXECUTION_ERROR (a tool
#                           executor raised unexpectedly, e.g. a
#                           network blip). As of this Kernel, which
#                           status counts as recoverable is no longer a
#                           private Kernel heuristic: _should_recover
#                           asks a real core.policies.policy_engine.
#                           PolicyEngine.is_recovery_authorized() call,
#                           making POLICY_SPEC.md's Failure Policy step
#                           4 ("attempt recovery only when the recovery
#                           is itself authorized") a real, inspectable
#                           Policy Layer decision the Kernel consults,
#                           per that spec's own "Policy Enforcement"
#                           section. Never retried: FAILED (an
#                           agent's own deliberate decision -- second-
#                           guessing that is not "recovery", it's
#                           overriding the agent), TOOL_ERROR (the tool
#                           ran and reported a real result, not a
#                           crash), APPROVAL_REQUIRED (retrying
#                           wouldn't change anything -- it's still
#                           waiting on a human), MAX_STEPS_EXCEEDED/
#                           INVALID_ACTION (retrying an identical plan
#                           would just reproduce the same outcome).
#                           Every retry rebuilds the agent and decision
#                           engine from scratch via the registration's
#                           factories rather than resuming -- simpler
#                           and safer than trying to resume a possibly-
#                           partial AgentState after an unexpected
#                           exception (SYSTEM_CONSTITUTION.md's "No
#                           unnecessary complexity"). KernelResult.
#                           recovery_attempts reports exactly how many
#                           retries actually happened, so recovery is
#                           inspectable, never silent.
#   VERIFY                  real (_verify) -- see KernelVerification.
#                           Build Phase 12 added a second, purely
#                           additive verification path alongside it:
#                           _trigger_independent_verification, which
#                           can run a real second agent (reviewer_agent,
#                           via an injected WorkflowVerifierRegistration)
#                           for real, independent, content-level
#                           verification of one specific completed
#                           action -- see that method's own docstring
#                           and KernelResult.independent_verification
#                           below for exactly what this covers.
#   HUMAN APPROVAL           real: an AgentLoopResult of
#   IF REQUIRED             APPROVAL_REQUIRED is surfaced as
#                           KernelResult.status == "AWAITING_APPROVAL",
#                           never silently resolved by the Kernel
#                           itself (Sec.3: "never silently make
#                           irreversible decisions")
#   PERSIST                 delegated to the agent's own tools (e.g.
#                           research_agent's write_research_findings,
#                           Build Phase 3) -- the Kernel must not
#                           "replace the memory layer" (Sec.3). Build
#                           Phase 14's new MemoryStore.write()/verify()
#                           are real, callable operations, but no agent
#                           specification in this project currently
#                           declares a capability to write into project
#                           memory (RESEARCH_AGENT.md explicitly
#                           forbids research_agent from "promoting its
#                           own findings directly into canonical
#                           knowledge") -- so this step remains
#                           delegated, not newly performed by the
#                           Kernel itself, exactly per Sec.3. See
#                           core/memory/MEMORY_SPEC.md's own v1 Scope
#                           section for the honest list of what is and
#                           isn't wired yet.
#   EVALUATE                real (KernelResult carries status +
#                           verification + the full AgentLoopResult),
#                           and, as of Build Phase 7, also carries a
#                           real answer to POLICY_SPEC.md's External
#                           Actions six-question checklist for the
#                           last tool actually invoked, when one was
#                           (KernelResult.policy_evaluation; see
#                           _evaluate_policy). This was previously a
#                           gap Build Phase 6 named and deliberately
#                           left open -- neither SecurityDecision nor
#                           ToolExecutionResult preserved which
#                           tool_id/action produced them, so there was
#                           nothing for the Kernel to look up. Build
#                           Phase 7 closed that gap at its source (see
#                           ToolExecutionResult's own docstring in
#                           core/tools/engine/tool_gateway.py), so
#                           this is now a real, inspectable per-run
#                           answer -- not a re-derivation of risk or
#                           approval (the Security Layer already
#                           decided those; this only asserts a
#                           complete answer exists) and not a gate on
#                           execution (the Tool Gateway already
#                           enforced authorization synchronously,
#                           before this Kernel ever sees a result --
#                           see this method's own docstring for why a
#                           missing/malformed answer degrades to None
#                           instead of failing the whole run).
#   LEARN                   explicit no-op: no lesson-recording
#                           subsystem exists yet
#   FINAL RESULT            real (KernelResult)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedTask:
    """
    Result of the Kernel's NORMALIZE step.
    """

    text: str


@dataclass(frozen=True)
class RetrievedContext:
    """
    Result of the Kernel's CONTEXT RETRIEVAL step (Build Phase 14),
    when a memory_store is configured (see Kernel.__init__'s own
    docstring). `records` is exactly what MemoryStore.search()
    returned for `query` -- most-recent-first, each carrying its own
    `verified` flag.

    This is inspectable diagnostic data ONLY. Nothing in this project
    reads `records` back into the normalized task, an agent's prompt,
    or any other execution input -- POLICY_SPEC.md's Memory
    Constraints ("recalled memory is untrusted context") and this
    module's own Sec.3 ("must not treat memory as authoritative
    without verification") both require that a recalled record never
    silently become part of what an agent is told is true. A future
    phase that wants an agent to actually consult memory during its
    own reasoning should do so through a real tool call (e.g.
    research_agent's read_project_memory), which passes through the
    full Security Layer and leaves the calling agent's own decision
    engine to decide what to do with the result -- not through this
    Kernel-level field.
    """

    query: str
    records: tuple[MemoryRecord, ...]


@dataclass(frozen=True)
class TaskClassification:
    """
    Result of the Kernel's CLASSIFY step: every registered agent
    subject whose `can_handle` predicate matched the normalized
    task, in registration order.
    """

    candidate_subjects: tuple[str, ...]


@dataclass(frozen=True)
class KernelPlan:
    """
    Result of the Kernel's PLAN step: exactly what will be executed
    and by whom.
    """

    subject: str
    agent: AgentCore
    decision_engine: AgentDecisionEngine
    max_steps: int


@dataclass(frozen=True)
class KernelVerification:
    """
    Result of the Kernel's VERIFY step.

    v1 verification rule: a COMPLETED loop result is only considered
    verified if its most recent tool result (if any) actually
    succeeded -- guarding against a decision engine that issues
    COMPLETE right after a tool call failed or was denied, which
    would otherwise report success on top of a real failure.

    This is deliberately narrow: a fast, generic, always-on
    consistency check, not a content-level re-verification of any
    claim. Build Phase 11 built exactly the "dedicated verification
    subsystem (independently re-checking claims, not just consistency-
    checking the agent's own last result)" this docstring used to name
    as future work -- reviewer_agent (core/agents/reviewer_agent.py) --
    and Build Phase 12 gave the Kernel a real, optional, purely
    additive way to actually run it after a specific completed action
    (see Kernel._trigger_independent_verification and
    KernelResult.independent_verification below). This method itself
    is unchanged by either phase and still means exactly what it always
    has: it never reads reviewer_agent's own result, and reviewer_agent
    never gates or changes what this method reports.
    """

    passed: bool
    reason: str


@dataclass(frozen=True)
class KernelResult:
    """
    Final result of the Kernel's FINAL RESULT step.

    `status` values:

        NO_AGENT_AVAILABLE   no registered agent's can_handle
                              predicate matched the task; nothing was
                              executed.
        COMPLETED             the agent finished and verification
                              passed.
        VERIFICATION_FAILED   the agent reported COMPLETED but
                              verification did not pass.
        AWAITING_APPROVAL     the underlying AgentLoopResult was
                              APPROVAL_REQUIRED -- execution paused at
                              the human-approval boundary, exactly as
                              KERNEL_SPEC.md Sec.3 requires ("never
                              silently make irreversible decisions").
        (anything else)       passed through verbatim from
                              AgentLoopResult.status (FAILED,
                              TOOL_ERROR, MAX_STEPS_EXCEEDED,
                              DECISION_ERROR, INVALID_ACTION,
                              EXECUTION_ERROR) -- the Kernel does not
                              invent new vocabulary for cases the
                              execution loop already reports clearly.

    `policy_evaluation` answers POLICY_SPEC.md's External Actions
    six-question checklist for the last tool actually invoked during
    this run (see Kernel._evaluate_policy). None when no tool was
    invoked at all, or when the identifying/security data needed to
    answer those six questions was itself incomplete -- this field is
    purely inspectable diagnostic data; it never changes `status`.

    `independent_verification` (Build Phase 12) is the AgentLoopResult
    of a real second agent (reviewer_agent) the Kernel triggered after
    this run's own completed action, when POLICY_SPEC.md's Workflow
    Constraints (PolicyEngine.evaluate_workflow_trigger) named one and
    a WorkflowVerifierRegistration for that exact subject was
    configured on this Kernel -- see Kernel._trigger_independent_
    verification's own docstring for exactly when this runs. `None`
    whenever no transition was triggered, or none is configured. Like
    `policy_evaluation`, this is purely inspectable diagnostic data: it
    never changes `status`, and a reviewer_agent finding of
    unsupported claims never retroactively fails the run it's
    reviewing -- second-guessing the primary agent's own already-
    reported outcome based on a secondary, advisory agent's opinion
    would itself violate this project's standing constraint that the
    system must never become so strict it refuses to execute/accept
    something.

    `retrieved_context` (Build Phase 14) is the CONTEXT RETRIEVAL
    step's own result -- see RetrievedContext's own docstring for
    exactly what it is (inspectable-only, untrusted, never fed back
    into execution) and Kernel._retrieve_context's docstring for when
    it is real vs. `None`.

    `token_usage` (Build Phase 19) is the real, normalized total token
    cost of this ENTIRE `run()` call -- every RECOVER IF NEEDED retry
    attempt (each one a full, fresh, separately-billed attempt per
    `_execute_once`'s own docstring), plus a triggered
    `independent_verification`'s own run, if one happened. `None` only
    when nothing executed at all (`NO_AGENT_AVAILABLE`) or the
    decision engine(s) involved don't expose usage -- never a
    fabricated zero. Purely inspectable, exactly like
    `policy_evaluation`/`independent_verification`/`retrieved_context`
    -- it never changes `status` or any other field.
    """

    status: str
    subject: str | None
    loop_result: AgentLoopResult | None
    verification: KernelVerification | None
    reason: str | None
    recovery_attempts: int = 0
    policy_evaluation: ExternalActionEvaluation | None = None
    independent_verification: AgentLoopResult | None = None
    retrieved_context: RetrievedContext | None = None
    token_usage: TokenUsage | None = None


@dataclass(frozen=True)
class AgentRegistration:
    """
    Registers one agent with the Kernel.

    `build_agent` and `build_decision_engine` are factories, not
    instances -- AgentCore is stateful (AgentState.history/
    last_result/status), so the Kernel builds a fresh agent and
    decision engine for every `Kernel.run()` call rather than reusing
    one across tasks, which would otherwise leak history between
    unrelated runs.
    """

    subject: str
    description: str
    can_handle: Callable[[NormalizedTask], bool]
    build_agent: Callable[[], AgentCore]
    build_decision_engine: Callable[[], AgentDecisionEngine]

    def __post_init__(self) -> None:

        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError(
                "AgentRegistration.subject must be a non-empty string."
            )

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                "AgentRegistration.description must be a non-empty "
                "string."
            )

        if not callable(self.can_handle):
            raise TypeError(
                "AgentRegistration.can_handle must be callable."
            )

        if not callable(self.build_agent):
            raise TypeError(
                "AgentRegistration.build_agent must be callable."
            )

        if not callable(self.build_decision_engine):
            raise TypeError(
                "AgentRegistration.build_decision_engine must be "
                "callable."
            )


@dataclass(frozen=True)
class WorkflowVerifierRegistration:
    """
    Registers exactly one agent with the Kernel as its Workflow
    Constraints (Build Phase 12) independent-verification target --
    deliberately a separate, lighter dataclass from AgentRegistration,
    not a reuse of it: this agent is never selected via CLASSIFY/
    _select_agent (it has no `can_handle`/`description` -- it is only
    ever triggered by Kernel._trigger_independent_verification, per a
    real PolicyEngine.evaluate_workflow_trigger() answer, after another
    agent's own primary run already completed).

    `build_agent`/`build_decision_engine` are factories, not instances,
    for the exact same reason AgentRegistration's own fields are (see
    its own docstring): a fresh agent/decision engine is built for
    every trigger, never reused across runs.
    """

    subject: str
    build_agent: Callable[[], AgentCore]
    build_decision_engine: Callable[[], AgentDecisionEngine]

    def __post_init__(self) -> None:

        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError(
                "WorkflowVerifierRegistration.subject must be a "
                "non-empty string."
            )

        if not callable(self.build_agent):
            raise TypeError(
                "WorkflowVerifierRegistration.build_agent must be "
                "callable."
            )

        if not callable(self.build_decision_engine):
            raise TypeError(
                "WorkflowVerifierRegistration.build_decision_engine "
                "must be callable."
            )


def contains_keyword_phrase(text: str, keywords: Sequence[str]) -> bool:
    """
    True if `text` contains any of `keywords` as a whole word/phrase --
    `\\bkeyword\\b`, never a plain substring check.

    Promoted (Build Phase 16) out of core/kernel/default_kernel.py's
    own private `_contains_keyword`, which introduced this exact
    word-boundary convention in Build Phase 8/11 after plain substring
    matching produced two real misrouting bugs there (research_agent's
    "find" keyword matching inside "finding"/"findings"; writer_agent's
    "report" keyword colliding with reviewer_agent's own domain -- see
    that module's own docstring for the full history). Pulled up to
    this module -- the same "shared helper, not a second copy" move
    Build Phase 15 already made for extract_first_artifact_path below
    -- so core/kernel/workflow_config.py's config-driven `can_handle`
    predicates (Build Phase 16) use the exact same matching convention
    as every hand-written agent/workflow predicate in
    core/kernel/default_kernel.py, rather than a second, potentially
    drifting implementation. `default_kernel.py`'s own `_contains_
    keyword` now simply delegates here.

    `text` is matched as given -- callers are responsible for
    lowercasing it first if that's the intended comparison (every
    existing caller in this project normalizes to lowercase before
    calling this function).
    """

    return any(
        re.search(r"\b" + re.escape(keyword) + r"\b", text)
        for keyword in keywords
    )


def extract_first_artifact_path(loop_result: AgentLoopResult) -> str | None:
    """
    Shared helper (Build Phase 15): the path of the first artifact
    `loop_result.last_result` reports, if any.

    This is exactly the duck-typed artifact-extraction logic
    Kernel._trigger_independent_verification introduced in Build Phase
    12 (handling both a dict artifact's `.get("path")` and an object
    artifact's `getattr(..., "path", None)`), pulled out into its own
    module-level function so Build Phase 15's WorkflowStep task
    builders (see core/kernel/default_kernel.py) can reuse the exact
    same convention rather than maintaining a second, subtly different
    copy of it. `_trigger_independent_verification` itself now calls
    this helper too, instead of the inline block it used to have.

    Returns `None` -- never raises -- whenever `loop_result.last_result`
    is `None`, carries no artifacts, or the first artifact carries no
    usable non-empty string `path`. This function only ever describes
    what is actually available; a caller that cannot proceed without an
    artifact (e.g. a WorkflowStep.build_task building the next step's
    task text) is responsible for raising its own error from a `None`
    result -- see WorkflowStep's own docstring below for exactly how
    Kernel.run_workflow() handles that.
    """

    last_result = loop_result.last_result

    if last_result is None:
        return None

    artifacts = getattr(last_result, "artifacts", None) or ()

    if not artifacts:
        return None

    first_artifact = artifacts[0]

    if isinstance(first_artifact, dict):
        path = first_artifact.get("path")
    else:
        path = getattr(first_artifact, "path", None)

    if not isinstance(path, str) or not path.strip():
        return None

    return path


@dataclass(frozen=True)
class WorkflowStep:
    """
    One step of a WorkflowDefinition (Build Phase 15): which agent
    runs, and how to build the task text it runs with.

    `subject` must name an agent already registered with this Kernel
    via register_agent() -- Kernel.register_workflow() enforces this
    eagerly, at registration time (see its own docstring). Deliberately
    NOT its own build_agent/build_decision_engine pair (unlike
    AgentRegistration and WorkflowVerifierRegistration): a workflow
    step always runs through the SAME registration ordinary CLASSIFY/
    _select_agent routing would use for that subject, so a workflow can
    never run an agent under different tools, permissions, or model
    wiring than that same agent would run under if invoked directly --
    POLICY_SPEC.md's Agent Constraints apply identically either way.

    `build_task` builds this step's task text. Called with
    (the workflow's own original task text, the PREVIOUS step's
    AgentLoopResult -- `None` for the first step, since there is no
    previous result yet) and must return a non-empty string. Mirrors
    the established `f"Review {path}."` pattern
    _trigger_independent_verification (Build Phase 12) already uses to
    build a follow-on task from a prior step's artifact -- see
    extract_first_artifact_path above, which every WorkflowStep in this
    project should use for that extraction rather than re-deriving it.
    `build_task` is free to raise (e.g. when the previous result
    carries no usable artifact to build this step's task from);
    Kernel.run_workflow() catches that and reports it as
    WorkflowRunResult.status == "STEP_TASK_BUILD_ERROR" rather than
    letting it propagate out of an otherwise-normal Kernel call.
    """

    subject: str
    build_task: Callable[[str, AgentLoopResult | None], str]

    def __post_init__(self) -> None:

        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError(
                "WorkflowStep.subject must be a non-empty string."
            )

        if not callable(self.build_task):
            raise TypeError("WorkflowStep.build_task must be callable.")


@dataclass(frozen=True)
class WorkflowDefinition:
    """
    Registers one named, declarative, multi-step workflow with the
    Kernel (Build Phase 15): a real answer to AGENT_REGISTRY.md's own
    Collaboration section ("Multiple agents may be selected when the
    task contains independent domains... parallel execution reduces
    total execution time... specialized expertise is required") and to
    Kernel._select_strategy's own long-standing docstring, which has
    named "choosing a multi-agent plan" as an intentional future
    extension point since before this phase existed.

    This is deliberately a hand-registered, declarative sequence, not a
    free-text planner: this project has no natural-language planning
    subsystem, and fabricating one here would hide that gap behind code
    that looks like it does more than it does (the same honesty
    standard every prior Build Phase has held to). What this phase adds
    is real: given one instruction that matches a workflow's own
    `can_handle`, the Kernel now runs a whole named, ordered sequence of
    already-registered agents end-to-end via Kernel.run_workflow(),
    instead of a caller having to invoke each agent by hand and wire
    the hand-off itself.

    `can_handle` is evaluated only by Kernel.run_workflow() -- entirely
    separate from Kernel.run()'s own CLASSIFY/_select_agent routing, so
    registering a workflow can never change which single agent an
    ordinary Kernel.run() call selects for any existing task, and a
    workflow's own vocabulary can never collide with an agent's (they
    are matched by different methods entirely). A caller decides
    up front whether it wants ordinary single-agent routing (Kernel.run)
    or a named multi-step workflow (Kernel.run_workflow).

    `steps` must contain at least two WorkflowStep entries -- a
    single-step "workflow" is just Kernel.run() under a different name,
    and this project already has that method.
    """

    name: str
    description: str
    can_handle: Callable[[NormalizedTask], bool]
    steps: tuple[WorkflowStep, ...]

    def __post_init__(self) -> None:

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "WorkflowDefinition.name must be a non-empty string."
            )

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                "WorkflowDefinition.description must be a non-empty "
                "string."
            )

        if not callable(self.can_handle):
            raise TypeError(
                "WorkflowDefinition.can_handle must be callable."
            )

        if not isinstance(self.steps, tuple) or not all(
            isinstance(step, WorkflowStep) for step in self.steps
        ):
            raise TypeError(
                "WorkflowDefinition.steps must be a tuple of "
                "WorkflowStep."
            )

        if len(self.steps) < 2:
            raise ValueError(
                "WorkflowDefinition.steps must contain at least two "
                "steps -- a single-step workflow is just Kernel.run()."
            )


@dataclass(frozen=True)
class WorkflowStepResult:
    """
    One completed (or halted-at) step of a Kernel.run_workflow() run.

    `token_usage` (Build Phase 19) is this step's OWN total token
    cost, across every RECOVER IF NEEDED retry attempt this one step
    made (see run_workflow()'s own step loop) -- deliberately a
    separate field from `loop_result.token_usage`, which only ever
    reflects the single final attempt kept as `loop_result`. `None`
    when nothing was billed or the decision engine doesn't expose
    usage.
    """

    subject: str
    loop_result: AgentLoopResult
    verification: KernelVerification
    token_usage: TokenUsage | None = None


@dataclass(frozen=True)
class WorkflowRunResult:
    """
    Final result of Kernel.run_workflow() (Build Phase 15).

    `status` values:

        NO_WORKFLOW_AVAILABLE   no registered WorkflowDefinition's
                                  can_handle predicate matched the
                                  task; nothing was executed.
        COMPLETED                every step ran, reported COMPLETED,
                                  and passed verification.
        AWAITING_APPROVAL        a step's own AgentLoopResult was
                                  APPROVAL_REQUIRED -- the workflow
                                  stops here, exactly like Kernel.run()
                                  itself never silently resolves an
                                  approval gate (KERNEL_SPEC.md Sec.3:
                                  "never silently make irreversible
                                  decisions"). Automating a sequence of
                                  agents must never bypass an approval
                                  gate any single one of them would
                                  otherwise stop at.
        VERIFICATION_FAILED      a step reported COMPLETED but did not
                                  pass verification -- the workflow
                                  stops here rather than feeding an
                                  unverified result into the next step.
        STEP_TASK_BUILD_ERROR    a step's own `build_task` raised (most
                                  often: the previous step's result
                                  carried no usable artifact to build
                                  this step's task from).
        (anything else)          passed through verbatim from the
                                  halting step's own AgentLoopResult.
                                  status (FAILED, TOOL_ERROR,
                                  MAX_STEPS_EXCEEDED, DECISION_ERROR,
                                  INVALID_ACTION, EXECUTION_ERROR) --
                                  the same "don't invent new vocabulary"
                                  rule KernelResult's own docstring
                                  already states.

    `completed_steps` is every step that actually ran, in order,
    including whichever step halted the workflow (that step's own
    `loop_result`/`verification` show exactly why). Per-step recovery
    (RECOVER IF NEEDED) is still applied to each individual step exactly
    as Kernel.run() applies it to a standalone task; there is no
    additional whole-workflow-level retry -- deliberately deferred as
    future work, so this phase does not overstate what it does.

    `token_usage` (Build Phase 19) is the real, normalized total token
    cost of every step that actually ran (summed from each step's own
    `WorkflowStepResult.token_usage`, which already includes that
    step's own retries) -- including the halting step, since it was
    billed too even though the workflow didn't complete. `None` only
    when no step ran at all (`NO_WORKFLOW_AVAILABLE`) or none of the
    decision engines involved exposed usage.
    """

    status: str
    workflow_name: str
    completed_steps: tuple[WorkflowStepResult, ...]
    reason: str | None
    token_usage: TokenUsage | None = None


class Kernel:
    """
    Real (v1) implementation of the Brain Kernel described in
    core/kernel/KERNEL_SPEC.md.

    The Kernel does not:

    - execute tools directly (delegated to AgentCore/AgentToolInterface)
    - access the Security Layer directly (delegated to the same)
    - replace the orchestration layer (delegated to OrchestrationEngine)
    - replace specialized agents (delegated to registered agents)
    - replace the memory layer (a real one now exists as of Build
      Phase 14 -- core/memory/memory_store.py -- but the Kernel only
      ever reads from it via _retrieve_context, opt-in and inspectable
      only; see this module's own docstring)
    - silently resolve an approval requirement

    `max_recovery_attempts` bounds RECOVER IF NEEDED (see this
    module's own docstring): the number of additional, fresh attempts
    made after an initial DECISION_ERROR/EXECUTION_ERROR before giving
    up and reporting it. Defaults to 1. Pass 0 to disable recovery
    entirely -- the first attempt's result is always returned as-is.

    `policy_engine` is the real (v1) core/policies/POLICY_SPEC.md
    implementation (core.policies.policy_engine.PolicyEngine) this
    Kernel consults for RECOVER IF NEEDED's authorization decision
    (see _should_recover), for answering POLICY_SPEC.md's External
    Actions six questions about the last tool actually invoked (see
    _evaluate_policy and KernelResult.policy_evaluation, real since
    Build Phase 7), and, as of Build Phase 12, for Workflow Constraints
    -- whether a completed action should trigger a second, specific
    agent next (see _trigger_independent_verification and
    KernelResult.independent_verification). Defaults to a fresh
    PolicyEngine(); injected mainly for tests that want to substitute
    or inspect it.

    `independent_verifier` (Build Phase 12) is an optional
    WorkflowVerifierRegistration naming exactly one agent this Kernel
    may trigger for real, independent, content-level verification of a
    specific completed action (currently: reviewer_agent, after a
    SUCCESSful writer_agent write_report call -- see
    core/policies/policy_engine.py's own docstring, WORKFLOW
    CONSTRAINTS, for the exact declared transition). Defaults to
    `None`, in which case _trigger_independent_verification always
    returns `None` and KernelResult.independent_verification is always
    `None` -- this is a purely opt-in, additive capability: an
    unconfigured Kernel behaves exactly as it did before Build Phase
    12, and this project's build_default_kernel() (core/kernel/
    default_kernel.py) only configures it when a caller explicitly
    asks for it, so no existing caller's behavior, cost, or test counts
    change unless they opt in.

    `memory_store` (Build Phase 14) is an optional
    core.memory.memory_store.MemoryStore this Kernel may query during
    CONTEXT RETRIEVAL (see _retrieve_context and RetrievedContext's
    own docstrings). Defaults to `None`, in which case
    _retrieve_context always returns `None` and
    KernelResult.retrieved_context is always `None` -- the exact same
    opt-in, additive shape `independent_verifier` already established
    above: an unconfigured Kernel behaves exactly as it did before
    Build Phase 14.

    `context_retrieval_limit` bounds how many memory records a single
    CONTEXT RETRIEVAL query may return (passed straight through to
    MemoryStore.search()'s own `limit`). Defaults to 5 -- enough to be
    useful without one task's retrieval becoming unbounded; only
    meaningful when `memory_store` is configured.

    `guardrail_engine` (Build Phase 23) is an optional
    core.agents.guardrails.OutputGuardrailEngine this Kernel threads
    through to every AgentExecutionLoop it drives for the PRIMARY
    task-executing agent -- both a fresh `run()`/`run_workflow()` step
    attempt and a `resume()`d one (see _execute_once's own docstring
    for why configuring this, like `checkpoint_store`, means bypassing
    the pluggable OrchestrationEngine seam). Defaults to `None`, in
    which case AgentExecutionLoop is built with no guardrail engine at
    all and behaves exactly as it did before Build Phase 23 -- the
    same opt-in, additive shape every optional Kernel component above
    already established.

    Deliberately, honestly narrower than "every agent this Kernel ever
    runs": a triggered `_trigger_independent_verification` run (the
    reviewer_agent case described above) still goes through
    `self.orchestration_engine.run()` directly and is not guardrail-
    checked. This mirrors `checkpoint_store`'s own identical, already-
    accepted scope boundary (that call site is not threaded with a
    checkpoint either) rather than introducing a new one -- widening
    either concern to cover that secondary, optional run is future
    work, not a silently-assumed part of this phase.

    `token_budget` (Build Phase 26) is an optional core.llm.budget.
    TokenBudget this Kernel threads through to every AgentExecutionLoop
    it drives for the PRIMARY task-executing agent -- exactly the same
    scope `guardrail_engine` (Build Phase 23) already established
    immediately above, right down to the same honest exclusion of a
    triggered `_trigger_independent_verification` run. Defaults to
    `None`, in which case AgentExecutionLoop is built with no token
    budget at all and behaves exactly as it did before this phase --
    the same opt-in, additive shape every optional Kernel component
    already established. See core/llm/budget.py's own module docstring
    for why, unlike `guardrail_engine`, a configured `token_budget`
    always enforces (there is no separate "observe only" mode for it).

    `run_multi_agent_workflow`/`resume_multi_agent_workflow` (Build
    Phase 25) chain already-registered agents into a real, compiled
    LangGraph graph (core.orchestration.multi_agent_workflow's
    MultiAgentWorkflowEngine, Build Phase 24) with a genuine
    cross-call pause/resume for human-approval gates -- see that pair
    of methods' own docstrings for exactly how this differs from both
    `run()` and `run_workflow()`. `self._multi_agent_engines` keeps
    one live engine per currently-paused run, keyed by its own
    `thread_id`, purely in-memory (no persistence across a process
    restart -- MultiAgentWorkflowEngine's own MemorySaver checkpointer
    is itself in-memory-only, see that module's docstring).
    """

    def __init__(
        self,
        *,
        orchestration_engine: OrchestrationEngine | None = None,
        max_recovery_attempts: int = 1,
        policy_engine: PolicyEngine | None = None,
        independent_verifier: WorkflowVerifierRegistration | None = None,
        memory_store: MemoryStore | None = None,
        context_retrieval_limit: int = 5,
        guardrail_engine: OutputGuardrailEngine | None = None,
        token_budget: TokenBudget | None = None,
    ) -> None:

        if not isinstance(max_recovery_attempts, int):
            raise TypeError("max_recovery_attempts must be an integer.")

        if max_recovery_attempts < 0:
            raise ValueError(
                "max_recovery_attempts must be zero or greater."
            )

        self._registrations: list[AgentRegistration] = []
        self._workflows: list[WorkflowDefinition] = []

        # Build Phase 25: one MultiAgentWorkflowEngine per in-flight
        # (i.e. currently AWAITING_APPROVAL) multi-agent workflow run,
        # keyed by that run's own thread_id -- kept around so
        # resume_multi_agent_workflow() can continue the SAME compiled
        # graph's SAME in-memory checkpoint (MemorySaver) rather than
        # a fresh one that would have no memory of the paused run. See
        # run_multi_agent_workflow()'s own docstring for the full
        # lifecycle (an engine is removed here once its run reaches a
        # terminal COMPLETED/HALTED status, exactly like Build Phase
        # 22's FileCheckpointStore deletes a checkpoint once its loop
        # actually returns).
        self._multi_agent_engines: dict[str, MultiAgentWorkflowEngine] = {}

        self.max_recovery_attempts = max_recovery_attempts

        self.orchestration_engine = (
            orchestration_engine
            if orchestration_engine is not None
            else create_default_orchestration_engine()
        )

        self.policy_engine = (
            policy_engine
            if policy_engine is not None
            else PolicyEngine()
        )

        if independent_verifier is not None and not isinstance(
            independent_verifier, WorkflowVerifierRegistration
        ):
            raise TypeError(
                "independent_verifier must be a "
                "WorkflowVerifierRegistration or None."
            )

        self.independent_verifier = independent_verifier

        if memory_store is not None and not isinstance(
            memory_store, MemoryStore
        ):
            raise TypeError(
                "memory_store must be a MemoryStore or None."
            )

        if not isinstance(context_retrieval_limit, int):
            raise TypeError("context_retrieval_limit must be an integer.")

        if context_retrieval_limit <= 0:
            raise ValueError(
                "context_retrieval_limit must be a positive integer."
            )

        self.memory_store = memory_store
        self.context_retrieval_limit = context_retrieval_limit

        if guardrail_engine is not None and not isinstance(
            guardrail_engine, OutputGuardrailEngine
        ):
            raise TypeError(
                "guardrail_engine must be an OutputGuardrailEngine "
                "or None."
            )

        self.guardrail_engine = guardrail_engine

        if token_budget is not None and not isinstance(
            token_budget, TokenBudget
        ):
            raise TypeError(
                "token_budget must be a TokenBudget or None."
            )

        self.token_budget = token_budget

    def register_agent(
        self,
        registration: AgentRegistration,
    ) -> None:
        """
        Register one agent with the Kernel.

        Raises ValueError if an agent is already registered for the
        same subject -- registrations are not silently overwritten.
        """

        if not isinstance(registration, AgentRegistration):
            raise TypeError(
                "registration must be an AgentRegistration."
            )

        for existing in self._registrations:
            if existing.subject == registration.subject:
                raise ValueError(
                    "An agent is already registered for subject "
                    f"{registration.subject!r}."
                )

        self._registrations.append(registration)

    def register_workflow(
        self,
        workflow: WorkflowDefinition,
    ) -> None:
        """
        Register one WorkflowDefinition with the Kernel (Build Phase
        15), for later selection by Kernel.run_workflow().

        Raises ValueError if a workflow is already registered under
        the same name -- registrations are not silently overwritten,
        mirroring register_agent()'s own duplicate-name rejection.

        Raises ValueError if any of `workflow.steps` names a subject
        with no matching AgentRegistration already registered on this
        Kernel via register_agent(). A workflow step always reuses an
        already-registered agent (see WorkflowStep's own docstring for
        why); naming an unregistered subject is a genuine build-time
        misconfiguration, and this catches it here, at registration
        time, rather than only the first time Kernel.run_workflow()
        actually reaches that step. This is why
        core/kernel/default_kernel.py's build_default_kernel() always
        registers every agent a workflow depends on before registering
        the workflow itself.
        """

        if not isinstance(workflow, WorkflowDefinition):
            raise TypeError("workflow must be a WorkflowDefinition.")

        for existing in self._workflows:
            if existing.name == workflow.name:
                raise ValueError(
                    "A workflow is already registered under the name "
                    f"{workflow.name!r}."
                )

        registered_subjects = {
            registration.subject for registration in self._registrations
        }

        for step in workflow.steps:
            if step.subject not in registered_subjects:
                raise ValueError(
                    f"WorkflowDefinition {workflow.name!r} names "
                    f"subject {step.subject!r} in one of its steps, "
                    "but no agent is registered for that subject on "
                    "this Kernel. Register every agent a workflow "
                    "depends on before registering the workflow "
                    "itself."
                )

        self._workflows.append(workflow)

    def run(
        self,
        task: str,
        *,
        max_steps: int = 10,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_id: str | None = None,
    ) -> KernelResult:
        """
        Run the full Kernel lifecycle for one objective, from
        NORMALIZE through FINAL RESULT.

        `checkpoint_store`/`checkpoint_id` (Build Phase 22) are
        entirely optional -- when `checkpoint_store` is `None` (the
        default), this call behaves exactly as it did before this
        phase existed. When provided, progress is durably saved after
        every successfully completed step, so a later process that
        never got the chance to see this call return can pick the task
        back up with `Kernel.resume()` instead of starting over -- see
        core/agents/checkpoint.py's own module docstring for the full
        design and its honestly-scoped limitations. `checkpoint_id`
        identifies the checkpoint; it is the caller's responsibility to
        keep it unique per logical task.
        """

        if checkpoint_store is not None and (
            not isinstance(checkpoint_id, str) or not checkpoint_id.strip()
        ):
            raise ValueError(
                "checkpoint_id must be a non-empty string when "
                "checkpoint_store is provided."
            )

        normalized = self._normalize(task)

        retrieved_context = self._retrieve_context(normalized)

        classification = self._classify(normalized)

        if not classification.candidate_subjects:
            return KernelResult(
                status="NO_AGENT_AVAILABLE",
                subject=None,
                loop_result=None,
                verification=None,
                reason=(
                    "No registered agent's can_handle predicate "
                    "matched this task."
                ),
                retrieved_context=retrieved_context,
            )

        registration = self._select_agent(
            self._select_strategy(classification)
        )

        recovery_attempts = 0
        loop_result = self._execute_once(
            registration=registration,
            normalized=normalized,
            max_steps=max_steps,
            checkpoint_store=checkpoint_store,
            checkpoint_id=checkpoint_id,
        )

        # Every attempt is a full, fresh, separately-billed run
        # (_execute_once's own docstring: "never a resume") -- so a
        # retry's own token cost is accumulated here as it happens,
        # not read off only the last attempt kept as `loop_result`.
        # A checkpoint carries across retries by `checkpoint_id`
        # alone: each fresh attempt starts checkpointing from step 0
        # again under the SAME id, so the file always reflects
        # whichever attempt is currently running, never a stale one.
        token_usage = loop_result.token_usage

        while (
            recovery_attempts < self.max_recovery_attempts
            and self._should_recover(loop_result)
        ):
            recovery_attempts += 1

            loop_result = self._execute_once(
                registration=registration,
                normalized=normalized,
                max_steps=max_steps,
                checkpoint_store=checkpoint_store,
                checkpoint_id=checkpoint_id,
            )

            token_usage = combine_token_usage(
                token_usage,
                loop_result.token_usage,
            )

        verification = self._verify(loop_result)

        policy_evaluation = self._evaluate_policy(loop_result)

        independent_verification = self._trigger_independent_verification(
            completed_subject=registration.subject,
            loop_result=loop_result,
            max_steps=max_steps,
        )

        # A triggered independent verification is a real, separate
        # agent run with its own real cost -- folded into this same
        # total rather than left invisible just because it was
        # automatic.
        token_usage = combine_token_usage(
            token_usage,
            (
                independent_verification.token_usage
                if independent_verification is not None
                else None
            ),
        )

        self._learn(loop_result, verification)

        status = self._final_status(loop_result, verification)

        return KernelResult(
            status=status,
            subject=registration.subject,
            loop_result=loop_result,
            verification=verification,
            reason=loop_result.reason,
            recovery_attempts=recovery_attempts,
            policy_evaluation=policy_evaluation,
            independent_verification=independent_verification,
            retrieved_context=retrieved_context,
            token_usage=token_usage,
        )

    def resume(
        self,
        checkpoint_id: str,
        *,
        checkpoint_store: CheckpointStore,
        max_steps: int = 10,
    ) -> KernelResult:
        """
        Resume a Build Phase 22 checkpoint left behind by a
        `Kernel.run()` call whose PROCESS stopped (crashed, was
        killed, or the environment restarted) before that call could
        return -- not a run that returned with a recoverable error
        inside the same still-running process (that is RECOVER IF
        NEEDED's job, `_should_recover`/`max_recovery_attempts`,
        completely unaffected by this method).

        Builds a fresh agent/decision-engine from the SAME
        AgentRegistration the original run used (looked up by the
        checkpoint's own `subject`), seeds the execution loop from the
        checkpoint instead of an empty AgentContext, and runs it
        through to a terminal AgentLoopResult exactly the way `run()`
        does for its own single `_execute_once` attempt -- including
        VERIFY, the policy evaluation, and a possible triggered
        independent verification.

        Deliberately narrower than `run()` in two documented ways: no
        NORMALIZE/CLASSIFY/AGENT SELECTION (the checkpoint already
        names the exact subject/task the original run resolved those
        to), and no RECOVER IF NEEDED retry loop around the resumed
        attempt itself -- if the resumed attempt comes back with a
        recoverable status, this returns that result as-is rather than
        silently re-attempting; a caller can call `resume()` again
        (the checkpoint will simply reflect wherever that attempt got
        to) or accept the terminal state. This is a real, honestly-
        scoped v1 boundary, not a silently-claimed one -- see
        core/agents/checkpoint.py's own module docstring for the rest
        of this feature's scope.
        """

        if not isinstance(checkpoint_store, CheckpointStore):
            raise TypeError(
                "checkpoint_store must be a CheckpointStore."
            )

        checkpoint = checkpoint_store.load(checkpoint_id)

        if checkpoint is None:
            raise ValueError(
                f"No checkpoint found for checkpoint_id={checkpoint_id!r}; "
                "nothing to resume."
            )

        registration = None

        for candidate in self._registrations:
            if candidate.subject == checkpoint.subject:
                registration = candidate
                break

        if registration is None:
            raise ValueError(
                f"No agent is registered for subject={checkpoint.subject!r}; "
                "cannot resume this checkpoint against this Kernel."
            )

        normalized = self._normalize(checkpoint.task)

        retrieved_context = self._retrieve_context(normalized)

        agent = registration.build_agent()

        if not isinstance(agent, AgentCore):
            raise TypeError(
                "AgentRegistration.build_agent() must return an "
                "AgentCore."
            )

        decision_engine = registration.build_decision_engine()

        agent.start_task(normalized.text)

        loop = AgentExecutionLoop(
            agent=agent,
            decision_engine=decision_engine,
            max_steps=max_steps,
            resume_from=checkpoint,
            checkpoint_store=checkpoint_store,
            checkpoint_id=checkpoint.checkpoint_id,
            guardrail_engine=self.guardrail_engine,
            token_budget=self.token_budget,
        )

        loop_result = loop.run()

        verification = self._verify(loop_result)

        policy_evaluation = self._evaluate_policy(loop_result)

        independent_verification = self._trigger_independent_verification(
            completed_subject=registration.subject,
            loop_result=loop_result,
            max_steps=max_steps,
        )

        token_usage = combine_token_usage(
            loop_result.token_usage,
            (
                independent_verification.token_usage
                if independent_verification is not None
                else None
            ),
        )

        self._learn(loop_result, verification)

        status = self._final_status(loop_result, verification)

        return KernelResult(
            status=status,
            subject=registration.subject,
            loop_result=loop_result,
            verification=verification,
            reason=loop_result.reason,
            recovery_attempts=0,
            policy_evaluation=policy_evaluation,
            independent_verification=independent_verification,
            retrieved_context=retrieved_context,
            token_usage=token_usage,
        )

    def run_workflow(
        self,
        task: str,
        *,
        max_steps: int = 10,
    ) -> WorkflowRunResult:
        """
        Run a registered, declarative, multi-step WorkflowDefinition
        for one objective (Build Phase 15) -- see WorkflowDefinition's
        own docstring for what this is and is not, and
        WorkflowRunResult's own docstring for the full status
        vocabulary.

        Entirely separate from Kernel.run()'s own NORMALIZE/CLASSIFY/
        AGENT SELECTION pipeline: this selects a WorkflowDefinition
        (first registered workflow whose `can_handle` matches, mirroring
        _select_agent's own first-match-in-registration-order rule), not
        a single agent, and a caller decides up front which of the two
        methods it wants. Registering a workflow can never change what
        an ordinary Kernel.run() call does for any task, and vice versa.

        Each step is executed exactly the way Kernel.run() executes its
        own single agent -- a fresh PLAN built from that step's
        registration, run through the SAME OrchestrationEngine, with
        the SAME RECOVER IF NEEDED retry policy (`self.
        max_recovery_attempts`, `self._should_recover`) applied
        individually to that one step. There is no additional
        whole-workflow-level retry: if a step exhausts its own recovery
        attempts and still doesn't complete, the workflow stops there
        (see below) -- deliberately deferred as future work rather than
        silently claimed here.

        The workflow STOPS at the first step whose result is not a
        verified COMPLETED, in this order of precedence:

          1. APPROVAL_REQUIRED  -> WorkflowRunResult.status
                                    "AWAITING_APPROVAL". Never silently
                                    resolved, never skipped, exactly
                                    like Kernel.run() itself
                                    (KERNEL_SPEC.md Sec.3).
          2. any other non-COMPLETED loop status (FAILED, TOOL_ERROR,
             MAX_STEPS_EXCEEDED, DECISION_ERROR, INVALID_ACTION,
             EXECUTION_ERROR) -> passed through verbatim as
             WorkflowRunResult.status, mirroring KernelResult's own
             "don't invent new vocabulary" rule.
          3. COMPLETED but verification does not pass -> "
             VERIFICATION_FAILED" -- the unverified result is never fed
             into the next step's `build_task`.

        Before a step even runs, its own `build_task` is called (with
        the workflow's original task text and the previous step's
        AgentLoopResult, `None` for the first step) to build that
        step's task text; if `build_task` raises, the workflow stops
        with WorkflowRunResult.status == "STEP_TASK_BUILD_ERROR" and
        that step's own attempt is never made -- there is nothing to
        execute without a task.

        Returns WorkflowRunResult.status == "NO_WORKFLOW_AVAILABLE",
        with an empty `completed_steps`, when no registered
        WorkflowDefinition's `can_handle` matched -- mirroring
        Kernel.run()'s own "NO_AGENT_AVAILABLE" precedent exactly:
        nothing was executed, and this is not itself an error.
        """

        normalized = self._normalize(task)

        workflow = self._select_workflow(normalized)

        if workflow is None:
            return WorkflowRunResult(
                status="NO_WORKFLOW_AVAILABLE",
                workflow_name="",
                completed_steps=(),
                reason=(
                    "No registered workflow's can_handle predicate "
                    "matched this task."
                ),
            )

        completed_steps: list[WorkflowStepResult] = []
        previous_result: AgentLoopResult | None = None

        def _workflow_token_usage() -> TokenUsage | None:
            # Recomputed fresh at every one of this method's return
            # points below from whatever `completed_steps` holds so
            # far -- including the halting step itself, since it was
            # genuinely billed even though the workflow didn't
            # complete. Build Phase 19.
            return combine_token_usage(
                *(step.token_usage for step in completed_steps)
            )

        for step in workflow.steps:

            registration = self._find_registration(step.subject)

            try:
                step_task_text = step.build_task(
                    normalized.text, previous_result
                )
            except Exception as exc:  # noqa: BLE001 -- a step's own
                # build_task is caller-supplied and may fail for any
                # reason (most often: no usable artifact in the
                # previous step's result); this is reported as a
                # normal, inspectable WorkflowRunResult, not an
                # uncaught exception out of an otherwise-real Kernel
                # call.
                return WorkflowRunResult(
                    status="STEP_TASK_BUILD_ERROR",
                    workflow_name=workflow.name,
                    completed_steps=tuple(completed_steps),
                    reason=(
                        f"Step {step.subject!r}'s build_task raised "
                        f"while building its task text: {exc}"
                    ),
                    token_usage=_workflow_token_usage(),
                )

            step_normalized = self._normalize(step_task_text)

            recovery_attempts = 0
            loop_result = self._execute_once(
                registration=registration,
                normalized=step_normalized,
                max_steps=max_steps,
            )

            # Same reasoning as Kernel.run()'s own accumulation above:
            # each retry is a full, fresh, separately-billed attempt.
            step_token_usage = loop_result.token_usage

            while (
                recovery_attempts < self.max_recovery_attempts
                and self._should_recover(loop_result)
            ):
                recovery_attempts += 1

                loop_result = self._execute_once(
                    registration=registration,
                    normalized=step_normalized,
                    max_steps=max_steps,
                )

                step_token_usage = combine_token_usage(
                    step_token_usage,
                    loop_result.token_usage,
                )

            verification = self._verify(loop_result)

            completed_steps.append(
                WorkflowStepResult(
                    subject=step.subject,
                    loop_result=loop_result,
                    verification=verification,
                    token_usage=step_token_usage,
                )
            )

            if loop_result.status == "APPROVAL_REQUIRED":
                return WorkflowRunResult(
                    status="AWAITING_APPROVAL",
                    workflow_name=workflow.name,
                    completed_steps=tuple(completed_steps),
                    reason=(
                        f"Step {step.subject!r} is awaiting human "
                        "approval; the workflow has stopped here."
                    ),
                    token_usage=_workflow_token_usage(),
                )

            if loop_result.status != "COMPLETED":
                return WorkflowRunResult(
                    status=loop_result.status,
                    workflow_name=workflow.name,
                    completed_steps=tuple(completed_steps),
                    reason=(
                        f"Step {step.subject!r} did not complete "
                        f"(status={loop_result.status!r}); the "
                        "workflow has stopped here."
                    ),
                    token_usage=_workflow_token_usage(),
                )

            if not verification.passed:
                return WorkflowRunResult(
                    status="VERIFICATION_FAILED",
                    workflow_name=workflow.name,
                    completed_steps=tuple(completed_steps),
                    reason=(
                        f"Step {step.subject!r} completed but did not "
                        f"pass verification: {verification.reason}"
                    ),
                    token_usage=_workflow_token_usage(),
                )

            previous_result = loop_result

        return WorkflowRunResult(
            status="COMPLETED",
            workflow_name=workflow.name,
            completed_steps=tuple(completed_steps),
            reason=None,
            token_usage=_workflow_token_usage(),
        )

    def run_multi_agent_workflow(
        self,
        *,
        subjects: Sequence[str],
        task: str,
        thread_id: str,
        approval_gates: Mapping[str, bool] | None = None,
        max_steps: int = 10,
    ) -> MultiAgentWorkflowResult:
        """
        Build Phase 25: chain already-registered agents (by `subjects`,
        in the given order) into a real MultiAgentWorkflowEngine (Build
        Phase 24) and run it -- e.g.
        subjects=("research_agent", "writer_agent", "reviewer_agent").

        Entirely separate from both Kernel.run() (single agent) and
        Kernel.run_workflow() (Build Phase 15's own declarative,
        plain-Python-loop multi-step mechanism, which cannot pause and
        later continue past an approval gate -- an "AWAITING_APPROVAL"
        WorkflowRunResult just stops, with nothing to resume it). This
        method's real advantage over run_workflow() is exactly that
        gap: a stage named in `approval_gates` pauses the whole
        workflow via a native LangGraph `interrupt()` (Build Phase 24),
        and a LATER, SEPARATE call to `resume_multi_agent_workflow()`
        with the SAME `thread_id` continues it from exactly where it
        paused -- real cross-call pause/resume, not "stop and report."

        `approval_gates` maps a subject to `True` to mark that stage as
        requiring human approval before the workflow advances past it
        (see WorkflowStage.requires_human_approval); a subject absent
        from this mapping (or the mapping itself omitted) never pauses.

        Every stage built here is threaded with this Kernel's own
        `self.guardrail_engine` (Build Phase 23) and `self.token_budget`
        (Build Phase 26), exactly mirroring how `_execute_once` threads
        both into a single agent's own AgentExecutionLoop -- `None`
        unless this Kernel was itself constructed with one.

        Deliberately narrower than Kernel.run(): no NORMALIZE/CLASSIFY/
        context-retrieval/policy-evaluation/independent-verification
        lifecycle around this call, and no `checkpoint_store` support
        for a stage's own in-flight progress (a stage that crashes
        mid-task restarts that stage from scratch on any future retry,
        unlike Kernel.resume()'s own crash recovery) -- both are real,
        honest, documented scope boundaries for this first Kernel-wired
        version, not silently-assumed coverage.

        Raises TypeError/ValueError for a bad `subjects`/`approval_gates`,
        or if `subjects` names a subject with no matching
        AgentRegistration -- the same "fail loud on misconfiguration"
        convention this class already uses elsewhere.
        """

        if not isinstance(subjects, Sequence) or isinstance(
            subjects, (str, bytes)
        ):
            raise TypeError(
                "subjects must be a sequence of subject strings."
            )

        if not subjects:
            raise ValueError("subjects must not be empty.")

        if approval_gates is not None and not isinstance(
            approval_gates, Mapping
        ):
            raise TypeError("approval_gates must be a Mapping or None.")

        stages = self._build_multi_agent_stages(
            subjects=subjects,
            approval_gates=approval_gates or {},
            max_steps=max_steps,
        )

        engine = MultiAgentWorkflowEngine(stages)

        result = engine.run(task=task, thread_id=thread_id)

        if result.status == "AWAITING_APPROVAL":
            self._multi_agent_engines[thread_id] = engine
        else:
            self._multi_agent_engines.pop(thread_id, None)

        return result

    def resume_multi_agent_workflow(
        self,
        *,
        thread_id: str,
        approval: object,
    ) -> MultiAgentWorkflowResult:
        """
        Continue a multi-agent workflow `run_multi_agent_workflow()`
        left AWAITING_APPROVAL for `thread_id`. See that method's own
        docstring for the full pause/resume design.

        Raises ValueError if `thread_id` names no currently-paused
        workflow -- either it was never started, already ran to a
        terminal status, or this Kernel instance is not the one that
        started it (the paused engine lives only in this Kernel's own
        in-memory `_multi_agent_engines`, not on disk -- see
        MultiAgentWorkflowEngine's own module docstring on
        MemorySaver's in-memory-only scope).
        """

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string.")

        engine = self._multi_agent_engines.get(thread_id)

        if engine is None:
            raise ValueError(
                f"No paused multi-agent workflow found for "
                f"thread_id={thread_id!r}. It may never have been "
                "started, may have already reached a terminal status, "
                "or may have been started on a different Kernel "
                "instance."
            )

        result = engine.resume(thread_id=thread_id, approval=approval)

        if result.status != "AWAITING_APPROVAL":
            self._multi_agent_engines.pop(thread_id, None)

        return result

    def _build_multi_agent_stages(
        self,
        *,
        subjects: Sequence[str],
        approval_gates: Mapping[str, bool],
        max_steps: int,
    ) -> tuple[WorkflowStage, ...]:
        """
        Deliberately does NOT reuse `_find_registration` -- that
        helper's own docstring documents it as only ever being called
        with a subject `register_workflow()` has already validated,
        raising an "internal error" RuntimeError otherwise. Here,
        `subjects` is ordinary caller input to a public method, never
        pre-validated by anything -- an unknown subject is a normal,
        expected, user-facing ValueError, not an internal-error
        RuntimeError.
        """

        stages: list[WorkflowStage] = []

        for subject in subjects:

            registration = None

            for candidate in self._registrations:
                if candidate.subject == subject:
                    registration = candidate
                    break

            if registration is None:
                raise ValueError(
                    f"No AgentRegistration found for subject "
                    f"{subject!r}. Register it with "
                    "Kernel.register_agent() before including it in a "
                    "multi-agent workflow."
                )

            stages.append(
                WorkflowStage(
                    name=registration.subject,
                    build_agent=registration.build_agent,
                    build_decision_engine=registration.build_decision_engine,
                    max_steps=max_steps,
                    requires_human_approval=bool(
                        approval_gates.get(subject, False)
                    ),
                    guardrail_engine=self.guardrail_engine,
                    token_budget=self.token_budget,
                )
            )

        return tuple(stages)

    def _select_workflow(
        self,
        normalized: NormalizedTask,
    ) -> WorkflowDefinition | None:
        """
        First registered WorkflowDefinition (in registration order)
        whose `can_handle` matches `normalized`, or `None` if none do
        -- mirrors _select_agent's own first-match-wins rule, applied
        to workflows instead of agents.
        """

        for workflow in self._workflows:
            if workflow.can_handle(normalized):
                return workflow

        return None

    def _find_registration(self, subject: str) -> AgentRegistration:
        """
        The AgentRegistration for `subject`. Only ever called with a
        subject a WorkflowStep names, which register_workflow() has
        already verified matches a real registration -- so reaching
        the RuntimeError below would mean this Kernel's own
        `_registrations` changed shape after registration (not
        possible through this class's public API today), not a
        reachable user-facing error.
        """

        for registration in self._registrations:
            if registration.subject == subject:
                return registration

        raise RuntimeError(
            f"Internal error: workflow step names subject {subject!r}, "
            "which has no matching AgentRegistration."
        )

    def _execute_once(
        self,
        *,
        registration: AgentRegistration,
        normalized: NormalizedTask,
        max_steps: int,
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_id: str | None = None,
    ) -> AgentLoopResult:
        """
        Build a fresh PLAN from `registration` and run it once through
        the OrchestrationEngine. Used both for the initial attempt and
        for every RECOVER IF NEEDED retry -- always a full, fresh
        attempt, never a resume (that per-process retry policy is
        untouched by, and unrelated to, Build Phase 22's checkpoint/
        resume -- see core/agents/checkpoint.py's own module docstring
        for the distinction between the two).

        When `checkpoint_store` is given, OR when this Kernel was
        constructed with a `guardrail_engine` (Build Phase 23), OR when
        this Kernel was constructed with a `token_budget` (Build Phase
        26), this bypasses the pluggable OrchestrationEngine seam and
        drives AgentExecutionLoop directly instead -- the exact same
        call SequentialOrchestrationEngine.run() itself makes
        internally -- so this attempt's progress can be checkpointed
        and/or its decisions guardrail-checked and/or its spend
        budget-capped. See checkpoint.py's, guardrails.py's, and
        budget.py's own module docstrings for why this deliberately
        does not (yet) flow through LangGraphOrchestrationEngine:
        threading any of these concerns through that engine too would
        require changing code this sandbox cannot install or execute
        even once (no PyPI access), and shipping an unverified change
        to it would break this project's own "nothing is done until a
        real pytest run confirms it" rule -- the same reasoning
        already applied at Build Phases 4, 18, 21, 22, and 23 for this
        exact untestable file.
        """

        plan = self._plan(
            registration=registration,
            normalized=normalized,
            max_steps=max_steps,
        )

        plan.agent.start_task(normalized.text)

        if (
            checkpoint_store is not None
            or self.guardrail_engine is not None
            or self.token_budget is not None
        ):

            loop = AgentExecutionLoop(
                agent=plan.agent,
                decision_engine=plan.decision_engine,
                max_steps=plan.max_steps,
                checkpoint_store=checkpoint_store,
                checkpoint_id=checkpoint_id,
                guardrail_engine=self.guardrail_engine,
                token_budget=self.token_budget,
            )

            return loop.run()

        return self.orchestration_engine.run(
            agent=plan.agent,
            decision_engine=plan.decision_engine,
            max_steps=plan.max_steps,
        )

    # -- Lifecycle steps ------------------------------------------------

    def _normalize(
        self,
        task: object,
    ) -> NormalizedTask:
        """
        NORMALIZE. Real: mirrors the same validation
        AgentCore.start_task() applies (non-empty string), applied
        once at the Kernel boundary before any agent is selected, and
        trims surrounding whitespace so an agent never sees a task
        that differs only by incidental formatting.
        """

        if not isinstance(task, str):
            raise TypeError("task must be a string.")

        stripped = task.strip()

        if not stripped:
            raise ValueError("task must not be empty.")

        return NormalizedTask(text=stripped)

    def _retrieve_context(
        self,
        normalized: NormalizedTask,
    ) -> RetrievedContext | None:
        """
        CONTEXT RETRIEVAL. Real when `self.memory_store` is configured
        (Build Phase 14): runs a real MemoryStore.search() for the
        normalized task text and returns the result as a
        RetrievedContext -- see that dataclass's own docstring for why
        this is inspectable-only and never fed back into `normalized`
        or any later step of this same `run()` call.

        Returns `None` in two cases, never raises for either (the same
        "degrade rather than fail an otherwise-real Kernel result over
        an optional, additive step" tolerance _evaluate_policy and
        _trigger_independent_verification already established -- see
        their own docstrings):

          - no `memory_store` is configured on this Kernel (the
            default -- see Kernel.__init__'s own docstring for why an
            unconfigured Kernel must behave exactly as it did before
            Build Phase 14)
          - MemoryStore.search() itself raises (e.g. ValueError for an
            empty query -- not reachable through _normalize's own
            non-empty guarantee today, but this method does not rely
            on that guarantee holding forever)
        """

        if self.memory_store is None:
            return None

        try:
            records = self.memory_store.search(
                normalized.text,
                limit=self.context_retrieval_limit,
            )
        except ValueError:
            return None

        return RetrievedContext(query=normalized.text, records=records)

    def _classify(
        self,
        normalized: NormalizedTask,
    ) -> TaskClassification:
        """
        CLASSIFY. Real: evaluates every registered agent's
        `can_handle` predicate against the normalized task and
        collects every match, in registration order.
        """

        candidates = tuple(
            registration.subject
            for registration in self._registrations
            if registration.can_handle(normalized)
        )

        return TaskClassification(candidate_subjects=candidates)

    def _select_strategy(
        self,
        classification: TaskClassification,
    ) -> TaskClassification:
        """
        STRATEGY SELECTION. Real but intentionally simple: today
        exactly one execution strategy exists (run the first matching
        agent to completion through an OrchestrationEngine), so this
        step is currently an identity pass-through of the
        classification. It is kept as its own step -- rather than
        folded into `_select_agent` -- so a future phase can add real
        strategy choices (e.g. ranking multiple candidate agents, or
        choosing a multi-agent plan) without changing this method's
        signature or where it's called from.
        """

        return classification

    def _select_agent(
        self,
        classification: TaskClassification,
    ) -> AgentRegistration:
        """
        AGENT SELECTION. Real: picks the first candidate subject (in
        registration order) and returns its full AgentRegistration.
        Only called when `classification.candidate_subjects` is
        non-empty -- `run()` already handles the empty case.
        """

        selected_subject = classification.candidate_subjects[0]

        for registration in self._registrations:
            if registration.subject == selected_subject:
                return registration

        raise RuntimeError(
            "Internal error: classification produced a subject with "
            "no matching registration."
        )

    def _plan(
        self,
        *,
        registration: AgentRegistration,
        normalized: NormalizedTask,
        max_steps: int,
    ) -> KernelPlan:
        """
        PLAN. Real: builds a fresh AgentCore and AgentDecisionEngine
        from the selected registration's factories (MODEL SELECTION
        and TOOL SELECTION are both already resolved by this point --
        see this module's own docstring) and packages them with the
        normalized task's execution budget into one KernelPlan.
        """

        agent = registration.build_agent()

        if not isinstance(agent, AgentCore):
            raise TypeError(
                "AgentRegistration.build_agent() must return an "
                "AgentCore."
            )

        decision_engine = registration.build_decision_engine()

        return KernelPlan(
            subject=registration.subject,
            agent=agent,
            decision_engine=decision_engine,
            max_steps=max_steps,
        )

    def _should_recover(
        self,
        loop_result: AgentLoopResult,
    ) -> bool:
        """
        RECOVER IF NEEDED. Real: delegates to
        self.policy_engine.is_recovery_authorized(), which returns
        True only for a status that indicates something crashed
        unexpectedly (DECISION_ERROR, EXECUTION_ERROR), never for a
        considered outcome the agent or loop reported deliberately
        (FAILED, TOOL_ERROR, APPROVAL_REQUIRED, MAX_STEPS_EXCEEDED,
        INVALID_ACTION). The Kernel itself no longer decides which
        statuses are recoverable -- that is now a Policy Layer
        decision (core/policies/policy_engine.py), per POLICY_SPEC.md's
        Failure Policy step 4 and its own "Policy Enforcement" section.
        See both modules' docstrings for the full reasoning.
        """

        return self.policy_engine.is_recovery_authorized(
            loop_result.status
        )

    def _verify(
        self,
        loop_result: AgentLoopResult,
    ) -> KernelVerification:
        """
        VERIFY. Real: see KernelVerification's own docstring for the
        exact rule. Only meaningful for a COMPLETED result; every
        other terminal status is reported as not-passed with an
        explanatory reason, since verification only judges a claimed
        success.
        """

        if loop_result.status != "COMPLETED":
            return KernelVerification(
                passed=False,
                reason=(
                    "Verification does not apply: the agent did not "
                    f"report COMPLETED (status={loop_result.status!r})."
                ),
            )

        last_result = loop_result.last_result

        if last_result is None:
            return KernelVerification(
                passed=True,
                reason=(
                    "Agent completed without executing any tool; "
                    "nothing to verify."
                ),
            )

        if getattr(last_result, "status", None) == "SUCCESS":
            return KernelVerification(
                passed=True,
                reason="Agent completed and its last tool call succeeded.",
            )

        return KernelVerification(
            passed=False,
            reason=(
                "Agent reported COMPLETED but its last tool call did "
                f"not succeed (status={getattr(last_result, 'status', None)!r})."
            ),
        )

    def _evaluate_policy(
        self,
        loop_result: AgentLoopResult,
    ) -> ExternalActionEvaluation | None:
        """
        Answers POLICY_SPEC.md's "External Actions" six questions for
        the external action actually performed during this run, when
        one was. Real: uses the identifying data ToolExecutionResult
        now carries (`.subject`, `.tool_id`, `.action`, added in Build
        Phase 7 -- see its own docstring in
        core/tools/engine/tool_gateway.py) together with the real
        SecurityDecision the Tool Gateway already computed
        (`.security_decision`). This never re-derives risk or
        approval itself -- see PolicyEngine.evaluate_external_action()'s
        own docstring -- it only packages an already-final answer into
        one inspectable record.

        This closes the gap Build Phase 6 named and deliberately left
        open (see core/policies/policy_engine.py's module docstring):
        before Build Phase 7, neither SecurityDecision nor
        ToolExecutionResult preserved which tool_id/action produced
        them, so there was nothing here for the Kernel to look up.

        This is deliberately a post-hoc, inspectable record, not a
        pre-execution gate: by the time the Kernel ever sees
        `loop_result`, the Tool Gateway has already synchronously
        authorized (or denied) every tool call the agent made during
        the loop -- POLICY_SPEC.md's own text ("Policies must remain
        separate from agents, tools, memory, and orchestration") rules
        out the Kernel/Policy Layer reaching into the loop to gate
        calls one at a time, and Kernel v1's architecture has no
        per-step hook of its own to do that from outside the loop
        anyway (see this module's own docstring, EXECUTE).

        Returns None when no tool was ever invoked (`last_result` is
        None -- e.g. the agent completed or failed without calling a
        tool): there is no external action here to answer these
        questions about.

        Also returns None -- rather than letting this propagate out of
        Kernel.run() -- when the identifying/security data itself is
        incomplete (PolicyEngine.evaluate_external_action() raises
        ValueError for that). This is a real, not just theoretical,
        possibility: `last_result` is only required to be duck-typed
        (see _verify's own precedent), so a caller-supplied
        AgentCore/ToolRuntime substitute outside this project's own
        ToolGateway may not carry a complete SecurityDecision. Per
        this project's standing constraint that the system must never
        become so strict it refuses to execute anything, an incomplete
        policy answer degrades this one inspectable field to None
        rather than failing an otherwise-real Kernel result over data
        that was never required to produce it.
        """

        last_result = loop_result.last_result

        if last_result is None:
            return None

        try:
            return self.policy_engine.evaluate_external_action(
                action=getattr(last_result, "action", None),
                subject=getattr(last_result, "subject", None),
                tool_id=getattr(last_result, "tool_id", None),
                security_decision=getattr(
                    last_result, "security_decision", None
                ),
            )
        except ValueError:
            return None

    def _trigger_independent_verification(
        self,
        *,
        completed_subject: str,
        loop_result: AgentLoopResult,
        max_steps: int,
    ) -> AgentLoopResult | None:
        """
        POLICY_SPEC.md's Workflow Constraints (Build Phase 12): after
        `completed_subject`'s run just completed, should a second,
        specific agent be triggered now for real, independent,
        content-level verification? Real, but deliberately narrow --
        see core/policies/policy_engine.py's own docstring (WORKFLOW
        CONSTRAINTS) for the one declared transition this currently
        recognizes.

        Returns `None` in every one of these cases, never raises for
        any of them (the same "degrade rather than fail an otherwise-
        real Kernel result over an optional, additive step" tolerance
        _evaluate_policy already established -- see its own docstring):

          - no tool was ever invoked during `completed_subject`'s run
            (nothing for PolicyEngine.evaluate_workflow_trigger() to
            answer about)
          - PolicyEngine.evaluate_workflow_trigger() names no
            transition for this exact (subject, tool_id, tool_status)
          - this Kernel has no `independent_verifier` configured at all
            (the default -- see Kernel.__init__'s own docstring for why
            an unconfigured Kernel must never fail or behave
            differently here)
          - `independent_verifier` is configured for a *different*
            subject than the one the policy actually named (defensive:
            never trigger an unrelated agent just because some
            transition fired)
          - the completed action's own result carries no usable
            artifact path to review (e.g. a caller-supplied
            ToolRuntime/AgentCore substitute outside this project's own
            ToolGateway, whose result shape does not match
            write_report's real `{"path": ..., "size_bytes": ...}`
            artifact -- the same duck-typed tolerance _verify/
            _evaluate_policy already apply to `last_result`)

        When a transition IS triggered, this builds a fresh verifier
        agent + decision engine from `self.independent_verifier`'s own
        factories (never reused across runs, same reasoning as
        AgentRegistration's own factories), starts it on a real task
        naming the exact artifact path the primary agent just
        published (`f"Review {path}."`), and runs it to a terminal
        result through the SAME OrchestrationEngine this Kernel already
        uses for its primary task -- reviewer_agent's own tools are all
        LOW-risk, read-only, and never require approval (see
        core/agents/REVIEWER_AGENT.md), so this can never itself pause
        on a human-approval gate the caller wasn't already expecting.
        `max_steps` reuses the same budget the primary task was given,
        for the same reason PLAN reuses it for RECOVER IF NEEDED
        retries: one consistent execution budget per Kernel.run() call,
        not a second free parameter to reason about.

        A caller-supplied `build_agent()` that does not return a real
        AgentCore raises TypeError, exactly like Kernel._plan() already
        does for the primary agent -- that is a genuine build-time
        misconfiguration bug in the registration itself, not incomplete
        runtime data, so it is deliberately NOT caught and degraded to
        None the way the cases above are.
        """

        last_result = loop_result.last_result

        if last_result is None:
            return None

        evaluation = self.policy_engine.evaluate_workflow_trigger(
            completed_subject=completed_subject,
            tool_id=getattr(last_result, "tool_id", None),
            tool_status=getattr(last_result, "status", None),
        )

        if not evaluation.should_trigger:
            return None

        if self.independent_verifier is None:
            return None

        if evaluation.next_subject != self.independent_verifier.subject:
            return None

        path = extract_first_artifact_path(loop_result)

        if path is None:
            return None

        verifier_agent = self.independent_verifier.build_agent()

        if not isinstance(verifier_agent, AgentCore):
            raise TypeError(
                "WorkflowVerifierRegistration.build_agent() must "
                "return an AgentCore."
            )

        verifier_decision_engine = (
            self.independent_verifier.build_decision_engine()
        )

        verifier_agent.start_task(f"Review {path}.")

        return self.orchestration_engine.run(
            agent=verifier_agent,
            decision_engine=verifier_decision_engine,
            max_steps=max_steps,
        )

    def _learn(
        self,
        loop_result: AgentLoopResult,
        verification: KernelVerification,
    ) -> None:
        """
        LEARN. Explicit no-op: no lesson-recording subsystem exists
        in this project yet. This method exists so a real
        implementation has an obvious, single place to be added
        later.
        """

        return None

    def _final_status(
        self,
        loop_result: AgentLoopResult,
        verification: KernelVerification,
    ) -> str:
        """
        FINAL RESULT status mapping. See KernelResult's own docstring
        for the full list of possible values.
        """

        if loop_result.status == "APPROVAL_REQUIRED":
            return "AWAITING_APPROVAL"

        if loop_result.status == "COMPLETED":
            return "COMPLETED" if verification.passed else "VERIFICATION_FAILED"

        return loop_result.status
