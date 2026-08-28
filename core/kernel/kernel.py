from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.agents.agent_core import (
    AgentCore,
)

from core.agents.agent_loop import (
    AgentLoopResult,
)

from core.agents.decision_engine import (
    AgentDecisionEngine,
)

from core.memory.memory_store import (
    MemoryRecord,
    MemoryStore,
)

from core.orchestration.engine_factory import (
    create_default_orchestration_engine,
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
#                           only "run one matching agent" exists
#                           today; ranking multiple candidates is
#                           future work once more than one agent is
#                           registered
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
    ) -> None:

        if not isinstance(max_recovery_attempts, int):
            raise TypeError("max_recovery_attempts must be an integer.")

        if max_recovery_attempts < 0:
            raise ValueError(
                "max_recovery_attempts must be zero or greater."
            )

        self._registrations: list[AgentRegistration] = []

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

    def run(
        self,
        task: str,
        *,
        max_steps: int = 10,
    ) -> KernelResult:
        """
        Run the full Kernel lifecycle for one objective, from
        NORMALIZE through FINAL RESULT.
        """

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
        )

        while (
            recovery_attempts < self.max_recovery_attempts
            and self._should_recover(loop_result)
        ):
            recovery_attempts += 1

            loop_result = self._execute_once(
                registration=registration,
                normalized=normalized,
                max_steps=max_steps,
            )

        verification = self._verify(loop_result)

        policy_evaluation = self._evaluate_policy(loop_result)

        independent_verification = self._trigger_independent_verification(
            completed_subject=registration.subject,
            loop_result=loop_result,
            max_steps=max_steps,
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
        )

    def _execute_once(
        self,
        *,
        registration: AgentRegistration,
        normalized: NormalizedTask,
        max_steps: int,
    ) -> AgentLoopResult:
        """
        Build a fresh PLAN from `registration` and run it once through
        the OrchestrationEngine. Used both for the initial attempt and
        for every RECOVER IF NEEDED retry -- always a full, fresh
        attempt, never a resume.
        """

        plan = self._plan(
            registration=registration,
            normalized=normalized,
            max_steps=max_steps,
        )

        plan.agent.start_task(normalized.text)

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
