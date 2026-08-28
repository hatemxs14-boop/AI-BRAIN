from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

from core.security.engine.security_decision import (
    SecurityDecision,
)


# ---------------------------------------------------------------------
# Real (v1) implementation of core/policies/POLICY_SPEC.md.
#
# Honestly scoped exactly like Kernel v1 (see core/kernel/kernel.py's
# own module docstring): implements every part of POLICY_SPEC.md for
# which a real foundation and real data already exist in this project,
# and represents every remaining part as an explicit, documented gap --
# never a fabricated implementation that would hide the gap instead of
# naming it. Concretely:
#
#   POLICY HIERARCHY        real, inspectable data (PolicyLevel), in
#                           the spec's own declared order. Nothing in
#                           this project yet needs to arbitrate a
#                           conflict between hierarchy levels -- there
#                           is exactly one enforcement path today (the
#                           Security Layer, hardened across Passes
#                           1-5) -- so this exists as an authoritative
#                           ordering ready for a future conflict to be
#                           checked against, not as active
#                           conflict-resolution logic.
#
#   EXTERNAL ACTIONS         real (evaluate_external_action): packages
#                           POLICY_SPEC.md's exact six-question
#                           checklist into one explicit, inspectable
#                           ExternalActionEvaluation, sourced from an
#                           already-computed SecurityDecision (the
#                           hardened Security Layer already answers the
#                           risk/approval questions correctly -- this
#                           method's job is to assert a COMPLETE answer
#                           to all six questions exists, never to
#                           re-derive any of them). Deliberately a pure
#                           function of caller-supplied inputs rather
#                           than something that reaches into a
#                           ToolExecutionResult/AgentLoopResult itself
#                           -- as of Build Phase 6, neither
#                           SecurityDecision nor ToolExecutionResult
#                           preserved which tool_id/action produced
#                           them (confirmed by reading both dataclasses
#                           directly, not assumed), so there was
#                           nothing for a caller to look up on its own.
#                           Build Phase 7 closed that specific,
#                           named gap at its source -- ToolExecutionResult
#                           now carries `.subject`/`.tool_id`/`.action`
#                           (see core/tools/engine/tool_gateway.py) --
#                           and wired this method into a real Kernel
#                           call site: Kernel._evaluate_policy calls it
#                           for the last tool actually invoked during a
#                           run, surfaced as
#                           KernelResult.policy_evaluation (see
#                           core/kernel/kernel.py). This method itself
#                           did not need to change for that -- it was
#                           real and fully tested before Build Phase 7;
#                           only the data available to callers changed.
#                           (Separately: POLICY_SPEC.md itself says
#                           "Policies must remain separate from agents,
#                           tools, memory, and orchestration" and that
#                           enforcement is the orchestration layer's
#                           job -- so this was never a candidate to
#                           wire into core/tools/engine/tool_gateway.py
#                           directly, only into Kernel/orchestration-
#                           level code, which is exactly where Build
#                           Phase 7 wired it.)
#
#   AGENT CONSTRAINTS         real for one concrete, checkable slice of
#                           this section as of Build Phase 9:
#                           evaluate_agent_scope() answers "operate only
#                           within declared responsibilities" / "never
#                           silently expand their scope" for the one
#                           artifact this project actually has that
#                           states an agent's declared scope in a
#                           checkable form -- each agent's own spec-
#                           declared "Tools > Allowed" list (e.g.
#                           RESEARCH_AGENT_DECLARED_TOOL_IDS in
#                           core/agents/research_agent.py,
#                           WRITER_AGENT_DECLARED_TOOL_IDS in
#                           core/agents/writer_agent.py). Both
#                           build_research_agent() and
#                           build_writer_agent() call it immediately
#                           after registering their tools, and raise a
#                           clear ValueError if the tool ids actually
#                           registered ever include one outside that
#                           agent's own declared set -- the same
#                           "silent drift becomes a loud, immediate
#                           error" pattern Pass 4 finding L already
#                           established for permissions.json's
#                           defaults/risk_levels sections, applied here
#                           one layer up (agent scope, not policy
#                           config). This never affects any individual
#                           request or tool call -- it is a build-time
#                           self-consistency guard, not a runtime gate,
#                           so it can never make a legitimate task
#                           refuse to execute; it can only catch a
#                           future code change that silently registers
#                           a tool an agent's own spec doesn't declare.
#                           This specific check was not buildable before
#                           Build Phase 8: with only research_agent ever
#                           registered, there was no second agent's
#                           scope to compare against, and "declared
#                           responsibilities" had no artifact more
#                           concrete than AgentIdentity.purpose (free
#                           text, nothing enforced against then or now).
#                           The REST of this section's bullets --
#                           "use only authorized tools", "respect
#                           memory access boundaries", "never bypass
#                           approval gates" -- are already enforced by
#                           the hardened Security Layer (Passes 1-5) at
#                           every actual tool call, independent of this
#                           method; "follow verification requirements"
#                           is already enforced by Kernel._verify()
#                           (real since Build Phase 4). This method adds
#                           a real check for the one specific gap those
#                           mechanisms don't already cover, not a
#                           restatement of what they already do.
#
#   FAILURE POLICY           real for the one step that was a bare,
#                           unlabeled implementation detail before this
#                           phase: is_recovery_authorized() is what
#                           Kernel._should_recover() (core/kernel/
#                           kernel.py) now actually calls, making
#                           "attempt recovery only when the recovery is
#                           itself authorized" a real Policy Layer
#                           decision the Kernel asks for, rather than a
#                           private Kernel heuristic -- exactly
#                           matching this spec's own "Policy
#                           Enforcement" section ("The orchestration
#                           layer is responsible for enforcing policies
#                           during execution ... The Policy Layer
#                           determines whether the action is
#                           permitted"). The other four Failure Policy
#                           steps are already real Kernel behavior, not
#                           new code this phase adds: stop (Kernel.run()
#                           cannot return without the operation having
#                           fully terminated), preserve diagnostic
#                           information (KernelResult.reason, real
#                           since Build Phase 4), never silently bypass
#                           (KernelResult.status is always the loop's
#                           real terminal status or an honestly-derived
#                           one -- see Kernel._final_status), escalate
#                           to the human when required
#                           (APPROVAL_REQUIRED -> AWAITING_APPROVAL,
#                           real since Build Phase 4). Restating those
#                           four as new code here would be exactly the
#                           "fabricate an implementation to look
#                           complete" anti-pattern this project has
#                           consistently avoided (see Kernel v1's own
#                           docstring).
#
# NOT IMPLEMENTED (documented, not fabricated):
#
#   WORKFLOW CONSTRAINTS     No workflow concept exists in this
#                           project. Kernel v1's STRATEGY SELECTION
#                           step is currently a single-agent
#                           passthrough (see core/kernel/kernel.py) --
#                           there is no multi-step, multi-agent plan
#                           yet for a workflow-level constraint to
#                           apply to.
#   MEMORY CONSTRAINTS       No memory layer exists in this project
#                           (see Kernel v1's CONTEXT RETRIEVAL/PERSIST/
#                           LEARN no-ops) -- there is nothing yet for
#                           "recalled memory is untrusted context" or
#                           "secrets must never be stored" to apply to.
#   CORE RULES               This section's eight principles are
#                           already satisfied as observable behavior by
#                           the existing, hardened stack (fail-closed
#                           defaults throughout the Security Layer,
#                           Kernel's VERIFY/HUMAN APPROVAL steps, tool
#                           permission enforcement in ToolGateway)
#                           rather than needing new code to restate
#                           them as a checklist -- doing so would test
#                           the restatement, not the actual system.
# ---------------------------------------------------------------------


class PolicyLevel(IntEnum):
    """
    POLICY_SPEC.md's Policy Hierarchy, in the spec's own declared
    order. Lower values are higher-priority: "A lower-level policy
    must never override a higher-level policy" (POLICY_SPEC.md).
    """

    SYSTEM_CONSTITUTION = 1
    SECURITY_AND_SAFETY = 2
    HUMAN_APPROVAL = 3
    TOOL_RISK = 4
    AGENT_CONSTRAINTS = 5
    WORKFLOW_CONSTRAINTS = 6


@dataclass(frozen=True)
class ExternalActionEvaluation:
    """
    Answers POLICY_SPEC.md's "External Actions" six questions for one
    action, in the spec's own order:

        1. What action will occur?              -> action
        2. Which agent requested it?             -> subject
        3. Which tool will perform it?           -> tool_id
        4. What is the risk level?               -> risk_level
        5. Is human approval required?           -> approval_required
        6. What verification is required
           afterward?                            -> verification_required
    """

    action: str
    subject: str
    tool_id: str
    risk_level: str
    approval_required: bool
    verification_required: bool


@dataclass(frozen=True)
class AgentScopeEvaluation:
    """
    Answers POLICY_SPEC.md's Agent Constraints "operate only within
    declared responsibilities" / "never silently expand their scope"
    for one agent's tool registrations, checked against that same
    agent's own spec-declared tool-id set.

    `unauthorized_tool_ids` is `actual_tool_ids - declared_tool_ids` --
    empty when the agent registered nothing outside its own declared
    scope. `within_scope` is `not unauthorized_tool_ids`, kept as its
    own field (rather than requiring every caller to re-derive it) so
    a caller only needs the one boolean it actually wants: a caller
    building the agent asks "is this build allowed to succeed?"
    (`within_scope`), while a caller diagnosing a violation asks
    "which tool ids exactly?" (`unauthorized_tool_ids`).
    """

    subject: str
    declared_tool_ids: frozenset[str]
    actual_tool_ids: frozenset[str]
    unauthorized_tool_ids: frozenset[str]
    within_scope: bool


class PolicyEngine:
    """
    Real (v1) implementation of core/policies/POLICY_SPEC.md. See this
    module's own docstring above for exactly what is real vs.
    deliberately not implemented yet, and why.
    """

    # The only two AgentLoopResult statuses POLICY_SPEC.md's Failure
    # Policy step 4 ("attempt recovery only when the recovery is
    # itself authorized") authorizes a retry for: something crashed
    # unexpectedly (a decision engine raising, e.g. a transient LLM/
    # network hiccup; a tool executor raising unexpectedly, e.g. a
    # network blip) -- never a considered outcome the agent or loop
    # already reported on purpose. See core/kernel/kernel.py's own
    # module docstring for the full per-status reasoning; this is the
    # authoritative policy decision Kernel._should_recover() consults.
    RECOVERY_AUTHORIZED_STATUSES = frozenset(
        {"DECISION_ERROR", "EXECUTION_ERROR"}
    )

    def is_recovery_authorized(
        self,
        status: str,
    ) -> bool:
        """
        POLICY_SPEC.md Failure Policy, step 4: "Attempt recovery only
        when the recovery is itself authorized." Real: returns True
        only for a status that indicates something crashed
        unexpectedly, never for a status the agent/loop reported as a
        deliberate, considered outcome.
        """

        return status in self.RECOVERY_AUTHORIZED_STATUSES

    def evaluate_external_action(
        self,
        *,
        action: str,
        subject: str,
        tool_id: str,
        security_decision: SecurityDecision,
    ) -> ExternalActionEvaluation:
        """
        POLICY_SPEC.md's "External Actions" section: "Before performing
        an external action, the system must determine" the six
        questions this method answers -- see ExternalActionEvaluation's
        own docstring for the exact mapping.

        `security_decision` is expected to be a real, computed
        SecurityDecision (from SecurityDecisionPoint.evaluate() /
        evaluate_with_approval()) in production -- this method never
        re-derives risk or approval itself; it only asserts a complete
        answer exists for every question POLICY_SPEC.md requires, per
        that same section's own rule: "If any required authorization
        is missing, execution must stop." It is accessed by attribute
        only (`.authorization.effective_risk`, `.approval.required`),
        the same duck-typed tolerance Kernel._verify() applies to
        AgentLoopResult.last_result (see core/kernel/kernel.py and
        tests/kernel/test_kernel.py's own explanatory comment on that
        pattern) -- this lets a lightweight test stand-in exercise the
        completeness checks below without constructing a full, real
        SecurityDecision object graph.

        Question 4 is answered from `security_decision.authorization.
        effective_risk`, never `security_decision.risk.level` --
        `risk.level` is the raw, pre-permission-floor RiskEngine
        assessment (see Pass 2 finding A / AuthorizationEngine.
        authorize()'s own comments), and a permission is free to
        declare a *more conservative* risk_level than that raw
        heuristic would guess. Reporting the raw value here would
        misstate the actual risk level the Security Layer used to
        reach its decision in exactly that case -- the same class of
        bug Pass 2 finding A fixed one layer up, and ToolGateway's own
        `_risk_is_consistent` check already deliberately avoids by
        comparing against `effective_risk` for the same reason.
        """

        if not isinstance(action, str) or not action.strip():
            raise ValueError(
                "action must be a non-empty string."
            )

        if not isinstance(subject, str) or not subject.strip():
            raise ValueError(
                "subject must be a non-empty string."
            )

        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError(
                "tool_id must be a non-empty string."
            )

        risk_level = getattr(
            getattr(security_decision, "authorization", None),
            "effective_risk",
            None,
        )

        if risk_level is None:
            raise ValueError(
                "security_decision has no authorization.effective_risk "
                "-- cannot answer POLICY_SPEC.md's External Actions "
                "question 4 without one. Execution must stop rather "
                "than guess (POLICY_SPEC.md: 'If any required "
                "authorization is missing, execution must stop.')."
            )

        approval_required = getattr(
            getattr(security_decision, "approval", None),
            "required",
            None,
        )

        if approval_required is None:
            raise ValueError(
                "security_decision has no approval.required -- cannot "
                "answer POLICY_SPEC.md's External Actions question 5 "
                "without one. Execution must stop rather than guess "
                "(POLICY_SPEC.md: 'If any required authorization is "
                "missing, execution must stop.')."
            )

        return ExternalActionEvaluation(
            action=action,
            subject=subject,
            tool_id=tool_id,
            risk_level=risk_level,
            approval_required=bool(approval_required),
            # Kernel v1 already verifies every COMPLETED result
            # unconditionally (KernelVerification, real since Build
            # Phase 4) -- verification is not yet a per-risk policy
            # variable in this project, so this answer is always True,
            # reflecting that actual behavior rather than a
            # risk-dependent rule that doesn't exist yet.
            verification_required=True,
        )

    def evaluate_agent_scope(
        self,
        *,
        subject: str,
        declared_tool_ids: Iterable[str],
        actual_tool_ids: Iterable[str],
    ) -> AgentScopeEvaluation:
        """
        POLICY_SPEC.md's Agent Constraints: "operate only within their
        declared responsibilities" / "never silently expand their
        scope", answered for one agent's real tool registrations
        against that agent's own spec-declared tool-id set. See this
        module's own docstring (AGENT CONSTRAINTS) for exactly what
        this does and does not cover, and AgentScopeEvaluation's own
        docstring for the returned fields.

        This is a pure function of its inputs -- it never inspects a
        live ToolRegistry/AgentCore itself, the same "caller supplies
        already-computed data, this method only evaluates it" shape
        already established by evaluate_external_action() above. Never
        raises for a scope violation itself (`within_scope=False` is
        just data); it is the caller's choice whether an out-of-scope
        registration is fatal -- both build_research_agent() and
        build_writer_agent() (core/agents/research_agent.py,
        core/agents/writer_agent.py) choose to raise a ValueError
        immediately when it is, per this module's own docstring.
        """

        if not isinstance(subject, str) or not subject.strip():
            raise ValueError(
                "subject must be a non-empty string."
            )

        declared = frozenset(declared_tool_ids)
        actual = frozenset(actual_tool_ids)
        unauthorized = actual - declared

        return AgentScopeEvaluation(
            subject=subject,
            declared_tool_ids=declared,
            actual_tool_ids=actual,
            unauthorized_tool_ids=unauthorized,
            within_scope=not unauthorized,
        )
