
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .approval_gate import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalResult,
)
from .authorization import (
    AuthorizationEngine,
    AuthorizationResult,
    Decision,
)
from .audit_logger import AuditLogger
from .risk_engine import RiskAssessment, RiskEngine
from .risk_engine import RiskLevel as EngineRiskLevel


@dataclass(frozen=True)
class SecurityDecision:
    decision: Decision
    risk: RiskAssessment
    authorization: AuthorizationResult
    approval: ApprovalDecision
    approval_result: ApprovalResult | None = None


class SecurityDecisionPoint:
    """
    Central security decision point.

    The Security Layer independently:

    1. Determines risk.
    2. Checks explicit authorization.
    3. Determines whether additional approval is required.
    4. Optionally resolves an explicit approval decision.
    5. Produces the final security decision.
    6. Records the complete decision in the audit log.

    A caller that goes on to actually attempt the operation this
    decision authorized (the Tool Gateway, after `evaluate()`/
    `evaluate_with_approval()` returns ALLOW/ALLOW_WITH_CONTROLS) is
    expected to report back what happened via
    `record_execution_outcome()`, so the audit trail captures not
    just what was decided but what actually occurred -- see that
    method's own docstring.

    Approval never grants permission.
    Authorization must succeed before an approval can result in ALLOW.
    """

    # SECURITY_SPEC.md's "Audit Logging" section requires the audit
    # trail to distinguish between exactly these five outcomes, "so
    # the system can determine whether an operation was merely
    # requested or actually executed." This maps every real
    # ToolExecutionResult.status (core/tools/engine/tool_gateway.py)
    # onto that spec vocabulary -- see record_execution_outcome()'s
    # own docstring for why this mapping is a second, distinct audit
    # event rather than a field folded into the existing one.
    #
    #   SUCCESS           -> executed  (the private executor ran and
    #                        its output passed validation)
    #   DENIED             -> blocked   (the Security Layer, risk-
    #                        consistency check, or registry/executor
    #                        lookup stopped it before it could run)
    #   APPROVAL_REQUIRED  -> requested (submitted, but still waiting
    #                        on a human decision -- neither executed
    #                        nor blocked yet)
    #   INVALID_INPUT      -> failed    (the request itself was
    #                        malformed; the private executor was
    #                        never even reached)
    #   INVALID_OUTPUT     -> failed    (the executor DID run, but its
    #                        output could not be trusted)
    #   ERROR              -> failed    (the executor ran and raised)
    #
    # Any tool_status this table doesn't recognize degrades to
    # "failed" rather than raising -- this method records a fact
    # about a tool call that has *already happened*; it must never be
    # the reason that call's own result fails to be returned to its
    # caller (the standing "never so strict it refuses to execute or
    # accept something" constraint applies to audit bookkeeping too).
    _EXECUTION_STATUS_BY_TOOL_STATUS: dict[str, str] = {
        "SUCCESS": "executed",
        "DENIED": "blocked",
        "APPROVAL_REQUIRED": "requested",
        "INVALID_INPUT": "failed",
        "INVALID_OUTPUT": "failed",
        "ERROR": "failed",
    }

    def __init__(
        self,
        policy_path: str,
        audit_log_path: str = "logs/security/audit.jsonl",
    ):
        self.risk_engine = RiskEngine()
        self.authorization_engine = AuthorizationEngine(policy_path)
        self.approval_gate = ApprovalGate()
        self.audit_logger = AuditLogger(audit_log_path)

    def evaluate(
        self,
        *,
        subject: str,
        resource: str,
        action: str,
        scope: str,
        metadata: dict | None = None,
    ) -> SecurityDecision:
        """
        Evaluate an operation without supplying an approval decision.

        LOW/MEDIUM operations can be allowed automatically when explicitly
        authorized.

        HIGH/CRITICAL operations return REQUIRE_APPROVAL when authorization
        succeeds but an approval boundary is required.

        Unauthorized operations always return DENY.

        `metadata` is caller-supplied context (e.g. a correlation ID or
        approval justification attached to the originating tool
        invocation). It is informational only -- it never affects the
        decision -- and is recorded in the audit trail so it isn't
        silently lost between the caller and the audit log.
        """

        return self._evaluate(
            subject=subject,
            resource=resource,
            action=action,
            scope=scope,
            approved=None,
            approved_by=None,
            metadata=metadata,
        )

    def evaluate_with_approval(
        self,
        *,
        subject: str,
        resource: str,
        action: str,
        scope: str,
        approved: bool,
        approved_by: str | None = None,
        metadata: dict | None = None,
    ) -> SecurityDecision:
        """
        Evaluate an operation with an explicit approval decision.

        Approval can only resolve an existing approval requirement.

        It cannot:

        - grant a missing permission,
        - reduce risk,
        - expand scope,
        - bypass authorization,
        - bypass CRITICAL human-approval requirements.

        `metadata` is informational only; see `evaluate()`.
        """

        return self._evaluate(
            subject=subject,
            resource=resource,
            action=action,
            scope=scope,
            approved=approved,
            approved_by=approved_by,
            metadata=metadata,
        )

    def record_execution_outcome(
        self,
        *,
        security_decision: SecurityDecision,
        tool_id: str,
        tool_status: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record what actually happened after a security decision was
        made, as a second, distinct audit event.

        `_evaluate()` already records one audit event per call, at
        the moment the security *decision* is made -- but that event
        is written BEFORE the Tool Gateway attempts to execute
        anything. An ALLOW decision was previously audited identically
        whether the private executor then actually ran successfully,
        crashed, produced output that failed validation, or never got
        a chance to run at all because the tool wasn't registered or
        had no executor -- SECURITY_SPEC.md's Audit Logging section
        names exactly this gap: "Audit records must distinguish
        between: requested / authorized / executed / blocked /
        failed. This allows the system to determine whether an
        operation was merely requested or actually executed."

        This is deliberately a SECOND event rather than a field
        appended after the fact to the first one: audit records must
        be append-only (AuditLogger never rewrites a line once
        written), and the first event is what "was this operation
        even authorized" answers on its own -- conflating the two
        would lose the ability to tell, from the first record alone,
        that a decision was made at all before its outcome was known.

        `security_decision` supplies subject/resource/action/scope by
        reading `security_decision.authorization.request` -- the same
        AuthorizationRequest the original decision was made from --
        rather than accepting them as separate parameters that could
        drift from what was actually evaluated.

        `tool_status` is the real ToolExecutionResult.status
        (core/tools/engine/tool_gateway.py); it is translated to the
        spec's five-term vocabulary via
        `_EXECUTION_STATUS_BY_TOOL_STATUS` and recorded as
        `execution_status`, alongside the raw `tool_status` itself so
        neither the original detail nor the spec-level category is
        lost.

        `metadata` mirrors `evaluate()`/`evaluate_with_approval()`'s
        own `metadata` parameter -- passing the same caller-supplied
        value here lets both audit events for one tool call be
        correlated without relying on timestamp proximity.
        """

        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError("tool_id must be a non-empty string.")

        if not isinstance(tool_status, str) or not tool_status.strip():
            raise ValueError("tool_status must be a non-empty string.")

        request = security_decision.authorization.request

        execution_status = self._EXECUTION_STATUS_BY_TOOL_STATUS.get(
            tool_status, "failed"
        )

        audit_event = {
            "event": "execution_outcome",
            "subject": request.subject,
            "resource": request.resource,
            "action": request.action,
            "scope": request.scope,
            "tool_id": tool_id,
            "decision": security_decision.decision.value,
            "tool_status": tool_status,
            "execution_status": execution_status,
            "summary": summary,
        }

        if metadata is not None:
            audit_event["metadata"] = metadata

        self.audit_logger.record(audit_event)

    def _evaluate(
        self,
        *,
        subject: str,
        resource: str,
        action: str,
        scope: str,
        approved: bool | None,
        approved_by: str | None,
        metadata: dict | None = None,
    ) -> SecurityDecision:
        # 1. Determine the actual risk independently.
        risk = self.risk_engine.assess(
            resource=resource,
            action=action,
            scope=scope,
        )

        # 2. Check explicit authorization. This also computes the
        #    effective risk: the independently-assessed risk, floored
        #    (raised, never lowered) by the matched permission's own
        #    declared risk_level when one applies.
        authorization = self.authorization_engine.authorize(
            subject=subject,
            resource=resource,
            action=action,
            scope=scope,
            risk_level=risk.level.name,
        )

        # 3. Determine whether additional approval is required, using
        #    the SAME effective risk that just determined the
        #    authorization decision -- never the pre-floor raw
        #    assessment. Using the raw assessment here let a
        #    deliberately-conservative permission (declared risk higher
        #    than RiskEngine's raw classification) drive authorization
        #    into REQUIRE_APPROVAL/DENY while `approval.approval_type`
        #    was still computed for the lower, un-floored tier -- e.g.
        #    "none" or "policy" when the real tier was CRITICAL/"human"
        #    -- which made `approval_gate.resolve()` reject even an
        #    explicit, valid human approval with "Unknown approval
        #    type; approval denied."
        approval = self.approval_gate.evaluate(
            EngineRiskLevel[authorization.effective_risk]
        )

        approval_result: ApprovalResult | None = None

        # 4. Authorization denial always results in DENY.
        #
        # Approval can never override missing authorization.
        if authorization.decision == Decision.DENY:
            final_decision = Decision.DENY

        # 5. If the authorization engine itself requires approval,
        #    resolve that requirement only when an approval decision
        #    has explicitly been supplied.
        elif authorization.decision == Decision.REQUIRE_APPROVAL:
            if approved is None:
                final_decision = Decision.REQUIRE_APPROVAL
            else:
                authorization_approval = self.approval_gate.resolve(
                    ApprovalDecision(
                        required=True,
                        approval_type=approval.approval_type,
                        reason=authorization.reason,
                    ),
                    approved=approved,
                    approved_by=approved_by,
                )

                approval_result = authorization_approval

                if authorization_approval.approved:
                    final_decision = Decision.ALLOW
                else:
                    final_decision = Decision.DENY

        # 6. Risk-based approval boundary.
        elif approval.required:
            if approved is None:
                final_decision = Decision.REQUIRE_APPROVAL
            else:
                approval_result = self.approval_gate.resolve(
                    approval,
                    approved=approved,
                    approved_by=approved_by,
                )

                if approval_result.approved:
                    final_decision = Decision.ALLOW
                else:
                    final_decision = Decision.DENY

        # 7. No additional approval is required.
        #
        # authorization.decision here is either ALLOW or
        # ALLOW_WITH_CONTROLS (REQUIRE_APPROVAL/DENY were already
        # handled above, and approval.required is False in this
        # branch). Preserve ALLOW_WITH_CONTROLS instead of collapsing
        # it into plain ALLOW: a MEDIUM-risk operation is meant to
        # execute automatically *with controls*, and losing that
        # distinction here made it invisible to every consumer of
        # `SecurityDecision.decision` -- including the audit log's own
        # "decision" field, which previously reported "ALLOW" even
        # when AuthorizationEngine had actually decided
        # ALLOW_WITH_CONTROLS. ToolGateway treats both the same for
        # execution purposes (see `_risk_is_consistent`/the DENY
        # checks in `execute()`), so preserving the distinction here
        # does not change whether anything runs -- only whether the
        # true security posture is honestly recorded.
        else:
            if authorization.decision == Decision.ALLOW_WITH_CONTROLS:
                final_decision = Decision.ALLOW_WITH_CONTROLS
            else:
                final_decision = Decision.ALLOW

        result = SecurityDecision(
            decision=final_decision,
            risk=risk,
            authorization=authorization,
            approval=approval,
            approval_result=approval_result,
        )

        # 8. Record the complete security decision.
        audit_event = {
            "event": "security_decision",
            "subject": subject,
            "resource": resource,
            "action": action,
            "scope": scope,
            "risk_level": risk.level.name,
            "risk_reasons": list(risk.reasons),
            "effective_risk": authorization.effective_risk,
            "authorization": authorization.decision.value,
            "authorization_reason": authorization.reason,
            "approval_required": approval.required,
            "approval_type": approval.approval_type,
            "approval_reason": approval.reason,
            "decision": final_decision.value,
        }

        if approval_result is not None:
            audit_event.update(
                {
                    "approval_result": (
                        "ALLOW"
                        if approval_result.approved
                        else "DENY"
                    ),
                    "approved_by": approval_result.approved_by,
                    "approval_result_reason": approval_result.reason,
                }
            )

        if metadata is not None:
            audit_event["metadata"] = metadata

        self.audit_logger.record(audit_event)

        return result

