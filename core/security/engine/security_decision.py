
from __future__ import annotations

from dataclasses import dataclass

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

    Approval never grants permission.
    Authorization must succeed before an approval can result in ALLOW.
    """

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
    ) -> SecurityDecision:
        """
        Evaluate an operation without supplying an approval decision.

        LOW/MEDIUM operations can be allowed automatically when explicitly
        authorized.

        HIGH/CRITICAL operations return REQUIRE_APPROVAL when authorization
        succeeds but an approval boundary is required.

        Unauthorized operations always return DENY.
        """

        return self._evaluate(
            subject=subject,
            resource=resource,
            action=action,
            scope=scope,
            approved=None,
            approved_by=None,
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
        """

        return self._evaluate(
            subject=subject,
            resource=resource,
            action=action,
            scope=scope,
            approved=approved,
            approved_by=approved_by,
        )

    def _evaluate(
        self,
        *,
        subject: str,
        resource: str,
        action: str,
        scope: str,
        approved: bool | None,
        approved_by: str | None,
    ) -> SecurityDecision:
        # 1. Determine the actual risk independently.
        risk = self.risk_engine.assess(
            resource=resource,
            action=action,
            scope=scope,
        )

        # 2. Check explicit authorization.
        authorization = self.authorization_engine.authorize(
            subject=subject,
            resource=resource,
            action=action,
            scope=scope,
            risk_level=risk.level.name,
        )

        # 3. Determine whether additional approval is required.
        approval = self.approval_gate.evaluate(risk.level)

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

        self.audit_logger.record(audit_event)

        return result

