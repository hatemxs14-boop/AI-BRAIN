
from __future__ import annotations

from dataclasses import dataclass

from .risk_engine import RiskLevel


@dataclass(frozen=True)
class ApprovalDecision:
    required: bool
    approval_type: str
    reason: str


@dataclass(frozen=True)
class ApprovalRequest:
    """
    Represents a request that requires an additional approval
    before execution.
    """

    approval_type: str
    reason: str


@dataclass(frozen=True)
class ApprovalResult:
    """
    Represents the result of an approval decision.

    approved:
        True  -> the requested action may continue.
        False -> the requested action must not continue.

    approved_by:
        Identifies who or what granted the approval.
        This is informational and must not be trusted as a permission
        by itself.
    """

    approved: bool
    approved_by: str | None
    reason: str


class ApprovalGate:
    """
    Determines whether an authorized capability requires
    additional approval before execution.

    This layer does not grant permissions.

    It provides two separate responsibilities:

    1. evaluate()
       Determines whether approval is required based on risk.

    2. resolve()
       Resolves an existing approval request after an explicit
       approval decision has been supplied.

    Permission and risk evaluation remain independent from approval.
    """

    def evaluate(self, risk_level: RiskLevel) -> ApprovalDecision:

        if risk_level == RiskLevel.LOW:
            return ApprovalDecision(
                required=False,
                approval_type="none",
                reason="LOW-risk operation does not require approval.",
            )

        if risk_level == RiskLevel.MEDIUM:
            return ApprovalDecision(
                required=False,
                approval_type="none",
                reason=(
                    "MEDIUM-risk operation may execute automatically "
                    "within controls."
                ),
            )

        if risk_level == RiskLevel.HIGH:
            return ApprovalDecision(
                required=True,
                approval_type="policy",
                reason=(
                    "HIGH-risk operation requires an applicable "
                    "approval policy."
                ),
            )

        if risk_level == RiskLevel.CRITICAL:
            return ApprovalDecision(
                required=True,
                approval_type="human",
                reason=(
                    "CRITICAL operation requires explicit "
                    "human approval."
                ),
            )

        # Defensive fail-closed behavior.
        return ApprovalDecision(
            required=True,
            approval_type="human",
            reason="Unknown risk level; approval required.",
        )

    def create_request(
        self,
        approval_decision: ApprovalDecision,
    ) -> ApprovalRequest | None:
        """
        Convert an ApprovalDecision into an explicit approval request.

        Returns None when no additional approval is required.
        """

        if not approval_decision.required:
            return None

        return ApprovalRequest(
            approval_type=approval_decision.approval_type,
            reason=approval_decision.reason,
        )

    def resolve(
        self,
        approval_decision: ApprovalDecision,
        *,
        approved: bool,
        approved_by: str | None = None,
    ) -> ApprovalResult:
        """
        Resolve an approval decision.

        Security rules:

        - LOW/MEDIUM operations do not require approval.
        - HIGH operations require an applicable approval decision.
        - CRITICAL operations require explicit human approval.
        - Unknown or invalid approval states fail closed.
        - Approval cannot create permission that did not already exist.
        """

        if not approval_decision.required:
            return ApprovalResult(
                approved=True,
                approved_by=None,
                reason="No additional approval is required.",
            )

        if not approved:
            return ApprovalResult(
                approved=False,
                approved_by=approved_by,
                reason="Approval was explicitly denied.",
            )

        if approval_decision.approval_type == "human":
            if not approved_by:
                return ApprovalResult(
                    approved=False,
                    approved_by=None,
                    reason=(
                        "Human approval requires an explicit "
                        "approver identity."
                    ),
                )

        elif approval_decision.approval_type == "policy":
            if not approved_by:
                return ApprovalResult(
                    approved=False,
                    approved_by=None,
                    reason=(
                        "Policy approval requires an explicit "
                        "identity for who or what asserted the "
                        "policy was satisfied."
                    ),
                )

        else:
            return ApprovalResult(
                approved=False,
                approved_by=approved_by,
                reason="Unknown approval type; approval denied.",
            )

        return ApprovalResult(
            approved=True,
            approved_by=approved_by,
            reason="Approval granted.",
        )

