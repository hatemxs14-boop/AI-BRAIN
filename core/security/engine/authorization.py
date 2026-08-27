from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Decision(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_WITH_CONTROLS = "ALLOW_WITH_CONTROLS"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class AuthorizationRequest:
    subject: str
    resource: str
    action: str
    scope: str
    risk_level: str


@dataclass(frozen=True)
class AuthorizationResult:
    decision: Decision
    reason: str
    request: AuthorizationRequest

    # The risk level that actually drove `decision` -- the independently
    # assessed risk floored (raised, never lowered) by the matched
    # permission's own declared risk_level, when one was matched.
    #
    # Every downstream consumer that needs to know "how risky did the
    # Security Layer ultimately treat this operation" (approval-type
    # selection, tool/permission risk-consistency checks, audit records)
    # MUST read this field instead of re-deriving risk on its own --
    # otherwise it can disagree with the decision actually made here,
    # which is exactly the class of bug this field exists to prevent.
    effective_risk: str


class AuthorizationEngine:
    """
    Central security decision point for AI-BRAIN.

    This engine is deliberately fail-closed:
    anything unknown, malformed, or outside the declared policy is denied.
    """

    def __init__(self, policy_path: str | Path):
        self.policy_path = Path(policy_path)
        self.policy = self._load_policy()

    def _load_policy(self) -> dict[str, Any]:
        if not self.policy_path.exists():
            raise FileNotFoundError(
                f"Security policy not found: {self.policy_path}"
            )

        try:
            with self.policy_path.open("r", encoding="utf-8") as file:
                policy = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unable to load security policy: {self.policy_path}"
            ) from exc

        if not isinstance(policy, dict):
            raise ValueError("Security policy must be a JSON object.")

        return policy

    def authorize(
        self,
        subject: str,
        resource: str,
        action: str,
        scope: str,
        risk_level: str,
    ) -> AuthorizationResult:

        request = AuthorizationRequest(
            subject=subject,
            resource=resource,
            action=action,
            scope=scope,
            risk_level=risk_level,
        )

        normalized_risk = risk_level.upper().strip()

        if normalized_risk not in RiskLevel.__members__:
            return AuthorizationResult(
                decision=Decision.DENY,
                reason="Unknown risk level.",
                request=request,
                # The input risk classification itself is invalid, so
                # there is nothing valid to float a floor from. Fail
                # closed to the strictest tier rather than guessing --
                # this value is only consumed for audit/approval-type
                # bookkeeping on an already-DENYed request.
                effective_risk="CRITICAL",
            )

        if not all(
            isinstance(value, str) and value.strip()
            for value in (subject, resource, action, scope)
        ):
            return AuthorizationResult(
                decision=Decision.DENY,
                reason="Malformed authorization request.",
                request=request,
                effective_risk=normalized_risk,
            )

        permissions = self.policy.get("permissions")

        if not isinstance(permissions, list):
            return AuthorizationResult(
                decision=Decision.DENY,
                reason="Invalid permissions policy.",
                request=request,
                effective_risk=normalized_risk,
            )

        matching_permission = self._find_matching_permission(
            subject=subject,
            resource=resource,
            action=action,
            scope=scope,
        )

        if matching_permission is None:
            return AuthorizationResult(
                decision=Decision.DENY,
                reason="No explicit permission grants this capability.",
                request=request,
                effective_risk=normalized_risk,
            )

        permission_risk = str(
            matching_permission.get("risk_level", "")
        ).upper()

        if permission_risk not in RiskLevel.__members__:
            return AuthorizationResult(
                decision=Decision.DENY,
                reason="Permission contains an unknown risk level.",
                request=request,
                effective_risk=normalized_risk,
            )

        # The permission's declared risk_level is a floor, not a ceiling:
        # a policy author may mark a capability more conservatively than
        # the independent RiskEngine assessment would. That must make the
        # operation *more* controlled (higher effective risk), never
        # unusable. Denying outright whenever the assessed risk happens to
        # be lower than the declared permission risk would make any
        # deliberately-conservative permission permanently unauthorizable,
        # even with human approval, which contradicts the security model.
        effective_risk_value = max(
            self._risk_value(normalized_risk),
            self._risk_value(permission_risk),
        )

        decision = self._decision_for_risk(effective_risk_value)

        return AuthorizationResult(
            decision=decision,
            reason="Explicit permission matched.",
            request=request,
            effective_risk=self._risk_name(effective_risk_value),
        )

    def _find_matching_permission(
        self,
        subject: str,
        resource: str,
        action: str,
        scope: str,
    ) -> dict[str, Any] | None:

        for permission in self.policy.get("permissions", []):
            if not isinstance(permission, dict):
                continue

            if permission.get("subject") != subject:
                continue

            if permission.get("resource") != resource:
                continue

            if permission.get("action") != action:
                continue

            if permission.get("scope") != scope:
                continue

            return permission

        return None

    @staticmethod
    def _risk_value(risk_level: str) -> int:
        values = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

        return values[risk_level]

    @staticmethod
    def _risk_name(risk_value: int) -> str:
        names = {
            1: "LOW",
            2: "MEDIUM",
            3: "HIGH",
            4: "CRITICAL",
        }

        return names[risk_value]

    @staticmethod
    def _decision_for_risk(risk_value: int) -> Decision:
        if risk_value == 1:
            return Decision.ALLOW

        if risk_value == 2:
            return Decision.ALLOW_WITH_CONTROLS

        if risk_value == 3:
            return Decision.REQUIRE_APPROVAL

        return Decision.REQUIRE_APPROVAL