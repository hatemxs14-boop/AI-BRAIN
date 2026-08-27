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
            )

        if not all(
            isinstance(value, str) and value.strip()
            for value in (subject, resource, action, scope)
        ):
            return AuthorizationResult(
                decision=Decision.DENY,
                reason="Malformed authorization request.",
                request=request,
            )

        permissions = self.policy.get("permissions")

        if not isinstance(permissions, list):
            return AuthorizationResult(
                decision=Decision.DENY,
                reason="Invalid permissions policy.",
                request=request,
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
            )

        permission_risk = str(
            matching_permission.get("risk_level", "")
        ).upper()

        if permission_risk not in RiskLevel.__members__:
            return AuthorizationResult(
                decision=Decision.DENY,
                reason="Permission contains an unknown risk level.",
                request=request,
            )

        # The permission's declared risk_level is a floor, not a ceiling:
        # a policy author may mark a capability more conservatively than
        # the independent RiskEngine assessment would. That must make the
        # operation *more* controlled (higher effective risk), never
        # unusable. Denying outright whenever the assessed risk happens to
        # be lower than the declared permission risk would make any
        # deliberately-conservative permission permanently unauthorizable,
        # even with human approval, which contradicts the security model.
        effective_risk = max(
            self._risk_value(normalized_risk),
            self._risk_value(permission_risk),
        )

        decision = self._decision_for_risk(effective_risk)

        return AuthorizationResult(
            decision=decision,
            reason="Explicit permission matched.",
            request=request,
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
    def _decision_for_risk(risk_value: int) -> Decision:
        if risk_value == 1:
            return Decision.ALLOW

        if risk_value == 2:
            return Decision.ALLOW_WITH_CONTROLS

        if risk_value == 3:
            return Decision.REQUIRE_APPROVAL

        return Decision.REQUIRE_APPROVAL