from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class RiskLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    reasons: tuple[str, ...]


class RiskEngine:
    """
    Determines the effective risk of an operation.

    The caller cannot lower the risk assigned by this engine.
    Risk is based on the actual capability being requested.
    """

    def assess(
        self,
        *,
        resource: str,
        action: str,
        scope: str,
        requested_risk: str | None = None,
    ) -> RiskAssessment:

        resource_value = resource.lower().strip()
        action_value = action.lower().strip()
        scope_value = scope.lower().strip()

        levels: list[RiskLevel] = []
        reasons: list[str] = []

        # Explicitly dangerous actions.
        if action_value in {
            "delete",
            "destroy",
            "disable_security",
            "grant_permission",
            "change_privilege",
            "extract_secret",
            "transfer_funds",
        }:
            levels.append(RiskLevel.CRITICAL)
            reasons.append(f"Critical action: {action}")

        # Sensitive resources.
        #
        # This is necessarily a finite, hand-maintained vocabulary --
        # anything not listed here falls through to the generic
        # buckets below instead of CRITICAL. Reviewed and expanded
        # once already after a real gap was found during audit
        # (resource="ssh_credentials" fell through to LOW): the set
        # below now covers the common synonyms for "this resource IS a
        # secret", but any new resource name a tool author invents
        # still needs to be added here explicitly. When adding a new
        # sensitive tool, check this set first rather than assuming
        # RiskEngine will "figure it out" from the name.
        if resource_value in {
            "secrets",
            "credentials",
            "ssh_credentials",
            "ssh_keys",
            "api_key",
            "api_keys",
            "access_token",
            "access_tokens",
            "auth_token",
            "auth_tokens",
            "private_key",
            "private_keys",
            "password",
            "passwords",
            "production_data",
            "security_policy",
            "system_privileges",
        }:
            levels.append(RiskLevel.CRITICAL)
            reasons.append(f"Sensitive resource: {resource}")

        # External effects.
        if resource_value in {
            "external_communication",
            "remote_repository",
            "external_network",
            "deployment",
        }:
            levels.append(RiskLevel.HIGH)
            reasons.append(f"External effect: {resource}")

        # Shell execution.
        if resource_value in {
            "shell",
            "terminal",
            "command_execution",
        }:
            if scope_value in {
                "unrestricted",
                "system",
                "host",
                "root",
            }:
                levels.append(RiskLevel.CRITICAL)
                reasons.append("Unrestricted system-level execution.")
            else:
                levels.append(RiskLevel.HIGH)
                reasons.append("Shell or command execution requested.")

        # File writes.
        if action_value in {"write", "modify"}:
            if scope_value in {
                "production",
                "system",
                "outside_workspace",
            }:
                levels.append(RiskLevel.HIGH)
                reasons.append("Write operation outside a normal workspace.")
            else:
                levels.append(RiskLevel.MEDIUM)
                reasons.append("Project state can be modified.")

        # Network access.
        if action_value in {"connect", "send", "upload"}:
            if scope_value in {
                "unrestricted",
                "arbitrary",
                "internet",
            }:
                levels.append(RiskLevel.HIGH)
                reasons.append("Broad external network access.")
            else:
                levels.append(RiskLevel.MEDIUM)
                reasons.append("External network interaction requested.")

        # Reading sensitive information.
        if action_value in {"read", "inspect", "retrieve"}:
            if resource_value in {
                "secrets",
                "credentials",
                "ssh_credentials",
                "ssh_keys",
                "api_key",
                "api_keys",
                "access_token",
                "access_tokens",
                "auth_token",
                "auth_tokens",
                "private_key",
                "private_keys",
                "password",
                "passwords",
                "private_data",
                "personal_data",
            }:
                levels.append(RiskLevel.CRITICAL)
                reasons.append("Sensitive information access.")

        # Safe read-only operations.
        if not levels:
            if action_value in {
                "read",
                "search",
                "analyze",
                "calculate",
                "inspect",
            }:
                levels.append(RiskLevel.LOW)
                reasons.append("Read-only or analytical operation.")
            else:
                levels.append(RiskLevel.MEDIUM)
                reasons.append(
                    "Operation is not explicitly classified as low risk."
                )

        effective_level = max(levels)

        # The caller may request a HIGHER risk classification,
        # but can never downgrade the engine's assessment.
        if requested_risk:
            requested = requested_risk.upper().strip()

            requested_levels = {
                "LOW": RiskLevel.LOW,
                "MEDIUM": RiskLevel.MEDIUM,
                "HIGH": RiskLevel.HIGH,
                "CRITICAL": RiskLevel.CRITICAL,
            }

            if requested in requested_levels:
                requested_level = requested_levels[requested]

                if requested_level > effective_level:
                    effective_level = requested_level
                    reasons.append(
                        "Caller requested a higher risk classification."
                    )

        return RiskAssessment(
            level=effective_level,
            reasons=tuple(reasons),
        )