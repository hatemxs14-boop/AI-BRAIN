"""
Regression tests for the "conservative-risk permission" bug class.

AuthorizationEngine.authorize() computes an internal risk floor --
effective_risk = max(independently assessed risk, the matched
permission's own declared risk_level) -- so that a permission
author can mark a capability MORE conservatively than RiskEngine's
keyword heuristic would, without that capability becoming permanently
unauthorizable.

Two other components used to re-derive risk from the raw, pre-floor
assessment instead of using AuthorizationResult.effective_risk:

  * SecurityDecisionPoint computed approval.approval_type from the raw
    risk, so a conservatively-declared permission drove authorization
    into REQUIRE_APPROVAL/DENY while approval_type was still the
    lower tier's -- resolving an explicit, valid approval then failed
    with "Unknown approval type; approval denied."
  * ToolGateway._risk_is_consistent compared a tool's declared
    risk_level against that same raw assessment and denied
    unconditionally whenever the tool honestly declared a higher risk
    than the raw heuristic saw -- even with explicit human approval.

These tests exercise the real ToolGateway/SecurityDecisionPoint/
AuthorizationEngine stack (no mocks) with a tool+permission pair
deliberately chosen so RiskEngine's raw keyword assessment
under-classifies it, to make sure this bug class cannot silently
reappear.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry


def _build_conservative_gateway(tmp_dir: Path):
    """
    A tool + matching permission both honestly declare CRITICAL risk,
    for a resource/action pair ("content_publishing"/"publish") that
    RiskEngine's hardcoded keyword lists do not recognize as dangerous
    on their own (it would otherwise assess this as HIGH via the
    generic unclassified-action fallback -- still short of the
    CRITICAL risk the permission itself declares, which is exactly
    the gap the effective-risk floor exists to close).
    """

    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": "research_agent",
                "resource": "content_publishing",
                "action": "publish",
                "scope": "public",
                "risk_level": "CRITICAL",
                "approval": "human",
            }
        ],
        "defaults": {
            "unknown_risk": "DENY",
            "unknown_permission": "DENY",
            "unknown_scope": "DENY",
            "authorization_failure": "DENY",
        },
    }

    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            id="publish_content",
            name="Publish Content",
            purpose="Publish content publicly.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            output_schema={"type": "string"},
            permissions=(
                "research_agent:content_publishing:publish:public",
            ),
            resource="content_publishing",
            action="publish",
            scope="public",
            risk_level="CRITICAL",
            error_handling={
                "retryable": False,
                "on_failure": "Do not retry publishing automatically.",
            },
        )
    )

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(
        tool_id="publish_content",
        executor=lambda: "published",
    )

    return gateway


def test_conservative_permission_requires_approval_not_denied():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway = _build_conservative_gateway(tmp_dir)

        result = gateway.execute(
            subject="research_agent",
            tool_id="publish_content",
            tool_kwargs={},
        )

        assert result.status == "APPROVAL_REQUIRED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_conservative_permission_succeeds_with_human_approval():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway = _build_conservative_gateway(tmp_dir)

        result = gateway.execute(
            subject="research_agent",
            tool_id="publish_content",
            tool_kwargs={},
            approved=True,
            approved_by="human_operator",
        )

        assert result.status == "SUCCESS"
        assert result.artifacts == ("published",)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_conservative_permission_still_denies_without_valid_approver():
    """
    The floor fix must not accidentally weaken CRITICAL's human-approval
    requirement: approved=True with no approver identity must still be
    denied.
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway = _build_conservative_gateway(tmp_dir)

        result = gateway.execute(
            subject="research_agent",
            tool_id="publish_content",
            tool_kwargs={},
            approved=True,
            approved_by=None,
        )

        assert result.status == "DENIED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------
# ApprovalGate: "policy" approval must require an approver identity,
# same as "human" -- it used to be a rubber stamp on approved=True
# alone, with no accountability for who/what asserted the policy was
# satisfied.
# ---------------------------------------------------------------------

from core.security.engine.approval_gate import (  # noqa: E402
    ApprovalDecision,
    ApprovalGate,
)


def test_policy_approval_requires_an_approver_identity():
    gate = ApprovalGate()

    decision = ApprovalDecision(
        required=True,
        approval_type="policy",
        reason="test",
    )

    result = gate.resolve(decision, approved=True, approved_by=None)

    assert result.approved is False


def test_policy_approval_succeeds_with_an_approver_identity():
    gate = ApprovalGate()

    decision = ApprovalDecision(
        required=True,
        approval_type="policy",
        reason="test",
    )

    result = gate.resolve(
        decision,
        approved=True,
        approved_by="ci_pipeline",
    )

    assert result.approved is True
    assert result.approved_by == "ci_pipeline"


# ---------------------------------------------------------------------
# RiskEngine: credential-adjacent resource names must classify as
# CRITICAL, not fall through to LOW. ssh_credentials was the concrete
# gap found during audit; these guard the expanded vocabulary against
# regressing back to under-classification.
# ---------------------------------------------------------------------

from core.security.engine.risk_engine import RiskEngine, RiskLevel  # noqa: E402


def test_risk_engine_classifies_credential_resources_as_critical():
    engine = RiskEngine()

    for resource in (
        "ssh_credentials",
        "api_key",
        "access_token",
        "private_key",
        "password",
    ):
        assessment = engine.assess(
            resource=resource,
            action="read",
            scope="workspace",
        )

        assert assessment.level == RiskLevel.CRITICAL, (
            f"resource={resource!r} was classified as "
            f"{assessment.level.name}, expected CRITICAL"
        )


def test_risk_engine_existing_classifications_unaffected():
    """
    The vocabulary expansion must not change risk for the tools already
    registered against permissions.json (web_search -> LOW,
    shell/workspace -> HIGH).
    """

    engine = RiskEngine()

    web_search = engine.assess(
        resource="web_search",
        action="search",
        scope="public_web",
    )
    assert web_search.level == RiskLevel.LOW

    shell = engine.assess(
        resource="shell",
        action="execute",
        scope="workspace",
    )
    assert shell.level == RiskLevel.HIGH
