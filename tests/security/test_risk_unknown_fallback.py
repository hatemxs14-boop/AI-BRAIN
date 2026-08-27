"""
Regression tests for RiskEngine's "genuinely unclassified" fallback.

SECURITY_SPEC.md's own risk model treats a resource/action combination
that matches none of RiskEngine's explicit rules as the UNKNOWN case
("UNKNOWN -> deny"; "fail closed when authorization is uncertain").
RiskEngine previously defaulted this case to MEDIUM, which could
auto-execute without any approval step -- silently treating "we don't
recognize this at all" the same as "we recognize this and it's
routine". That fallback is now HIGH: nothing unrecognized executes
silently, it requires at least a policy check-in first -- but,
deliberately, it is NOT a hard DENY. A literal UNKNOWN tier that
denies unconditionally would make any new, not-yet-vocabularied tool
or action permanently unauthorizable even with explicit approval --
already the root cause of two earlier bugs in this project
(Pass 1 fix #4, Pass 2 finding A) -- which is exactly the "too strict
to execute anything" failure mode this change must avoid.

These tests confirm both halves: the fallback is now more
conservative (HIGH, not MEDIUM), and a legitimate, explicitly
permitted use of an unrecognized action can still execute once
approved -- it is not locked out forever.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from core.security.engine.risk_engine import RiskEngine, RiskLevel
from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry


def test_unclassified_action_now_assesses_as_high_not_medium():
    engine = RiskEngine()

    # "archive" on a made-up resource matches none of RiskEngine's
    # explicit rules: not a critical action, not a sensitive resource,
    # not shell, not write/modify, not connect/send/upload, not
    # read/inspect/retrieve, and not in the safe read-only action set.
    assessment = engine.assess(
        resource="knowledge_base_snapshot",
        action="archive",
        scope="workspace",
    )

    assert assessment.level == RiskLevel.HIGH


def test_read_only_actions_remain_low_after_the_fallback_change():
    """
    Non-regression: the safe read-only branch must still take
    priority over the generic fallback for the actions it explicitly
    lists.
    """

    engine = RiskEngine()

    for action in ("read", "search", "analyze", "calculate", "inspect"):
        assessment = engine.assess(
            resource="knowledge_base_snapshot",
            action=action,
            scope="workspace",
        )

        assert assessment.level == RiskLevel.LOW, (
            f"action={action!r} was classified as "
            f"{assessment.level.name}, expected LOW"
        )


def test_unrecognized_but_explicitly_permitted_action_is_not_permanently_denied():
    """
    The "not too strict" guarantee: an operation RiskEngine cannot
    classify, but that an administrator has explicitly authorized via
    permissions.json, must still be executable -- automatically once
    approved, never permanently blocked outright.
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy = {
            "version": "1.0",
            "permissions": [
                {
                    "subject": "research_agent",
                    "resource": "knowledge_base_snapshot",
                    "action": "archive",
                    "scope": "workspace",
                    "risk_level": "HIGH",
                    "approval": "policy",
                }
            ],
        }

        policy_path = tmp_dir / "permissions.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                id="archive_snapshot",
                name="Archive Knowledge Base Snapshot",
                purpose="Archive a knowledge base snapshot.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                output_schema={"type": "string"},
                permissions=(
                    "research_agent:knowledge_base_snapshot:archive:"
                    "workspace",
                ),
                resource="knowledge_base_snapshot",
                action="archive",
                scope="workspace",
                risk_level="HIGH",
                error_handling={
                    "retryable": False,
                    "on_failure": "Surface the archive failure for review.",
                },
            )
        )

        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)
        gateway.register_executor(
            tool_id="archive_snapshot",
            executor=lambda: "archived",
        )

        # Without approval: requires approval, but is not denied outright.
        pending = gateway.execute(
            subject="research_agent",
            tool_id="archive_snapshot",
            tool_kwargs={},
        )
        assert pending.status == "APPROVAL_REQUIRED"

        # With an explicit policy approval: it actually runs.
        approved = gateway.execute(
            subject="research_agent",
            tool_id="archive_snapshot",
            tool_kwargs={},
            approved=True,
            approved_by="release_policy_bot",
        )
        assert approved.status == "SUCCESS"
        assert approved.artifacts == ("archived",)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
