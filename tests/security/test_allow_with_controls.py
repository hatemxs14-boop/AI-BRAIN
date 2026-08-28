"""
Regression tests for `Decision.ALLOW_WITH_CONTROLS`.

`AuthorizationEngine.authorize()` returns a distinct
`Decision.ALLOW_WITH_CONTROLS` for MEDIUM-risk operations (see
`_decision_for_risk`) -- SECURITY_SPEC.md's own risk model calls this
out explicitly ("MEDIUM -> allow with controls"). `SecurityDecisionPoint
._evaluate()` used to silently rewrite this into plain `Decision.ALLOW`
whenever no approval was required, which made the distinction invisible
to every downstream consumer -- most importantly the audit log's own
"decision" field, which reported "ALLOW" even though the real decision
was ALLOW_WITH_CONTROLS.

These tests exercise the real ToolGateway/SecurityDecisionPoint/
AuthorizationEngine stack (no mocks) against a MEDIUM-risk permission
and confirm:

  1. The tool still executes successfully -- preserving the distinct
     decision value must never block anything that used to run (that
     would be the exact "too strict to execute anything" failure mode
     this project has already hit, and fixed, twice before).
  2. `SecurityDecision.decision` and the recorded audit event both
     honestly report ALLOW_WITH_CONTROLS instead of collapsing it to
     ALLOW.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from core.security.engine.authorization import Decision
from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry


def _build_medium_risk_gateway(tmp_dir: Path):
    """
    A tool + matching permission both declare MEDIUM risk, for a
    resource/action pair RiskEngine's hardcoded keyword lists
    genuinely can't classify as anything more specific ("bounded
    automated analysis"-shaped, not a critical action, not a sensitive
    resource, not shell, not a write/network action, not a
    read-sensitive combination). This exercises the *raw* MEDIUM path,
    where the permission's own declared risk agrees with (rather than
    overrides) RiskEngine's independent assessment via the generic
    unclassified fallback bumped to HIGH elsewhere -- so here we pick
    an action RiskEngine explicitly classifies as MEDIUM on its own
    (a "write"/"modify" action inside the ordinary workspace scope).
    """

    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": "research_agent",
                "resource": "project_files",
                "action": "write",
                "scope": "workspace",
                "risk_level": "MEDIUM",
                "approval": "none",
            }
        ],
    }

    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            id="write_file",
            name="Write File",
            purpose="Write a file inside the workspace.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            output_schema={"type": "string"},
            permissions=(
                "research_agent:project_files:write:workspace",
            ),
            resource="project_files",
            action="write",
            scope="workspace",
            risk_level="MEDIUM",
            error_handling={
                "retryable": True,
                "max_retries": 1,
                "on_failure": "Surface the write failure to the agent.",
            },
        )
    )

    audit_log_path = tmp_dir / "audit.jsonl"

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(audit_log_path),
    )

    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(
        tool_id="write_file",
        executor=lambda: "written",
    )

    return gateway, audit_log_path


def _read_last_audit_event(audit_log_path: Path) -> dict:
    """
    Return the last recorded "security_decision" audit event.

    Build Phase 13 made ToolGateway.execute() record a SECOND, later
    audit event per call ("execution_outcome", via
    SecurityDecisionPoint.record_execution_outcome()) reporting what
    actually happened after the security decision below was made --
    see that method's own docstring. Filtering by event type here
    (rather than assuming the security-decision event is simply the
    last line) keeps this helper correct regardless of how many
    distinct audit event types a given call produces.
    """
    lines = audit_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    security_decision_events = [
        event for event in events if event.get("event") == "security_decision"
    ]
    return security_decision_events[-1]


def test_medium_risk_still_executes_automatically():
    """
    Non-regression: preserving ALLOW_WITH_CONTROLS must not newly
    block a MEDIUM-risk operation that used to auto-execute.
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, _ = _build_medium_risk_gateway(tmp_dir)

        result = gateway.execute(
            subject="research_agent",
            tool_id="write_file",
            tool_kwargs={},
        )

        assert result.status == "SUCCESS"
        assert result.artifacts == ("written",)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_medium_risk_decision_is_reported_as_allow_with_controls():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, _ = _build_medium_risk_gateway(tmp_dir)

        result = gateway.execute(
            subject="research_agent",
            tool_id="write_file",
            tool_kwargs={},
        )

        assert (
            result.security_decision.decision
            == Decision.ALLOW_WITH_CONTROLS
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_medium_risk_audit_event_records_allow_with_controls_not_allow():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_medium_risk_gateway(tmp_dir)

        result = gateway.execute(
            subject="research_agent",
            tool_id="write_file",
            tool_kwargs={},
        )

        assert result.status == "SUCCESS"

        event = _read_last_audit_event(audit_log_path)

        assert event["decision"] == "ALLOW_WITH_CONTROLS"
        assert event["authorization"] == "ALLOW_WITH_CONTROLS"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_low_risk_is_still_reported_as_plain_allow():
    """
    Non-regression: this fix must not change LOW-risk operations,
    which should still be reported as plain ALLOW, not
    ALLOW_WITH_CONTROLS.
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        policy = {
            "version": "1.0",
            "permissions": [
                {
                    "subject": "research_agent",
                    "resource": "web_search",
                    "action": "search",
                    "scope": "public_web",
                    "risk_level": "LOW",
                    "approval": "none",
                }
            ],
        }

        policy_path = tmp_dir / "permissions.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                id="web_search",
                name="Web Search",
                purpose="Search the public web.",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                output_schema={"type": "string"},
                permissions=(
                    "research_agent:web_search:search:public_web",
                ),
                resource="web_search",
                action="search",
                scope="public_web",
                risk_level="LOW",
                error_handling={
                    "retryable": True,
                    "max_retries": 2,
                    "on_failure": "Surface the search error to the agent.",
                },
            )
        )

        security = SecurityDecisionPoint(
            policy_path=str(policy_path),
            audit_log_path=str(tmp_dir / "audit.jsonl"),
        )

        gateway = ToolGateway(security=security, registry=registry)
        gateway.register_executor(
            tool_id="web_search",
            executor=lambda: "results",
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={},
        )

        assert result.status == "SUCCESS"
        assert result.security_decision.decision == Decision.ALLOW
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
