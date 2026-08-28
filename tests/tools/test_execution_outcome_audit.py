"""
Build Phase 13: the audit trail now records what actually happened
after a security decision, not just the decision itself.

SECURITY_SPEC.md's "Audit Logging" section requires audit records to
distinguish between "requested / authorized / executed / blocked /
failed ... This allows the system to determine whether an operation
was merely requested or actually executed." Before this phase, only
one audit event existed per ToolGateway.execute() call --
SecurityDecisionPoint._evaluate()'s own "security_decision" event,
written BEFORE execution was attempted. An ALLOW decision was audited
identically whether the private executor then actually ran
successfully, crashed, produced output that failed validation, or
never got the chance to run because the tool wasn't registered or had
no executor.

SecurityDecisionPoint.record_execution_outcome() closes that gap by
recording a second, distinct "execution_outcome" audit event, and
ToolGateway._finalize() calls it from every one of execute()'s return
points. These tests exercise the real ToolGateway/SecurityDecisionPoint
/AuditLogger stack end-to-end (no mocks) and read the resulting audit
log back off disk, plus unit-test record_execution_outcome() directly
for its validation and unknown-status fallback behavior.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from core.security.engine.authorization import Decision
from core.security.engine.security_decision import (
    SecurityDecision,
    SecurityDecisionPoint,
)
from core.security.engine.approval_gate import ApprovalDecision
from core.security.engine.authorization import (
    AuthorizationRequest,
    AuthorizationResult,
)
from core.security.engine.risk_engine import RiskAssessment
from core.security.engine.risk_engine import RiskLevel as EngineRiskLevel
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry


def _build_gateway(tmp_dir: Path, *, executor):
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="web_search",
            name="Web Search",
            purpose="Search the public web.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
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

    audit_log_path = tmp_dir / "audit.jsonl"

    security = SecurityDecisionPoint(
        policy_path="core/security/schemas/permissions.json",
        audit_log_path=str(audit_log_path),
    )

    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(tool_id="web_search", executor=executor)

    return gateway, audit_log_path


def _read_audit_events(audit_log_path: Path) -> list[dict]:
    lines = audit_log_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def _fake_security_decision(*, decision: Decision = Decision.ALLOW) -> SecurityDecision:
    """
    A minimal, real SecurityDecision (not a mock) built directly from
    the same dataclasses SecurityDecisionPoint itself produces, for
    unit-testing record_execution_outcome() in isolation from a full
    ToolGateway.execute() call.
    """

    request = AuthorizationRequest(
        subject="research_agent",
        resource="web_search",
        action="search",
        scope="public_web",
        risk_level="LOW",
    )
    authorization = AuthorizationResult(
        decision=decision,
        reason="test",
        request=request,
        effective_risk="LOW",
    )
    risk = RiskAssessment(level=EngineRiskLevel.LOW, reasons=("test",))
    approval = ApprovalDecision(required=False, approval_type="none", reason="test")

    return SecurityDecision(
        decision=decision,
        risk=risk,
        authorization=authorization,
        approval=approval,
    )


def test_successful_execution_is_recorded_as_executed():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(
            tmp_dir, executor=lambda query: "SUCCESS: " + query
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={"query": "AI agents"},
        )

        assert result.status == "SUCCESS"

        events = _read_audit_events(audit_log_path)
        assert [event["event"] for event in events] == [
            "security_decision",
            "execution_outcome",
        ]

        outcome = events[-1]
        assert outcome["execution_status"] == "executed"
        assert outcome["tool_status"] == "SUCCESS"
        assert outcome["subject"] == "research_agent"
        assert outcome["resource"] == "web_search"
        assert outcome["action"] == "search"
        assert outcome["scope"] == "public_web"
        assert outcome["tool_id"] == "web_search"
        assert outcome["decision"] == "ALLOW"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_executor_crash_is_recorded_as_failed():
    def _raising_executor(query):
        raise RuntimeError("boom")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(
            tmp_dir, executor=_raising_executor
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={"query": "AI agents"},
        )

        assert result.status == "ERROR"

        outcome = _read_audit_events(audit_log_path)[-1]
        assert outcome["event"] == "execution_outcome"
        assert outcome["execution_status"] == "failed"
        assert outcome["tool_status"] == "ERROR"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_invalid_output_is_recorded_as_failed():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # The executor runs and returns a value, but the tool's own
        # output_schema (registered as {"type": "string"} by
        # _build_gateway) rejects it -- INVALID_OUTPUT, which still
        # means the private executor genuinely ran.
        gateway, audit_log_path = _build_gateway(
            tmp_dir, executor=lambda query: 12345
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={"query": "AI agents"},
        )

        assert result.status == "INVALID_OUTPUT"

        outcome = _read_audit_events(audit_log_path)[-1]
        assert outcome["execution_status"] == "failed"
        assert outcome["tool_status"] == "INVALID_OUTPUT"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_invalid_input_never_reaches_the_executor_and_is_recorded_as_failed():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(
            tmp_dir, executor=lambda query: "SUCCESS: " + query
        )

        # Missing the required "query" field.
        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={},
        )

        assert result.status == "INVALID_INPUT"

        outcome = _read_audit_events(audit_log_path)[-1]
        assert outcome["event"] == "execution_outcome"
        assert outcome["execution_status"] == "failed"
        assert outcome["tool_status"] == "INVALID_INPUT"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_unregistered_tool_is_recorded_as_blocked():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(
            tmp_dir, executor=lambda query: "SUCCESS: " + query
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id="not_a_real_tool",
            tool_kwargs={"query": "AI agents"},
        )

        assert result.status == "DENIED"

        outcome = _read_audit_events(audit_log_path)[-1]
        assert outcome["event"] == "execution_outcome"
        assert outcome["execution_status"] == "blocked"
        assert outcome["tool_status"] == "DENIED"
        assert outcome["tool_id"] == "not_a_real_tool"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_missing_executor_is_recorded_as_blocked():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(
            tmp_dir, executor=lambda query: "SUCCESS: " + query
        )
        # Overwrite the registered executor mapping so none exists,
        # without going through register_executor's "already
        # registered" guard.
        gateway._executors.clear()

        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={"query": "AI agents"},
        )

        assert result.status == "DENIED"

        outcome = _read_audit_events(audit_log_path)[-1]
        assert outcome["execution_status"] == "blocked"
        assert outcome["tool_status"] == "DENIED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_metadata_is_recorded_on_the_execution_outcome_event_too():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(
            tmp_dir, executor=lambda query: "SUCCESS: " + query
        )

        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={"query": "AI agents"},
            metadata={"correlation_id": "req-7"},
        )

        assert result.status == "SUCCESS"

        outcome = _read_audit_events(audit_log_path)[-1]
        assert outcome["metadata"] == {"correlation_id": "req-7"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_record_execution_outcome_rejects_empty_tool_id():
    security = SecurityDecisionPoint(
        policy_path="core/security/schemas/permissions.json",
        audit_log_path=str(Path(tempfile.mkdtemp()) / "audit.jsonl"),
    )

    with pytest.raises(ValueError, match="tool_id"):
        security.record_execution_outcome(
            security_decision=_fake_security_decision(),
            tool_id="",
            tool_status="SUCCESS",
            summary="irrelevant",
        )


def test_record_execution_outcome_rejects_empty_tool_status():
    security = SecurityDecisionPoint(
        policy_path="core/security/schemas/permissions.json",
        audit_log_path=str(Path(tempfile.mkdtemp()) / "audit.jsonl"),
    )

    with pytest.raises(ValueError, match="tool_status"):
        security.record_execution_outcome(
            security_decision=_fake_security_decision(),
            tool_id="web_search",
            tool_status="",
            summary="irrelevant",
        )


def test_record_execution_outcome_degrades_unknown_tool_status_to_failed():
    """
    An unrecognized tool_status must never raise: this method records
    a fact about a tool call that already happened, and must never be
    the reason that call's own result fails to reach its caller.
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        audit_log_path = tmp_dir / "audit.jsonl"
        security = SecurityDecisionPoint(
            policy_path="core/security/schemas/permissions.json",
            audit_log_path=str(audit_log_path),
        )

        security.record_execution_outcome(
            security_decision=_fake_security_decision(),
            tool_id="web_search",
            tool_status="SOME_FUTURE_STATUS_NOT_YET_MAPPED",
            summary="irrelevant",
        )

        event = _read_audit_events(audit_log_path)[-1]
        assert event["execution_status"] == "failed"
        assert event["tool_status"] == "SOME_FUTURE_STATUS_NOT_YET_MAPPED"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
