"""
Regression test for caller-supplied metadata reaching the audit log.

`ToolInvocation.metadata` (e.g. a correlation ID or an approval
justification attached by the caller) used to be dropped silently
between `ToolRuntime.execute()` and the Security Layer: `metadata`
was never threaded through `ToolGateway.execute()` /
`SecurityDecisionPoint.evaluate()` / `evaluate_with_approval()`, so it
never reached `AuditLogger.record()` even though the audit event is
the one place this context is actually useful (correlating a security
decision back to the request that triggered it).

This test exercises the real ToolGateway/SecurityDecisionPoint/
AuditLogger stack (no mocks) end-to-end: it executes a tool with
explicit metadata and reads the resulting audit log back off disk to
confirm the metadata is actually present in the recorded event, and
that omitting metadata does not add a spurious key.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.registry.tool_registry import ToolDefinition, ToolRegistry


def _build_gateway(tmp_dir: Path):
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            id="web_search",
            name="Web Search",
            purpose="Search the public web.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
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

    gateway.register_executor(
        tool_id="web_search",
        executor=lambda query: "SUCCESS: " + query,
    )

    return gateway, audit_log_path


def _read_last_audit_event(audit_log_path: Path) -> dict:
    lines = audit_log_path.read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def test_caller_metadata_is_recorded_in_the_audit_log():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(tmp_dir)

        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={"query": "AI agents"},
            metadata={"correlation_id": "req-42", "reason": "user asked"},
        )

        assert result.status == "SUCCESS"

        event = _read_last_audit_event(audit_log_path)

        assert event.get("metadata") == {
            "correlation_id": "req-42",
            "reason": "user asked",
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_missing_metadata_does_not_add_a_spurious_audit_key():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(tmp_dir)

        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={"query": "AI agents"},
        )

        assert result.status == "SUCCESS"

        event = _read_last_audit_event(audit_log_path)

        assert "metadata" not in event
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_metadata_is_also_recorded_on_the_approval_path():
    """
    metadata must reach the audit log through
    evaluate_with_approval() too, not only the plain evaluate() path.
    """

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        gateway, audit_log_path = _build_gateway(tmp_dir)

        # web_search is LOW risk and does not require approval, but
        # supplying approved=True/approved_by routes through
        # evaluate_with_approval() regardless -- that's the code path
        # under test here, independent of whether approval is needed.
        result = gateway.execute(
            subject="research_agent",
            tool_id="web_search",
            tool_kwargs={"query": "AI agents"},
            approved=True,
            approved_by="human_operator",
            metadata={"correlation_id": "req-99"},
        )

        assert result.status == "SUCCESS"

        event = _read_last_audit_event(audit_log_path)

        assert event.get("metadata") == {"correlation_id": "req-99"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
