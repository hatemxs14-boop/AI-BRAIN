"""
Tests for core.tools.implementations.read_project_memory_tool (Build
Phase 14).

Exercises the real ToolGateway/SecurityDecisionPoint stack (no mocks)
against a real, isolated MemoryStore -- LOW risk, auto-executing, like
every other read-only tool this project has (web_search, read_document,
read_webpage) -- plus a direct unit test of the executor factory
itself for input validation.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from core.memory.memory_store import MemoryEntry, MemoryStore
from core.security.engine.security_decision import SecurityDecisionPoint
from core.tools.engine.tool_gateway import ToolGateway
from core.tools.implementations.read_project_memory_tool import (
    READ_PROJECT_MEMORY_TOOL,
    READ_PROJECT_MEMORY_TOOL_ID,
    create_read_project_memory_executor,
)
from core.tools.registry.tool_registry import ToolRegistry


def _build_gateway(tmp_dir: Path, *, store: MemoryStore):
    registry = ToolRegistry()
    registry.register(READ_PROJECT_MEMORY_TOOL)

    policy = {
        "version": "1.0",
        "permissions": [
            {
                "subject": "research_agent",
                "resource": "project_memory",
                "action": "read",
                "scope": "workspace",
                "risk_level": "LOW",
                "approval": "none",
            }
        ],
    }
    policy_path = tmp_dir / "permissions.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    security = SecurityDecisionPoint(
        policy_path=str(policy_path),
        audit_log_path=str(tmp_dir / "audit.jsonl"),
    )

    gateway = ToolGateway(security=security, registry=registry)
    gateway.register_executor(
        tool_id=READ_PROJECT_MEMORY_TOOL_ID,
        executor=create_read_project_memory_executor(store),
    )

    return gateway


def test_read_project_memory_returns_matching_records_end_to_end():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        store.write(
            MemoryEntry(
                subject="research_agent",
                kind="note",
                content="The Eiffel Tower is in Paris.",
            )
        )

        gateway = _build_gateway(tmp_dir, store=store)

        result = gateway.execute(
            subject="research_agent",
            tool_id=READ_PROJECT_MEMORY_TOOL_ID,
            tool_kwargs={"query": "eiffel"},
        )

        assert result.status == "SUCCESS"
        (artifact,) = result.artifacts
        assert artifact["query"] == "eiffel"
        assert len(artifact["results"]) == 1
        assert artifact["results"][0]["content"] == "The Eiffel Tower is in Paris."
        assert artifact["results"][0]["verified"] is False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_read_project_memory_reports_the_real_verified_flag():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        record = store.write(
            MemoryEntry(
                subject="research_agent",
                kind="finding",
                content="Confirmed by two independent sources.",
            )
        )
        store.verify(record.id, verified_by="reviewer_agent")

        gateway = _build_gateway(tmp_dir, store=store)

        result = gateway.execute(
            subject="research_agent",
            tool_id=READ_PROJECT_MEMORY_TOOL_ID,
            tool_kwargs={"query": "confirmed"},
        )

        assert result.status == "SUCCESS"
        (artifact,) = result.artifacts
        verified_flags = {r["verified"] for r in artifact["results"]}
        assert verified_flags == {False, True}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_read_project_memory_returns_empty_results_when_nothing_matches():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        gateway = _build_gateway(tmp_dir, store=store)

        result = gateway.execute(
            subject="research_agent",
            tool_id=READ_PROJECT_MEMORY_TOOL_ID,
            tool_kwargs={"query": "nothing to find"},
        )

        assert result.status == "SUCCESS"
        (artifact,) = result.artifacts
        assert artifact["results"] == []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_read_project_memory_is_low_risk_and_auto_executes_without_approval():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        gateway = _build_gateway(tmp_dir, store=store)

        result = gateway.execute(
            subject="research_agent",
            tool_id=READ_PROJECT_MEMORY_TOOL_ID,
            tool_kwargs={"query": "anything"},
        )

        assert result.status == "SUCCESS"
        assert result.security_decision.decision.value == "ALLOW"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_create_read_project_memory_executor_rejects_empty_query():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        executor = create_read_project_memory_executor(store)

        with pytest.raises(ValueError, match="query"):
            executor(query="")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_create_read_project_memory_executor_rejects_non_positive_limit():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        store = MemoryStore(str(tmp_dir / "memory.jsonl"))
        with pytest.raises(ValueError, match="limit"):
            create_read_project_memory_executor(store, limit=0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
